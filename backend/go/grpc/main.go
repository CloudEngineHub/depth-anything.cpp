// Command da3-grpc is a standalone LocalAI gRPC Backend server that wraps the
// Depth Anything 3 ggml port (the da3 purego binding) and exposes it via the
// LocalAI Backend gRPC contract.
//
// Depth has no native OpenAI endpoint, so the model is exposed two ways:
//
//   - GenerateImage(src, dst): run depth on the src image and write a
//     min-max-normalised grayscale depth PNG to dst.
//   - Predict(images[0]): run depth+pose and return a JSON blob with depth
//     dimensions, depth stats and the camera extrinsics/intrinsics.
//
// The C side is NOT reentrant (shared ggml graph allocator), so all inference
// is serialized behind a single mutex, mirroring LocalAI's base.SingleThread.
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"net"
	"os"
	"sync"

	"github.com/mudler/depth-anything-cpp/backend/go/da3"
	"github.com/mudler/depth-anything-cpp/backend/go/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// server implements proto.BackendServer for the Depth Anything 3 backend.
type server struct {
	proto.UnimplementedBackendServer

	soPath string

	mu    sync.Mutex // serializes LoadModel + inference (C side non-reentrant)
	model *da3.Model
}

// Health returns OK once the process is up.
func (s *server) Health(_ context.Context, _ *proto.HealthMessage) (*proto.Reply, error) {
	return &proto.Reply{Message: []byte("OK")}, nil
}

// LoadModel opens the shared library (once) and loads the GGUF model named by
// opts.ModelFile (falling back to opts.Model).
func (s *server) LoadModel(_ context.Context, opts *proto.ModelOptions) (*proto.Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := da3.Open(s.soPath); err != nil {
		return &proto.Result{Success: false, Message: err.Error()}, nil
	}

	modelPath := opts.GetModelFile()
	if modelPath == "" {
		modelPath = opts.GetModel()
	}
	if modelPath == "" {
		return &proto.Result{Success: false, Message: "no model file provided (ModelFile/Model empty)"}, nil
	}

	threads := int(opts.GetThreads())
	if threads <= 0 {
		threads = 1
	}

	m, err := da3.Load(modelPath, threads)
	if err != nil {
		return &proto.Result{Success: false, Message: err.Error()}, nil
	}

	if s.model != nil {
		s.model.Close()
	}
	s.model = m
	log.Printf("loaded model %q (abi=%d, threads=%d)", modelPath, da3.ABIVersion(), threads)
	return &proto.Result{Success: true, Message: "loaded " + modelPath}, nil
}

// depthResult is the JSON payload returned by Predict.
type depthResult struct {
	DepthW     int         `json:"depth_w"`
	DepthH     int         `json:"depth_h"`
	DepthMin   float32     `json:"depth_min"`
	DepthMax   float32     `json:"depth_max"`
	Extrinsics [12]float32 `json:"extrinsics"` // 3x4 row-major
	Intrinsics [9]float32  `json:"intrinsics"` // 3x3 row-major
}

// Predict runs depth+pose on the first supplied image and returns depth
// statistics + camera pose as JSON in Reply.Message. The image is taken from
// PredictOptions.Images[0], which LocalAI passes either as a filesystem path or
// a base64-encoded payload.
func (s *server) Predict(_ context.Context, opts *proto.PredictOptions) (*proto.Reply, error) {
	imgs := opts.GetImages()
	if len(imgs) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Predict: no image supplied in Images[]")
	}

	imgPath, cleanup, err := materializeImage(imgs[0])
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "Predict: %v", err)
	}
	defer cleanup()

	depth, h, w, ext, intr, err := s.runDepthPose(imgPath)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "Predict: %v", err)
	}

	dmin, dmax := minMax(depth)
	res := depthResult{
		DepthW: w, DepthH: h,
		DepthMin: dmin, DepthMax: dmax,
		Extrinsics: ext, Intrinsics: intr,
	}
	payload, err := json.Marshal(res)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "Predict: marshal: %v", err)
	}
	return &proto.Reply{Message: payload}, nil
}

// GenerateImage runs depth on req.Src and writes a normalised grayscale depth
// PNG to req.Dst.
func (s *server) GenerateImage(_ context.Context, req *proto.GenerateImageRequest) (*proto.Result, error) {
	if req.GetSrc() == "" {
		return &proto.Result{Success: false, Message: "GenerateImage: empty src"}, nil
	}
	if req.GetDst() == "" {
		return &proto.Result{Success: false, Message: "GenerateImage: empty dst"}, nil
	}

	depth, h, w, _, _, err := s.runDepthPose(req.GetSrc())
	if err != nil {
		return &proto.Result{Success: false, Message: err.Error()}, nil
	}

	if err := writeDepthPNG(req.GetDst(), depth, h, w); err != nil {
		return &proto.Result{Success: false, Message: err.Error()}, nil
	}
	return &proto.Result{Success: true, Message: "wrote " + req.GetDst()}, nil
}

// Status reports the backend readiness.
func (s *server) Status(_ context.Context, _ *proto.HealthMessage) (*proto.StatusResponse, error) {
	state := proto.StatusResponse_UNINITIALIZED
	s.mu.Lock()
	if s.model != nil {
		state = proto.StatusResponse_READY
	}
	s.mu.Unlock()
	return &proto.StatusResponse{State: state}, nil
}

// runDepthPose serializes inference behind the mutex and runs depth+pose.
func (s *server) runDepthPose(imgPath string) ([]float32, int, int, [12]float32, [9]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model == nil {
		return nil, 0, 0, [12]float32{}, [9]float32{}, fmt.Errorf("model not loaded (call LoadModel first)")
	}
	return s.model.DepthPose(imgPath)
}

// materializeImage returns a filesystem path for an image argument that may be
// either an existing path or a base64-encoded payload. When the input is
// base64 it is decoded into a temp file; cleanup removes it (no-op for a path).
func materializeImage(arg string) (path string, cleanup func(), err error) {
	cleanup = func() {}
	if _, statErr := os.Stat(arg); statErr == nil {
		return arg, cleanup, nil
	}
	// Strip an optional data URL prefix.
	b64 := arg
	if i := indexComma(b64); i >= 0 && hasDataPrefix(b64) {
		b64 = b64[i+1:]
	}
	data, decErr := base64.StdEncoding.DecodeString(b64)
	if decErr != nil {
		return "", cleanup, fmt.Errorf("image is neither an existing path nor valid base64: %v", decErr)
	}
	f, tErr := os.CreateTemp("", "da3-input-*.png")
	if tErr != nil {
		return "", cleanup, tErr
	}
	if _, wErr := f.Write(data); wErr != nil {
		f.Close()
		os.Remove(f.Name())
		return "", cleanup, wErr
	}
	f.Close()
	name := f.Name()
	return name, func() { os.Remove(name) }, nil
}

func hasDataPrefix(s string) bool {
	return len(s) >= 5 && s[:5] == "data:"
}

func indexComma(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			return i
		}
	}
	return -1
}

// writeDepthPNG min-max normalises a depth map and writes it as an 8-bit
// grayscale PNG. Near = bright (255), far = dark (0), matching the usual
// depth-map convention for inverse-depth-like outputs.
func writeDepthPNG(dst string, depth []float32, h, w int) error {
	if h <= 0 || w <= 0 || len(depth) < h*w {
		return fmt.Errorf("writeDepthPNG: bad dims h=%d w=%d len=%d", h, w, len(depth))
	}
	dmin, dmax := minMax(depth)
	span := dmax - dmin
	if span <= 0 || math.IsNaN(float64(span)) {
		span = 1
	}
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := depth[y*w+x]
			n := (v - dmin) / span // 0..1
			if math.IsNaN(float64(n)) {
				n = 0
			}
			if n < 0 {
				n = 0
			} else if n > 1 {
				n = 1
			}
			img.Pix[y*img.Stride+x] = uint8(n * 255)
		}
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func minMax(v []float32) (mn, mx float32) {
	if len(v) == 0 {
		return 0, 0
	}
	mn, mx = v[0], v[0]
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			continue
		}
		if x < mn {
			mn = x
		}
		if x > mx {
			mx = x
		}
	}
	return mn, mx
}

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "gRPC listen address")
	soFlag := flag.String("so", "", "path to libdepthanything.so (overrides DA3_SO env)")
	flag.Parse()

	soPath := *soFlag
	if soPath == "" {
		soPath = os.Getenv("DA3_SO")
	}
	if soPath == "" {
		log.Fatal("no shared library path: set DA3_SO or pass --so")
	}

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *addr, err)
	}

	srv := grpc.NewServer()
	proto.RegisterBackendServer(srv, &server{soPath: soPath})
	log.Printf("da3-grpc serving on %s (so=%s)", *addr, soPath)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
