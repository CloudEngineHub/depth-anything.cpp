package main

import (
	"context"
	"net"
	"os"
	"testing"
	"time"

	"github.com/mudler/depth-anything-cpp/backend/go/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestSmokeBackend dials an in-process gRPC server and exercises
// Health -> LoadModel -> GenerateImage. It is skipped unless DA3_SO,
// DA3_TEST_GGUF and DA3_TEST_IMAGE point at a real shared library, model and
// input image.
func TestSmokeBackend(t *testing.T) {
	soPath := os.Getenv("DA3_SO")
	gguf := os.Getenv("DA3_TEST_GGUF")
	img := os.Getenv("DA3_TEST_IMAGE")
	if soPath == "" || gguf == "" || img == "" {
		t.Skip("set DA3_SO, DA3_TEST_GGUF and DA3_TEST_IMAGE to run the smoke test")
	}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	proto.RegisterBackendServer(srv, &server{soPath: soPath})
	go srv.Serve(lis)
	defer srv.Stop()

	conn, err := grpc.NewClient(lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	client := proto.NewBackendClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Health
	hr, err := client.Health(ctx, &proto.HealthMessage{})
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if string(hr.GetMessage()) != "OK" {
		t.Fatalf("Health: got %q want OK", hr.GetMessage())
	}

	// LoadModel
	lr, err := client.LoadModel(ctx, &proto.ModelOptions{ModelFile: gguf, Threads: 4})
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if !lr.GetSuccess() {
		t.Fatalf("LoadModel failed: %s", lr.GetMessage())
	}

	// Status should now be READY
	st, err := client.Status(ctx, &proto.HealthMessage{})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if st.GetState() != proto.StatusResponse_READY {
		t.Fatalf("Status: got %v want READY", st.GetState())
	}

	// GenerateImage -> depth PNG
	dst := t.TempDir() + "/da3_depth.png"
	gr, err := client.GenerateImage(ctx, &proto.GenerateImageRequest{Src: img, Dst: dst})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if !gr.GetSuccess() {
		t.Fatalf("GenerateImage failed: %s", gr.GetMessage())
	}
	fi, err := os.Stat(dst)
	if err != nil {
		t.Fatalf("output PNG missing: %v", err)
	}
	if fi.Size() == 0 {
		t.Fatalf("output PNG is empty")
	}
	t.Logf("smoke OK: Health=OK LoadModel=Success GenerateImage wrote %d bytes to %s", fi.Size(), dst)

	// Predict -> JSON stats
	pr, err := client.Predict(ctx, &proto.PredictOptions{Images: []string{img}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(pr.GetMessage()) == 0 {
		t.Fatalf("Predict returned empty message")
	}
	t.Logf("Predict JSON: %s", pr.GetMessage())
}
