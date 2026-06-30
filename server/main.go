// da3 demo server — browse sample scenes or upload a video/photos and turn them
// into a coherent point cloud (any pose-capable DA3 model) or gaussians (giant),
// rendered in the embedded WebGL viewer. Mirrors free-splatter's demo UX.
//
//	go run . -lib ../build/libdepthanything.so -models-dir ../models
package main

import (
	"embed"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

type server struct {
	api       *capi
	reg       *Registry
	scenesDir string
	workDir   string
	maxSplats int

	inferMu sync.Mutex // serializes ALL inference (single ggml backend)

	// async video bakes: at most one at a time
	jobsMu  sync.Mutex
	jobs    map[string]*sceneJob
	bakeSem chan struct{}

	// in-memory LRU of reconstruct results (id -> splat bytes)
	resMu   sync.Mutex
	results map[string][]byte
	order   []string
	counter int64
}

func (s *server) infer(fn func() error) error {
	s.inferMu.Lock()
	defer s.inferMu.Unlock()
	return fn()
}

func main() {
	addr := flag.String("addr", ":8794", "listen address")
	lib := flag.String("lib", "../build/libdepthanything.so", "path to libdepthanything.so")
	modelsDir := flag.String("models-dir", "../models", "directory with the .gguf weights")
	scenesDir := flag.String("scenes-dir", "../.cache/da3-scenes", "directory of baked/uploaded scenes")
	workDir := flag.String("work-dir", "/tmp/da3-demo", "scratch dir for uploads/frames")
	threads := flag.Int("threads", 12, "inference threads")
	maxLive := flag.Int("max-live", 1, "max resident model contexts (LRU evicted)")
	maxSplats := flag.Int("max-splats", 1500000, "cap splats per output")
	flag.Parse()

	api, err := openCAPI(*lib)
	if err != nil {
		log.Fatalf("load library: %v", err)
	}
	log.Printf("libdepthanything ABI %d", api.abiVersion())

	_ = os.MkdirAll(*scenesDir, 0o755)
	_ = os.MkdirAll(*workDir, 0o755)

	s := &server{
		api:       api,
		reg:       NewRegistry(api, *modelsDir, *threads, *maxLive),
		scenesDir: *scenesDir,
		workDir:   *workDir,
		maxSplats: *maxSplats,
		jobs:      map[string]*sceneJob{},
		bakeSem:   make(chan struct{}, 1),
		results:   map[string][]byte{},
	}

	avail := s.reg.available()
	if len(avail) == 0 {
		log.Printf("WARNING: no models found in %s — run scripts/fetch_models.sh", *modelsDir)
	} else {
		names := make([]string, len(avail))
		for i, m := range avail {
			names[i] = m.Name
		}
		log.Printf("models available: %v", names)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/reconstruct", s.handleReconstruct)
	mux.HandleFunc("/api/splat/", s.handleSplat)
	mux.HandleFunc("/api/scenes", s.handleScenes)
	mux.HandleFunc("/api/scene/from-video", s.handleSceneFromVideo)
	mux.HandleFunc("/api/scene/status/", s.handleSceneStatus)
	mux.Handle("/scenes-assets/", http.StripPrefix("/scenes-assets/", http.FileServer(http.Dir(*scenesDir))))

	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("/", http.FileServer(http.FS(sub)))

	log.Printf("da3 demo on http://localhost%s  (scenes=%s)", *addr, *scenesDir)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

// GET /api/models
func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	avail := s.reg.available()
	def := ""
	for _, m := range avail {
		if m.Default {
			def = m.Name
		}
	}
	if def == "" && len(avail) > 0 {
		def = avail[0].Name
	}
	writeJSON(w, map[string]any{"models": avail, "default": def})
}

// GET /api/splat/{id}
func (s *server) handleSplat(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Path[len("/api/splat/"):]
	s.resMu.Lock()
	b, ok := s.results[id]
	s.resMu.Unlock()
	if !ok {
		http.Error(w, "no such result", 404)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(b)
}

func (s *server) storeResult(b []byte) string {
	s.resMu.Lock()
	defer s.resMu.Unlock()
	s.counter++
	id := "r" + strconv.FormatInt(s.counter, 10)
	s.results[id] = b
	s.order = append(s.order, id)
	for len(s.order) > 16 {
		old := s.order[0]
		s.order = s.order[1:]
		delete(s.results, old)
	}
	return id
}

// POST /api/reconstruct  (multipart: images[], model, mode)
func (s *server) handleReconstruct(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(256 << 20); err != nil {
		http.Error(w, "bad form: "+err.Error(), 400)
		return
	}
	model := r.FormValue("model")
	if model == "" {
		model = "da3-base"
	}
	mode := r.FormValue("mode")
	if mode == "" {
		mode = "points"
	}
	confPct := atofDefault(r.FormValue("conf_pct"), 55)
	ptSize := float32(atofDefault(r.FormValue("point_size"), 1.2))

	files := r.MultipartForm.File["images"]
	if len(files) == 0 {
		http.Error(w, "no images", 400)
		return
	}
	dir := filepath.Join(s.workDir, "recon", strconv.FormatInt(time.Now().UnixNano(), 10))
	_ = os.MkdirAll(dir, 0o755)
	paths := make([]string, 0, len(files))
	for i, fh := range files {
		src, err := fh.Open()
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		p := filepath.Join(dir, fmt.Sprintf("img_%03d%s", i, filepath.Ext(fh.Filename)))
		dst, err := os.Create(p)
		if err != nil {
			src.Close()
			http.Error(w, err.Error(), 500)
			return
		}
		io.Copy(dst, src)
		dst.Close()
		src.Close()
		paths = append(paths, p)
	}

	t0 := time.Now()
	var splat []byte
	var nOut int
	err := s.infer(func() error {
		return s.reg.WithModel(model, func(ctx uintptr, mi *ModelInfo) error {
			if mode == "gaussians" {
				if !mi.Gaussians {
					return fmt.Errorf("model %q has no gaussian head (use da3-giant)", model)
				}
				// GS head is fixed at a 224x224 (16x16 patch) input.
				g224 := paths[0] + ".g224.jpg"
				if e := squareResize(paths[0], g224, 224); e != nil {
					return e
				}
				g, e := s.api.Gaussians(ctx, g224)
				if e != nil {
					return e
				}
				splat = gaussiansToSplat(g, s.maxSplats)
				nOut = g.N
				return nil
			}
			c, e := s.api.PointsMulti(ctx, paths, confPct, ptSize)
			if e != nil {
				return e
			}
			n := c.N
			if s.maxSplats > 0 && n > s.maxSplats {
				n = s.maxSplats
			}
			splat = pointsToSplat(c, n, 1.0)
			nOut = n
			return nil
		})
	})
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	id := s.storeResult(splat)
	writeJSON(w, map[string]any{
		"id": id, "model": model, "mode": mode,
		"n": nOut, "size": len(splat), "seconds": time.Since(t0).Seconds(),
	})
}
