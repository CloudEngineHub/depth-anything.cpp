// models.go — the curated DA3 model catalog and a lazy-loading registry.
// Big checkpoints (giant/nested are ~5 GB each) are loaded on first use and
// evicted LRU so we never hold more than -max-live at once.
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
)

// ModelInfo describes one selectable model and its capabilities.
type ModelInfo struct {
	Name      string   `json:"name"`
	Label     string   `json:"label"`
	Backbone  string   `json:"backbone"`
	File      string   `json:"-"`           // primary gguf
	File2     string   `json:"-"`           // metric branch (nested only)
	Pose      bool     `json:"pose"`        // has camera pose -> multi-view fusion
	Gaussians bool     `json:"gaussians"`   // has GS head (gaussian output)
	Metric    bool     `json:"metric"`      // absolute metre scale
	Modes     []string `json:"modes"`       // subset of {"points","gaussians"}
	Default   bool     `json:"default"`     // pre-selected in the UI
}

// catalog is the curated set fetched by scripts/fetch_models.sh.
var catalog = []ModelInfo{
	{Name: "da3-small", Label: "DA3 Small — fast", Backbone: "ViT-S",
		File: "depth-anything-small-f32.gguf", Pose: true, Modes: []string{"points"}},
	{Name: "da3-base", Label: "DA3 Base — default", Backbone: "ViT-B",
		File: "depth-anything-base-q4_k.gguf", Pose: true, Modes: []string{"points"}, Default: true},
	{Name: "da3-large", Label: "DA3 Large — quality", Backbone: "ViT-L",
		File: "depth-anything-large-f32.gguf", Pose: true, Modes: []string{"points"}},
	{Name: "da3-giant", Label: "DA3 Giant — points + gaussians", Backbone: "ViT-g",
		File: "depth-anything-giant-f32.gguf", Pose: true, Gaussians: true, Modes: []string{"points", "gaussians"}},
	{Name: "da3-nested-metric", Label: "DA3 Nested — metric scale", Backbone: "ViT-g + ViT-L",
		File: "depth-anything-nested-anyview.gguf", File2: "depth-anything-nested-metric.gguf",
		Pose: true, Metric: true, Modes: []string{"points"}},
}

func findModel(name string) *ModelInfo {
	for i := range catalog {
		if catalog[i].Name == name {
			return &catalog[i]
		}
	}
	return nil
}

// loaded is one resident model context plus a serialization lock.
type loaded struct {
	ctx   uintptr
	runMu sync.Mutex
}

// Registry lazily loads/evicts model contexts. All inference goes through
// WithModel, which holds the registry lock across load+run so eviction can never
// free a context that is in use (max-live defaults to 1 for the big checkpoints).
type Registry struct {
	api       *capi
	dir       string
	threads   int32
	maxLive   int
	mu        sync.Mutex
	live      map[string]*loaded
	order     []string // LRU, oldest first
}

func NewRegistry(api *capi, dir string, threads, maxLive int) *Registry {
	if maxLive < 1 {
		maxLive = 1
	}
	return &Registry{api: api, dir: dir, threads: int32(threads), maxLive: maxLive,
		live: map[string]*loaded{}}
}

// available returns the catalog entries whose files are present on disk.
func (r *Registry) available() []ModelInfo {
	var out []ModelInfo
	for _, m := range catalog {
		if r.present(&m) {
			out = append(out, m)
		}
	}
	return out
}

func (r *Registry) present(m *ModelInfo) bool {
	if _, err := os.Stat(filepath.Join(r.dir, m.File)); err != nil {
		return false
	}
	if m.File2 != "" {
		if _, err := os.Stat(filepath.Join(r.dir, m.File2)); err != nil {
			return false
		}
	}
	return true
}

// WithModel loads (or reuses) the named model and runs fn with its context held
// under both the registry lock and the model's run lock.
func (r *Registry) WithModel(name string, fn func(ctx uintptr, mi *ModelInfo) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	mi := findModel(name)
	if mi == nil {
		return fmt.Errorf("unknown model %q", name)
	}
	if !r.present(mi) {
		return fmt.Errorf("model %q not downloaded (run scripts/fetch_models.sh)", name)
	}
	l, ok := r.live[name]
	if !ok {
		for len(r.order) >= r.maxLive && len(r.order) > 0 {
			ev := r.order[0]
			r.order = r.order[1:]
			r.api.freeCtx(r.live[ev].ctx)
			delete(r.live, ev)
			log.Printf("model: evicted %s", ev)
		}
		var ctx uintptr
		if mi.File2 != "" {
			ctx = r.api.loadNested(filepath.Join(r.dir, mi.File), filepath.Join(r.dir, mi.File2), r.threads)
		} else {
			ctx = r.api.load(filepath.Join(r.dir, mi.File), r.threads)
		}
		if ctx == 0 {
			return fmt.Errorf("failed to load model %q", name)
		}
		l = &loaded{ctx: ctx}
		r.live[name] = l
		r.order = append(r.order, name)
		log.Printf("model: loaded %s", name)
	} else {
		// move to MRU
		for i, n := range r.order {
			if n == name {
				r.order = append(r.order[:i], r.order[i+1:]...)
				break
			}
		}
		r.order = append(r.order, name)
	}
	return fn(l.ctx, mi)
}
