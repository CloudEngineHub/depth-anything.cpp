// scenes.go — sample-scene listing + the async video->scene pipeline. A video is
// turned into ONE coherent cloud via sliding-window streaming (overlapping fused
// windows stitched by a weighted-Umeyama Sim3), then sliced by frame into acc_*
// "build-up" steps for the viewer.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type manifestStep struct {
	Splat  string   `json:"splat"`
	Images []string `json:"images"`
	N      int      `json:"n"`
	Label  string   `json:"label,omitempty"`
}

type sceneManifest struct {
	Model string         `json:"model"`
	Mode  string         `json:"mode"`
	Steps []manifestStep `json:"steps"`
}

type sceneInfo struct {
	Name   string `json:"name"`
	Label  string `json:"label"`
	Steps  int    `json:"steps"`
	Thumb  string `json:"thumb"`
	Source string `json:"source"`
	Model  string `json:"model"`
	Mode   string `json:"mode"`
}

type sceneJob struct {
	State string `json:"state"` // running | done | error
	Total int    `json:"total"`
	Done  int    `json:"done"`
	Kept  int    `json:"kept"`
	Scene string `json:"scene,omitempty"`
	Err   string `json:"error,omitempty"`
}

var slugRe = regexp.MustCompile(`[^a-z0-9_-]+`)

func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "-")
	s = slugRe.ReplaceAllString(s, "")
	if s == "" {
		s = "scene"
	}
	if len(s) > 48 {
		s = s[:48]
	}
	return s
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// GET /api/scenes
func (s *server) handleScenes(w http.ResponseWriter, r *http.Request) {
	ents, _ := os.ReadDir(s.scenesDir)
	out := []sceneInfo{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		mf := filepath.Join(s.scenesDir, e.Name(), "manifest.json")
		b, err := os.ReadFile(mf)
		if err != nil {
			continue
		}
		var m sceneManifest
		if json.Unmarshal(b, &m) != nil || len(m.Steps) == 0 {
			continue
		}
		thumb := ""
		if len(m.Steps[0].Images) > 0 {
			thumb = "/scenes-assets/" + e.Name() + "/" + m.Steps[0].Images[0]
		}
		src := "baked"
		if _, err := os.Stat(filepath.Join(s.scenesDir, e.Name(), ".uploaded")); err == nil {
			src = "uploaded"
		}
		out = append(out, sceneInfo{
			Name: e.Name(), Label: prettify(e.Name()), Steps: len(m.Steps),
			Thumb: thumb, Source: src, Model: m.Model, Mode: m.Mode,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	writeJSON(w, map[string]any{"scenes": out})
}

func prettify(s string) string {
	return strings.Title(strings.ReplaceAll(strings.ReplaceAll(s, "-", " "), "_", " "))
}

// GET /api/scene/status/{job}
func (s *server) handleSceneStatus(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/scene/status/")
	s.jobsMu.Lock()
	j, ok := s.jobs[id]
	s.jobsMu.Unlock()
	if !ok {
		http.Error(w, "no such job", 404)
		return
	}
	writeJSON(w, j)
}

// POST /api/scene/from-video  (multipart: video, name, model, mode, max_frames, conf_pct, point_size)
func (s *server) handleSceneFromVideo(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(512 << 20); err != nil {
		http.Error(w, "bad form: "+err.Error(), 400)
		return
	}
	file, hdr, err := r.FormFile("video")
	if err != nil {
		http.Error(w, "missing video", 400)
		return
	}
	defer file.Close()
	name := slug(r.FormValue("name"))
	if name == "scene" && hdr != nil {
		name = slug(strings.TrimSuffix(hdr.Filename, filepath.Ext(hdr.Filename)))
	}
	model := r.FormValue("model")
	if model == "" {
		model = "da3-base"
	}
	mode := r.FormValue("mode")
	if mode == "" {
		mode = "points"
	}
	maxFrames := atoiDefault(r.FormValue("max_frames"), 60)
	if maxFrames < 2 {
		maxFrames = 2
	}
	if maxFrames > 600 {
		maxFrames = 600
	}
	chunkSize := atoiDefault(r.FormValue("chunk_size"), 12)
	if chunkSize < 2 {
		chunkSize = 2
	}
	if chunkSize > 24 {
		chunkSize = 24
	}
	overlap := atoiDefault(r.FormValue("overlap"), 3)
	if overlap < 0 {
		overlap = 0
	}
	if overlap > chunkSize-1 {
		overlap = chunkSize - 1
	}
	fps := atofDefault(r.FormValue("fps"), 6)
	if fps < 0.5 {
		fps = 0.5
	}
	if fps > 30 {
		fps = 30
	}
	confPct := atofDefault(r.FormValue("conf_pct"), 55)
	ptSize := float32(atofDefault(r.FormValue("point_size"), 1.2))

	// save upload
	jobDir := filepath.Join(s.workDir, "uploads", name)
	_ = os.MkdirAll(jobDir, 0o755)
	vpath := filepath.Join(jobDir, "input"+filepath.Ext(hdr.Filename))
	out, err := os.Create(vpath)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		http.Error(w, err.Error(), 500)
		return
	}
	out.Close()

	s.jobsMu.Lock()
	s.jobs[name] = &sceneJob{State: "running"}
	s.jobsMu.Unlock()

	go func() {
		s.bakeSem <- struct{}{}
		defer func() { <-s.bakeSem }()
		err := s.bakeVideo(name, vpath, jobDir, model, mode, maxFrames, chunkSize, overlap, fps, confPct, ptSize)
		s.jobsMu.Lock()
		j := s.jobs[name]
		if err != nil {
			j.State, j.Err = "error", err.Error()
		} else {
			j.State, j.Scene = "done", name
		}
		s.jobsMu.Unlock()
	}()

	writeJSON(w, map[string]string{"job": name, "name": name})
}

// bakeVideo: ffmpeg -> frames -> DA3 -> acc_*.splat + thumbnails + manifest.
func (s *server) bakeVideo(name, vpath, jobDir, model, mode string, maxFrames, chunkSize, overlap int, fps, confPct float64, ptSize float32) error {
	framesDir := filepath.Join(jobDir, "frames")
	_ = os.RemoveAll(framesDir)
	_ = os.MkdirAll(framesDir, 0o755)
	// Bounded extraction (fps cap + downscale), then even stride to maxFrames.
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", vpath,
		"-vf", fmt.Sprintf("fps=%g,scale=640:-2", fps), filepath.Join(framesDir, "all%05d.jpg"))
	if b, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg: %v: %s", err, string(b))
	}
	all, _ := filepath.Glob(filepath.Join(framesDir, "all*.jpg"))
	sort.Strings(all)
	if len(all) < 2 {
		return fmt.Errorf("video produced %d frames (need >=2)", len(all))
	}
	stride := (len(all) + maxFrames - 1) / maxFrames
	if stride < 1 {
		stride = 1
	}
	var sel []string
	for i := 0; i < len(all) && len(sel) < maxFrames; i += stride {
		sel = append(sel, all[i])
	}

	sceneDir := filepath.Join(s.scenesDir, name)
	_ = os.RemoveAll(sceneDir)
	_ = os.MkdirAll(sceneDir, 0o755)
	_ = os.WriteFile(filepath.Join(sceneDir, ".uploaded"), []byte("1"), 0o644)

	s.setJob(name, func(j *sceneJob) { j.Total = len(sel) })

	mi := findModel(model)
	if mi == nil {
		return fmt.Errorf("unknown model %q", model)
	}

	// Gaussian mode: single representative (middle) frame -> GS splat.
	if mode == "gaussians" {
		if !mi.Gaussians {
			return fmt.Errorf("model %q has no gaussian head (use da3-giant)", model)
		}
		mid := sel[len(sel)/2]
		thumb := "view_0.jpg"
		_ = copyScaled(mid, filepath.Join(sceneDir, thumb), 360)
		// GS head is fixed at a 224x224 (16x16 patch) input.
		g224 := filepath.Join(jobDir, "g224.jpg")
		if e := squareResize(mid, g224, 224); e != nil {
			return e
		}
		var splat []byte
		err := s.infer(func() error {
			return s.reg.WithModel(model, func(ctx uintptr, _ *ModelInfo) error {
				g, e := s.api.Gaussians(ctx, g224)
				if e != nil {
					return e
				}
				splat = gaussiansToSplat(g, s.maxSplats)
				return nil
			})
		})
		if err != nil {
			return err
		}
		if e := os.WriteFile(filepath.Join(sceneDir, "acc_full.splat"), splat, 0o644); e != nil {
			return e
		}
		s.setJob(name, func(j *sceneJob) { j.Done, j.Kept = len(sel), 1 })
		return writeManifest(sceneDir, sceneManifest{Model: model, Mode: mode, Steps: []manifestStep{
			{Splat: "acc_full.splat", Images: []string{thumb}, N: 1, Label: "gaussians · single frame"}}})
	}

	// Point-cloud mode: sliding-window streaming over all selected frames.
	if !mi.Pose {
		return fmt.Errorf("model %q has no camera pose; cannot fuse frames", model)
	}
	// thumbnails first (so progress feels live)
	thumbs := make([]string, len(sel))
	for i, f := range sel {
		thumbs[i] = fmt.Sprintf("view_%d.jpg", i)
		_ = copyScaled(f, filepath.Join(sceneDir, thumbs[i]), 360)
	}

	var cloud *Cloud
	err := s.infer(func() error {
		return s.reg.WithModel(model, func(ctx uintptr, _ *ModelInfo) error {
			c, e := s.api.PointsStream(ctx, sel, chunkSize, overlap, confPct, ptSize, s.maxSplats)
			if e != nil {
				return e
			}
			cloud = c
			return nil
		})
	})
	if err != nil {
		return err
	}
	s.setJob(name, func(j *sceneJob) { j.Done = len(sel); j.Kept = len(sel) })

	// Build-up: acc_k.splat = points from the first k views (frames outer). For long
	// clips, stride the steps so the viewer gets ~30 progressive reveals (always
	// including the final full cloud), not one file per frame.
	steps := []manifestStep{}
	prefix := 0
	capN := s.maxSplats
	nv := cloud.N0(len(sel))
	bstep := (nv + 29) / 30
	if bstep < 1 {
		bstep = 1
	}
	for k := 1; k <= nv; k++ {
		prefix += int(cloud.Counts[k-1])
		if k < 2 {
			continue
		}
		if k%bstep != 0 && k != nv {
			continue
		}
		n := prefix
		if capN > 0 && n > capN {
			n = capN
		}
		fn := fmt.Sprintf("acc_%d.splat", k)
		if e := os.WriteFile(filepath.Join(sceneDir, fn), pointsToSplat(cloud, n, 1.0), 0o644); e != nil {
			return e
		}
		label := ""
		if k == nv {
			label = fmt.Sprintf("full %d-view cloud · %d pts", k, prefix)
		}
		steps = append(steps, manifestStep{Splat: fn, Images: thumbs[:k], N: k, Label: label})
	}
	return writeManifest(sceneDir, sceneManifest{Model: model, Mode: mode, Steps: steps})
}

// N0 guards Counts length vs the number of selected views.
func (c *Cloud) N0(nViews int) int {
	if len(c.Counts) < nViews {
		return len(c.Counts)
	}
	return nViews
}

func writeManifest(dir string, m sceneManifest) error {
	b, _ := json.MarshalIndent(m, "", "  ")
	return os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0o644)
}

func copyScaled(src, dst string, px int) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", src,
		"-vf", fmt.Sprintf("scale=%d:-2", px), dst)
	return cmd.Run()
}

// squareResize center-crops to a px*px square (the GS head's fixed input size).
func squareResize(src, dst string, px int) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", src,
		"-vf", fmt.Sprintf("scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d", px, px, px, px), dst)
	return cmd.Run()
}

func (s *server) setJob(name string, fn func(*sceneJob)) {
	s.jobsMu.Lock()
	if j := s.jobs[name]; j != nil {
		fn(j)
	}
	s.jobsMu.Unlock()
}

func atoiDefault(s string, d int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return v
	}
	return d
}

func atofDefault(s string, d float64) float64 {
	if v, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
		return v
	}
	return d
}
