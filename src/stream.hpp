#pragma once
// Sliding-window streaming reconstruction: turn a long ordered frame list into ONE
// coherent global colored point cloud, mirroring upstream da3_streaming. Each
// overlapping window is run through the existing fused multi-view pass
// (Engine::depth_pose_multi); consecutive windows are stitched into a single global
// frame with a weighted-Umeyama Sim3 solved on the overlap frames. No ggml/model
// changes — pure host orchestration on top of inference + back_project.
#include "model_loader.hpp"   // da::Config
#include "engine.hpp"         // da::ViewResult
#include "icp.hpp"            // da::IcpParams (per-seam refinement, task B)
#include "posegraph.hpp"      // da::PoseGraphParams (loop closure, task C)
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace da {
class Engine;

struct StreamParams {
    int    chunk_size    = 12;        // frames per fused window (<= ~24)
    int    overlap       = 3;         // shared frames between consecutive windows
    double conf_pct      = 55.0;      // per-window confidence percentile gate
    float  point_size    = 1.2f;      // per-point radius multiplier
    int    global_budget = 6'000'000; // <=0 => unlimited; total emitted point cap
    int    min_overlap_pts = 50;      // degenerate guard for the Sim3 solve

    bool     icp_refine = false;      // task B: refine each seam with point-to-plane ICP
    IcpParams icp;                    // ICP params (used only when icp_refine)

    bool   metric       = false;      // rescale each window to METRES via the nested
                                      // metric branch (requires a load_nested() engine);
                                      // per-window robust-median scale_factor applied to
                                      // depth + pose translation before stitching.

    bool   loop_close   = false;      // task C: detect loops + Sim3 pose-graph optimize
    double loop_pos_thr = 1.5;        // propose a loop when window centres are within this (m)
    double loop_dir_thr = 0.3;        // ...and view directions agree (min cosine)
    int    loop_min_gap = 3;          // ...and are at least this many windows apart
    double loop_match_dist = 0.30;    // TIGHT correspondence gate for the loop measurement (m)
    int    loop_min_corr = 150;       // accept a loop only with >= this many ICP correspondences
    double loop_info    = 5.0;        // information weight of a loop edge vs a sequential edge (=1)
    PoseGraphParams pose_graph;       // optimizer params (used only when loop_close)
};

struct StreamCloud {
    std::vector<float>   xyz;     // 3N, GLOBAL frame (OpenCV axes)
    std::vector<uint8_t> rgb;     // 3N
    std::vector<float>   radius;  // N
    std::vector<int>     counts;  // per INPUT frame (length = #frames); build-up prefix sums
    int                  warnings = 0; // # windows that fell back to a degraded seam
    std::vector<float>   window_pos;   // 3*W: global camera centre of each window (post-optimization)
    std::vector<int>     window_mid_frame; // W: representative input-frame index per window
    std::vector<float>   frame_pos;    // 3*F: global camera centre per input frame (OpenCV axes)
    std::vector<float>   frame_fwd;    // 3*F: global unit view direction per input frame (OpenCV axes)
    int                  loops_found = 0; // # accepted loop-closure edges (task C)
    float                metric_scale = 0.f; // mean per-window metric scale applied (0 if not metric)
};

// Stitch frame_paths (in order) into out. Returns false + sets err on failure.
bool stream_points(Engine& eng, const std::vector<std::string>& frame_paths,
                   const Config& cfg, const StreamParams& p,
                   StreamCloud& out, std::string& err);

// A source of per-window fused views. Given the half-open global frame range
// [w0,w1), fill `views` (one ViewResult per frame in the window, carrying
// depth/conf/intr/ext) and `rgb` (per-view processed-resolution color, H*W*3
// row-major) and the common H,W. Returns false + sets err on failure. This is
// the ONLY dependency of the stitcher on inference: production wraps the Engine;
// tests inject synthetic surface-rendered views (true depth => isolates the
// stitching error from neural-depth error).
using WindowSource = std::function<bool(int w0, int w1,
                                        std::vector<ViewResult>& views,
                                        std::vector<std::vector<uint8_t>>& rgb,
                                        int& H, int& W, std::string& err)>;

// Pure sliding-window stitch/fuse over `nframes` frames, pulling each window
// from `source`. stream_points() is a thin wrapper that builds `source` from an
// Engine + frame paths. Kept public so tests can drive the exact production
// stitching logic with synthetic ground truth.
bool stream_points_core(int nframes, const StreamParams& p,
                        const WindowSource& source, StreamCloud& out, std::string& err);

} // namespace da
