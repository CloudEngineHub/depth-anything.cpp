#pragma once
// Sliding-window streaming reconstruction: turn a long ordered frame list into ONE
// coherent global colored point cloud, mirroring upstream da3_streaming. Each
// overlapping window is run through the existing fused multi-view pass
// (Engine::depth_pose_multi); consecutive windows are stitched into a single global
// frame with a weighted-Umeyama Sim3 solved on the overlap frames. No ggml/model
// changes — pure host orchestration on top of inference + back_project.
#include "model_loader.hpp"   // da::Config
#include <cstdint>
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
};

struct StreamCloud {
    std::vector<float>   xyz;     // 3N, GLOBAL frame (OpenCV axes)
    std::vector<uint8_t> rgb;     // 3N
    std::vector<float>   radius;  // N
    std::vector<int>     counts;  // per INPUT frame (length = #frames); build-up prefix sums
    int                  warnings = 0; // # windows that fell back to a degraded seam
};

// Stitch frame_paths (in order) into out. Returns false + sets err on failure.
bool stream_points(Engine& eng, const std::vector<std::string>& frame_paths,
                   const Config& cfg, const StreamParams& p,
                   StreamCloud& out, std::string& err);

} // namespace da
