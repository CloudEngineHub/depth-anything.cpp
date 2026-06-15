#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "image_io.hpp"
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace da {
enum class TaskMode { DEPTH, DEPTH_POSE, MULTIVIEW, RECONSTRUCT, NESTED_METRIC };

// Per-view result of the multi-view pipeline.
struct ViewResult {
    std::vector<float> depth, conf;  // each [H*W] row-major
    std::array<float,12> ext;        // 3x4 row-major
    std::array<float,9>  intr;       // 3x3 row-major
};

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads);
    const Config& config() const { return ml_.config(); }
    // M1: debug entry returning backbone features for out_layers (filled in T16).
    bool backbone_features(const std::vector<float>& input_image, int H, int W,
                           std::vector<std::vector<float>>& feats_out);
    // Full pipeline: image file -> preprocess -> backbone -> DualDPT depth head.
    // depth_out/conf_out are [H*W] row-major; H,W set to the processed dims.
    bool depth(const std::string& image_path, std::vector<float>& depth_out,
               std::vector<float>& conf_out, int& H, int& W);
    bool depth_image(const Image& img, std::vector<float>& depth_out,
                     std::vector<float>& conf_out, int& H, int& W);
    // Full pipeline incl pose. ext = 3x4 row-major (12), intr = 3x3 row-major (9).
    bool depth_pose(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                    std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    bool depth_pose_path(const std::string& image_path, std::vector<float>& depth, std::vector<float>& conf,
                         std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    // Multi-view: one backbone_mv pass over all images -> per-view depth + pose.
    // All images must preprocess to the same H,W (else returns false).
    bool depth_pose_multi(const std::vector<Image>& imgs, std::vector<ViewResult>& out, int& H, int& W);
private:
    ModelLoader ml_;
    Backend be_;
};
} // namespace da
