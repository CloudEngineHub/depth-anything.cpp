#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <memory>
#include <string>
#include <vector>

namespace da {
enum class TaskMode { DEPTH, DEPTH_POSE, MULTIVIEW, RECONSTRUCT, NESTED_METRIC };

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads);
    const Config& config() const { return ml_.config(); }
    // M1: debug entry returning backbone features for out_layers (filled in T16).
    bool backbone_features(const std::vector<float>& input_image, int H, int W,
                           std::vector<std::vector<float>>& feats_out);
private:
    ModelLoader ml_;
    Backend be_;
};
} // namespace da
