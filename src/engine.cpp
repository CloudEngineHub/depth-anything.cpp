#include "engine.hpp"
#include "common.hpp"
#include "dino_backbone.hpp"

namespace da {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads){
    std::unique_ptr<Engine> e(new Engine());
    if (!e->ml_.load(path)) { DA_LOG("engine: load failed"); return nullptr; }
    e->be_.set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) { DA_LOG("engine: offload failed"); return nullptr; }
    return e;
}
bool Engine::backbone_features(const std::vector<float>& input_chw, int H, int W,
                               std::vector<std::vector<float>>& feats_out){
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> cams;
    return bb.forward(input_chw, H, W, feats_out, cams);
}
} // namespace da
