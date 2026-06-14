#include "engine.hpp"
#include "common.hpp"

namespace da {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads){
    std::unique_ptr<Engine> e(new Engine());
    if (!e->ml_.load(path)) { DA_LOG("engine: load failed"); return nullptr; }
    e->be_.set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) { DA_LOG("engine: offload failed"); return nullptr; }
    return e;
}
bool Engine::backbone_features(const std::vector<float>&, int, int,
                               std::vector<std::vector<float>>&){
    DA_LOG("backbone_features not implemented until T16");
    return false;
}
} // namespace da
