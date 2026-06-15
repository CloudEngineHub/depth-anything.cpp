#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "rope2d.hpp"
#include <vector>
namespace da {
class DinoBackbone {
public:
    DinoBackbone(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // input_chw: [3,H,W] normalized. Returns tokens flattened [embed * (1+N_patch)],
    // ggml-order (embed fastest): element (token,e) at token*embed + e.
    bool prepare_tokens(const std::vector<float>& input_chw, int H, int W, std::vector<float>& out_tokens);
    bool forward(const std::vector<float>& input_chw, int H, int W,
                 std::vector<std::vector<float>>& feats,
                 std::vector<std::vector<float>>& cam_tokens);  // implemented in T15 - leave declared
    // Multi-view (S>=1) backbone with cross-view global attention.
    //   views_chw : S items, each [3,H,W] normalized.
    //   feats     : [L][S] each [256*1536] (cat[local_x, norm(x)], token0 stripped, per view).
    //   cam_tokens: [L][S] each [1536]     (cat[local_x[t0], x[t0]] RAW, per view).
    bool forward_mv(const std::vector<std::vector<float>>& views_chw, int H, int W,
                    std::vector<std::vector<std::vector<float>>>& feats,
                    std::vector<std::vector<std::vector<float>>>& cam_tokens);
private:
    std::vector<float> interp_pos_embed(int gh, int gw) const;  // host bicubic -> [(1+gh*gw)*embed], token-major embed-minor
    ModelLoader& ml_; Backend& be_;
};
}
