#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>

namespace da {

// Full DualDPT MAIN depth path (DA3) as a ggml forward, taking the four backbone
// out-layer features (feat_5/7/9/11) as input and producing the dense depth +
// confidence maps. Mirrors DualDPT._forward_impl (main branch only).
class DptHead {
public:
    DptHead(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}

    // feats: 4 vectors each [N*C] = [256*1536], token-major / channel-minor
    //        (flat index = token*C + channel), i.e. the backbone out-layer features.
    // Returns depth [H*W] and conf [H*W] (row-major h,w; w fastest).
    bool depth(const std::vector<std::vector<float>>& feats, int H, int W,
               std::vector<float>& depth_out, std::vector<float>& conf_out);

    // Same as depth() but also reads back the 4 post-resize stage maps and the
    // post-output_conv1 fused map for the layer-isolation parity gates.
    //   stages[s]: [out_channels[s] * Hs * Ws] (ggml [W,H,C] order)
    //   fused    : [64 * 128 * 128]            (ggml [W,H,C] order)
    bool depth_debug(const std::vector<std::vector<float>>& feats, int H, int W,
                     std::vector<float>& depth_out, std::vector<float>& conf_out,
                     std::vector<std::vector<float>>& stages, std::vector<float>& fused);

    // Metric single-head: dim_in = embed (cat_token false), NO head.norm (norm_type
    // "idt"), output_dim 1 (depth only) + a parallel sky head. depth = exp(logit),
    // sky = relu(logit). feats: 4 vectors each [N*embed]. Returns depth & sky [H*W].
    bool depth_sky(const std::vector<std::vector<float>>& feats, int H, int W,
                   std::vector<float>& depth_out, std::vector<float>& sky_out);

private:
    bool run(const std::vector<std::vector<float>>& feats, int H, int W,
             std::vector<float>& depth_out, std::vector<float>& conf_out,
             std::vector<std::vector<float>>* stages, std::vector<float>* fused,
             std::vector<float>* sky_out = nullptr);
    ModelLoader& ml_;
    Backend& be_;
};

} // namespace da
