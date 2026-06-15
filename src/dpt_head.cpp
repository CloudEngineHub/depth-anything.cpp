#include "dpt_head.hpp"
#include "dpt_blocks.hpp"
#include "uv_posembed.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <string>

namespace da {

// Build the *0.1-scaled UV positional embedding as a ggml graph input in
// [W,H,C] (w fastest, then h, then c) memory layout, matching feature maps.
// uv_pos_embed returns (y,x,c) flat = (y*W + x)*C + c (RAW, no ratio).
static ggml_tensor* add_uv_input(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                                 int W, int H, int C, float aspect, float ratio) {
    std::vector<float> uv = uv_pos_embed(/*pw=*/W, /*ph=*/H, C, aspect);
    std::vector<float> buf((size_t)W * H * C);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                buf[(size_t)c * H * W + (size_t)h * W + w] =
                    ratio * uv[((size_t)h * W + w) * C + c];
    const int64_t ne[4] = { W, H, C, 1 };
    return be.add_graph_input_nd(ctx, pool, buf.data(), ne, 4);
}

bool DptHead::run(const std::vector<std::vector<float>>& feats, int H, int W,
                  std::vector<float>& depth_out, std::vector<float>& conf_out,
                  std::vector<std::vector<float>>* stages, std::vector<float>* fused) {
    if (feats.size() != 4) return false;
    const Config& cfg = ml_.config();
    const int patch = (int)cfg.patch_size;           // 14
    const int pw = W / patch, ph = H / patch;        // 16,16
    const int N = ph * pw;                            // 256
    const int C = 1536;                               // dim_in (feat channel)
    const int oc[4] = { 96, 192, 384, 768 };          // out_channels
    const float aspect = (float)W / (float)H;        // 1.0
    const float ratio = 0.1f;
    const float eps = 1e-5f;                          // head.norm is nn.LayerNorm default

    for (const auto& f : feats)
        if ((int)f.size() != N * C) return false;

    auto t = [&](const std::string& n) { return ml_.tensor(n); };

    std::vector<float> stage_caps[4];
    std::vector<float> fused_cap;

    GraphInputPool pool;
    std::vector<float> logits;
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* norm_w = t("head.norm.weight");
        ggml_tensor* norm_b = t("head.norm.bias");

        ggml_tensor* l[4];
        for (int s = 0; s < 4; ++s) {
            // feat input [C=1536, N=256] (ne0=channel fastest, ne1=token)
            const int64_t fne[2] = { C, N };
            ggml_tensor* x = be_.add_graph_input_nd(ctx, pool, feats[s].data(), fne, 2);
            // LayerNorm over channel dim (ne0).
            x = layernorm(ctx, x, norm_w, norm_b, eps);
            // [C,N] -> [N,C] -> [W,H,C,1]  (token = h*pw + w => ne0=w, ne1=h)
            x = ggml_cont(ctx, ggml_transpose(ctx, x));      // [N,C]
            x = ggml_reshape_4d(ctx, x, pw, ph, C, 1);       // [W,H,C,1]
            // projects[s]: 1x1 conv 1536 -> oc[s]
            x = conv2d(ctx, t("head.proj." + std::to_string(s) + ".weight"),
                       t("head.proj." + std::to_string(s) + ".bias"), x, 1, 0);
            // + 0.1 * UV pos-embed at [pw,ph,oc]
            ggml_tensor* pe = add_uv_input(ctx, be_, pool, pw, ph, oc[s], aspect, ratio);
            x = ggml_add(ctx, x, pe);
            // resize_layers[s]
            if (s == 0)
                x = conv_transpose2d_p0(ctx, t("head.resize.0.weight"),
                                        t("head.resize.0.bias"), x, 4);   // ->64x64
            else if (s == 1)
                x = conv_transpose2d_p0(ctx, t("head.resize.1.weight"),
                                        t("head.resize.1.bias"), x, 2);   // ->32x32
            else if (s == 3)
                x = conv2d(ctx, t("head.resize.3.weight"),
                           t("head.resize.3.bias"), x, 2, 1);            // ->8x8
            // s==2 Identity
            l[s] = x;
            if (stages) be_.capture(x, &stage_caps[s]);
        }

        // _fuse: layer{i}_rn = 3x3 pad1 conv (NO bias), out_channels[i-1] -> 128
        ggml_tensor* l1_rn = conv2d(ctx, t("head.scratch.layer1_rn.weight"), nullptr, l[0], 1, 1);
        ggml_tensor* l2_rn = conv2d(ctx, t("head.scratch.layer2_rn.weight"), nullptr, l[1], 1, 1);
        ggml_tensor* l3_rn = conv2d(ctx, t("head.scratch.layer3_rn.weight"), nullptr, l[2], 1, 1);
        ggml_tensor* l4_rn = conv2d(ctx, t("head.scratch.layer4_rn.weight"), nullptr, l[3], 1, 1);

        // refinenet4 (no residual / no rc1): top=l4_rn, size=l3 (16x16)
        ggml_tensor* out = feature_fusion(ctx, l4_rn, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            t("head.scratch.rn4.rc2.c1.weight"), t("head.scratch.rn4.rc2.c1.bias"),
            t("head.scratch.rn4.rc2.c2.weight"), t("head.scratch.rn4.rc2.c2.bias"),
            t("head.scratch.rn4.out.weight"), t("head.scratch.rn4.out.bias"), 16, 16);
        // refinenet3: lateral=l3_rn, size=l2 (32x32)
        out = feature_fusion(ctx, out, l3_rn,
            t("head.scratch.rn3.rc1.c1.weight"), t("head.scratch.rn3.rc1.c1.bias"),
            t("head.scratch.rn3.rc1.c2.weight"), t("head.scratch.rn3.rc1.c2.bias"),
            t("head.scratch.rn3.rc2.c1.weight"), t("head.scratch.rn3.rc2.c1.bias"),
            t("head.scratch.rn3.rc2.c2.weight"), t("head.scratch.rn3.rc2.c2.bias"),
            t("head.scratch.rn3.out.weight"), t("head.scratch.rn3.out.bias"), 32, 32);
        // refinenet2: lateral=l2_rn, size=l1 (64x64)
        out = feature_fusion(ctx, out, l2_rn,
            t("head.scratch.rn2.rc1.c1.weight"), t("head.scratch.rn2.rc1.c1.bias"),
            t("head.scratch.rn2.rc1.c2.weight"), t("head.scratch.rn2.rc1.c2.bias"),
            t("head.scratch.rn2.rc2.c1.weight"), t("head.scratch.rn2.rc2.c1.bias"),
            t("head.scratch.rn2.rc2.c2.weight"), t("head.scratch.rn2.rc2.c2.bias"),
            t("head.scratch.rn2.out.weight"), t("head.scratch.rn2.out.bias"), 64, 64);
        // refinenet1: lateral=l1_rn, scale_factor 2 -> 128x128
        out = feature_fusion(ctx, out, l1_rn,
            t("head.scratch.rn1.rc1.c1.weight"), t("head.scratch.rn1.rc1.c1.bias"),
            t("head.scratch.rn1.rc1.c2.weight"), t("head.scratch.rn1.rc1.c2.bias"),
            t("head.scratch.rn1.rc2.c1.weight"), t("head.scratch.rn1.rc2.c1.bias"),
            t("head.scratch.rn1.rc2.c2.weight"), t("head.scratch.rn1.rc2.c2.bias"),
            t("head.scratch.rn1.out.weight"), t("head.scratch.rn1.out.bias"), 0, 0);
        // output_conv1: 3x3 pad1, 128 -> 64
        out = conv2d(ctx, t("head.scratch.out1.weight"), t("head.scratch.out1.bias"), out, 1, 1);
        if (fused) be_.capture(out, &fused_cap);

        // upsample to (H,W), + 0.1*UV(64)
        out = interp_bilinear_ac(ctx, out, W, H);             // [224,224,64]
        ggml_tensor* pe2 = add_uv_input(ctx, be_, pool, W, H, 64, aspect, ratio);
        out = ggml_add(ctx, out, pe2);
        // output_conv2: conv 64->32 (3x3 pad1) -> relu -> conv 32->2 (1x1)
        out = conv2d(ctx, t("head.scratch.out2a.weight"), t("head.scratch.out2a.bias"), out, 1, 1);
        out = ggml_relu(ctx, out);
        out = conv2d(ctx, t("head.scratch.out2b.weight"), t("head.scratch.out2b.bias"), out, 1, 0);
        return out;                                            // [W,H,2,1] logits
    }, logits);
    if (!ok) return false;

    const size_t HW = (size_t)H * W;
    if (logits.size() != 2 * HW) return false;
    // channel 0 = depth = exp(logits); channel 1 = conf = exp(logits)+1.
    depth_out.resize(HW);
    conf_out.resize(HW);
    for (size_t i = 0; i < HW; ++i) {
        depth_out[i] = std::exp(logits[i]);
        conf_out[i]  = std::exp(logits[HW + i]) + 1.0f;
    }
    if (stages) {
        stages->resize(4);
        for (int s = 0; s < 4; ++s) (*stages)[s] = std::move(stage_caps[s]);
    }
    if (fused) *fused = std::move(fused_cap);
    return true;
}

bool DptHead::depth(const std::vector<std::vector<float>>& feats, int H, int W,
                    std::vector<float>& depth_out, std::vector<float>& conf_out) {
    return run(feats, H, W, depth_out, conf_out, nullptr, nullptr);
}

bool DptHead::depth_debug(const std::vector<std::vector<float>>& feats, int H, int W,
                          std::vector<float>& depth_out, std::vector<float>& conf_out,
                          std::vector<std::vector<float>>& stages, std::vector<float>& fused) {
    return run(feats, H, W, depth_out, conf_out, &stages, &fused);
}

} // namespace da
