#include "vit_block.hpp"
#include "ggml_extend.hpp"
namespace da {

// Reference parity note (DA3 / DINOv2):
//   Block.norm1/norm2 are nn.LayerNorm(dim, eps=ln_eps) with ln_eps=1e-6.
//   The Attention module's q_norm/k_norm are nn.LayerNorm(head_dim) constructed
//   WITHOUT an eps argument (Block does not forward norm_layer to attn_class),
//   so they use torch's DEFAULT nn.LayerNorm eps = 1e-5 -- NOT the block's 1e-6.
//   Therefore the qk_norm LayerNorm inside attention() must use 1e-5.
static constexpr float QK_NORM_EPS = 1e-5f;

BlockWeights load_block(const ModelLoader& ml, int i){
    auto t=[&](const char* s){ return ml.tensor("vit.blk."+std::to_string(i)+"."+s); };
    BlockWeights w;
    w.n1_w=t("norm1.weight"); w.n1_b=t("norm1.bias");
    w.n2_w=t("norm2.weight"); w.n2_b=t("norm2.bias");
    w.ls1=t("ls1"); w.ls2=t("ls2");
    w.fc1_w=t("mlp_fc1.weight"); w.fc1_b=t("mlp_fc1.bias");
    w.fc2_w=t("mlp_fc2.weight"); w.fc2_b=t("mlp_fc2.bias");
    w.attn=load_attn(ml, i);
    return w;
}
ggml_tensor* vit_block(ggml_context* ctx, ggml_tensor* x, const BlockWeights& w,
                       int H, int D, float eps, ggml_tensor* cosb, ggml_tensor* sinb){
    // qk_norm uses torch-default eps (1e-5), distinct from the block norm eps (1e-6).
    ggml_tensor* a = attention(ctx, layernorm(ctx, x, w.n1_w, w.n1_b, eps), w.attn, H, D, QK_NORM_EPS, cosb, sinb);
    if (w.ls1) a = layerscale(ctx, a, w.ls1);
    x = ggml_add(ctx, x, a);
    ggml_tensor* m = linear(ctx, w.fc1_w, layernorm(ctx, x, w.n2_w, w.n2_b, eps), w.fc1_b);
    m = gelu_erf(ctx, m);
    m = linear(ctx, w.fc2_w, m, w.fc2_b);
    if (w.ls2) m = layerscale(ctx, m, w.ls2);
    return ggml_add(ctx, x, m);
}
}
