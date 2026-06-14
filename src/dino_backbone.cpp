#include "dino_backbone.hpp"
#include "ggml_extend.hpp"
#include "common.hpp"
#include <cmath>
#include <algorithm>
namespace da {
static float cubic(float x){ // Catmull-Rom, a=-0.75 (PyTorch bicubic)
    const float a=-0.75f; x=std::fabs(x);
    if (x<1) return ((a+2)*x - (a+3))*x*x + 1;
    if (x<2) return (((x-5)*x+8)*x-4)*a;
    return 0;
}
std::vector<float> DinoBackbone::interp_pos_embed(int gh, int gw) const {
    const auto& c = ml_.config();
    ggml_tensor* pe = ml_.tensor("vit.pos_embed");   // ggml ne0=embed (768), ne1=rows (1370)
    const int embed = (int)c.embed_dim, M = (int)c.pos_embed_grid;
    const float* p = (const float*)pe->data;         // row r, channel ch at p[r*embed + ch]
    auto src = [&](int r, int cc, int ch)->float{
        int row = 1 + r*M + cc; return p[(size_t)row*embed + ch];
    };
    std::vector<float> out((size_t)(1+gh*gw)*embed);
    for (int ch=0; ch<embed; ++ch) out[ch] = p[ch];   // cls pos-embed (row 0)
    const float sx = (float)(gw + c.interp_offset)/M, sy = (float)(gh + c.interp_offset)/M;
    for (int oy=0; oy<gh; ++oy){
        float iy = (oy+0.5f)/sy - 0.5f; int y0=(int)std::floor(iy); float fy=iy-y0;
        for (int ox=0; ox<gw; ++ox){
            float ix = (ox+0.5f)/sx - 0.5f; int x0=(int)std::floor(ix); float fx=ix-x0;
            int orow = 1 + oy*gw + ox;
            for (int ch=0; ch<embed; ++ch){
                float acc=0;
                for (int m=-1;m<=2;++m){ float wyv=cubic(fy-m); int yy=std::min(std::max(y0+m,0),M-1);
                    for (int n=-1;n<=2;++n){ float wxv=cubic(fx-n); int xx=std::min(std::max(x0+n,0),M-1);
                        acc += wyv*wxv*src(yy,xx,ch); }}
                out[(size_t)orow*embed + ch] = acc;
            }
        }
    }
    return out;
}
bool DinoBackbone::prepare_tokens(const std::vector<float>& input_chw, int H, int W, std::vector<float>& out_tokens){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch, embed=(int)c.embed_dim;
    std::vector<float> pos = interp_pos_embed(gh, gw);
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ine[4] = { W, H, 3, 1 };
        ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, input_chw.data(), ine, 4);
        ggml_tensor* pw = ml_.tensor("vit.patch_embed.weight");   // conv weight
        ggml_tensor* pb = ml_.tensor("vit.patch_embed.bias");
        ggml_tensor* x = ggml_conv_2d(ctx, pw, img, patch, patch, 0,0,1,1); // -> [gw,gh,embed,1]
        x = ggml_reshape_2d(ctx, x, (int64_t)gw*gh, embed);                  // [N_patch, embed]
        x = ggml_cont(ctx, ggml_transpose(ctx, x));                         // [embed, N_patch]
        x = ggml_add(ctx, x, pb);                                           // bias broadcast over tokens
        ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
        x = ggml_concat(ctx, cls, x, 1);                                    // [embed, 1+N_patch]
        const int64_t pne[2] = { embed, 1 + (int64_t)gh*gw };
        ggml_tensor* pe = be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2);
        x = ggml_add(ctx, x, pe);
        return x;
    }, out_tokens);
}
}
