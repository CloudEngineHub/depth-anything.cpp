#include "dino_backbone.hpp"
#include "ggml_extend.hpp"
#include "vit_block.hpp"
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

bool DinoBackbone::forward(const std::vector<float>& input_chw, int H, int W,
                           std::vector<std::vector<float>>& feats,
                           std::vector<std::vector<float>>& cam_tokens){
    const auto& c = ml_.config();
    const int patch=(int)c.patch_size, gh=H/patch, gw=W/patch;
    const int embed=(int)c.embed_dim, heads=(int)c.num_heads, hd=(int)c.head_dim;
    const int Npatch=gh*gw, Ntok=1+Npatch;
    const float eps=c.ln_eps;

    // Two RoPE position sets over the 257 tokens. Token 0 (cls/cam) is a "special"
    // row at (0,0) for both. Patch token t (1..256): pos_local = (row+1, col+1);
    // pos_nodiff = (1,1) for every patch (zeros_like + 1).
    std::vector<float> pos_local(2*Ntok, 0.f), pos_nodiff(2*Ntok, 0.f);
    for (int t=1;t<Ntok;++t){ int idx=t-1; int row=idx/gw, col=idx%gw;
        pos_local[2*t+0]=(float)(row+1); pos_local[2*t+1]=(float)(col+1);
        pos_nodiff[2*t+0]=1.f;           pos_nodiff[2*t+1]=1.f; }
    RopeTables rt_local  = build_rope_tables(pos_local,  Ntok, hd, c.rope_freq);
    RopeTables rt_nodiff = build_rope_tables(pos_nodiff, Ntok, hd, c.rope_freq);

    std::vector<float> pos = interp_pos_embed(gh, gw);

    // Camera token (N=1): reference slot only -> camera_token row 0 (ne1 index 0).
    ggml_tensor* camt = ml_.tensor("vit.camera_token");          // ggml ne0=embed, ne1=2
    std::vector<float> cam0(embed, 0.f);
    if (camt){ const float* cp=(const float*)camt->data; for (int e=0;e<embed;++e) cam0[e]=cp[e]; }

    const std::vector<int32_t>& outL = c.out_layers;
    const size_t NL = outL.size();
    feats.assign(NL, {}); cam_tokens.assign(NL, {});

    // Per-out-layer raw captures: local_x (last LOCAL output) and x (post-block).
    std::vector<std::vector<float>> raw_local(NL), raw_x(NL);

    GraphInputPool pool;
    std::vector<float> throwaway;
    bool ok = be_.forward_capture([&](ggml_context* ctx) -> ggml_tensor* {
        // --- prepare tokens (same graph as prepare_tokens) ---
        const int64_t ine[4]={W,H,3,1};
        ggml_tensor* img = be_.add_graph_input_nd(ctx, pool, input_chw.data(), ine, 4);
        ggml_tensor* x = ggml_conv_2d(ctx, ml_.tensor("vit.patch_embed.weight"), img, patch,patch,0,0,1,1);
        x = ggml_reshape_2d(ctx, x, (int64_t)Npatch, embed);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_add(ctx, x, ml_.tensor("vit.patch_embed.bias"));
        ggml_tensor* cls = ggml_reshape_2d(ctx, ml_.tensor("vit.cls_token"), embed, 1);
        x = ggml_concat(ctx, cls, x, 1);
        const int64_t pne[2]={embed, Ntok};
        x = ggml_add(ctx, x, be_.add_graph_input_nd(ctx, pool, pos.data(), pne, 2));

        // --- rope inputs (both position sets) + camera token input ---
        ggml_tensor *clb,*slb,*cnb,*snb;
        build_rope_inputs(ctx, be_, pool, rt_local,  clb, slb);
        build_rope_inputs(ctx, be_, pool, rt_nodiff, cnb, snb);
        const int64_t camne[2]={embed,1};
        ggml_tensor* cam_in = be_.add_graph_input_nd(ctx, pool, cam0.data(), camne, 2);

        ggml_tensor* local_x = x;          // last LOCAL-attention output
        for (int i=0;i<(int)c.depth;++i){
            // Cam-token overwrite BEFORE block i==alt_start: x[:,token0] = cam_token.
            if (c.alt_start>=0 && i==c.alt_start){
                ggml_tensor* rest = ggml_cont(ctx, ggml_view_2d(ctx, x, embed, Ntok-1,
                                                                x->nb[1], x->nb[1]));
                x = ggml_concat(ctx, cam_in, rest, 1);
            }
            const bool global   = (c.alt_start>=0 && i>=c.alt_start && (i%2==1));
            const bool use_rope  = (c.rope_start>=0 && i>=c.rope_start);
            ggml_tensor* cb = use_rope ? (global? cnb: clb) : nullptr;
            ggml_tensor* sb = use_rope ? (global? snb: slb) : nullptr;
            BlockWeights bw = load_block(ml_, i);
            x = vit_block(ctx, x, bw, heads, hd, eps, cb, sb);
            if (!global) local_x = x;       // track most recent LOCAL output only
            for (size_t o=0;o<NL;++o) if (outL[o]==i){
                be_.capture(local_x, &raw_local[o]);
                be_.capture(x,       &raw_x[o]);
            }
        }
        return x;                            // final readback discarded
    }, throwaway);
    if (!ok) return false;

    // --- host post-process matching get_intermediate_layers (width == 2*embed) ---
    // feat   = cat([local_x, vit_norm(x)]) over channels, patches 1..256  -> [256,1536]
    // cam    = cat([local_x[token0], x[token0]]) RAW (no norm)            -> [1536]
    ggml_tensor* nw = ml_.tensor("vit.norm.weight");
    ggml_tensor* nb = ml_.tensor("vit.norm.bias");
    const float* nwp=(const float*)nw->data;
    const float* nbp=(const float*)nb->data;
    auto layernorm_host=[&](const float* row)->std::vector<float>{
        double mean=0; for(int e=0;e<embed;++e) mean+=row[e]; mean/=embed;
        double var=0; for(int e=0;e<embed;++e){ double d=row[e]-mean; var+=d*d; } var/=embed;
        double inv=1.0/std::sqrt(var+(double)eps); std::vector<float> o(embed);
        for(int e=0;e<embed;++e) o[e]=(float)((row[e]-mean)*inv)*nwp[e]+nbp[e];
        return o;
    };
    for (size_t o=0;o<NL;++o){
        const auto& lx=raw_local[o]; const auto& xx=raw_x[o];  // both [embed, Ntok], ne0=embed fastest
        // camera token: raw cat of token-0 halves (second half UN-normed).
        std::vector<float> camcat((size_t)2*embed);
        for(int e=0;e<embed;++e){ camcat[e]=lx[e]; camcat[embed+e]=xx[e]; }
        cam_tokens[o]=std::move(camcat);
        // features for patches 1..256, channel = cat([local_x_raw, norm(x)]).
        std::vector<float> f((size_t)Npatch*2*embed);
        for(int t=1;t<Ntok;++t){
            const float* lrow=&lx[(size_t)t*embed];
            const float* xrow=&xx[(size_t)t*embed];
            std::vector<float> no=layernorm_host(xrow);
            float* dst=&f[(size_t)(t-1)*2*embed];
            for(int e=0;e<embed;++e){ dst[e]=lrow[e]; dst[embed+e]=no[e]; }
        }
        feats[o]=std::move(f);
    }
    return true;
}
}
