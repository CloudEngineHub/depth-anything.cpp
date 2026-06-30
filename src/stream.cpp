#include "stream.hpp"

#include "engine.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "reconstruct.hpp"
#include "sim3.hpp"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace da {

// Dense per-pixel world points (in a window's LOCAL frame) for a handful of
// overlap frames, kept around to align the NEXT window against.
struct OverlapCache {
    int H = 0, W = 0, n = 0;          // n cached frames (== overlap of a full window)
    std::vector<uint8_t> valid;       // n*H*W : finite,d>0,conf>=thr at snapshot time
    std::vector<double>  xyz;         // n*H*W*3 : LOCAL-frame world points
    std::vector<float>   conf;        // n*H*W
    Sim3 G;                           // LOCAL -> GLOBAL for the cached window
};

// Back-project one frame's pixels into LOCAL world space, filling dense valid/xyz/conf.
// Validity is GEOMETRIC only (finite, d>0) — alignment uses conf as a weight, not a
// hard gate, so the overlap correspondence isn't starved at low-confidence window
// edges. conf is stored raw (or 1.0 when the model gives no conf) for that weighting.
static void backproject_frame_dense(const ViewResult& view,
                                     const std::array<float,9>& K,
                                     const std::array<float,16>& E4,
                                     int H, int W, bool have_conf,
                                     uint8_t* valid, double* xyz, float* conf) {
    std::array<float,9> Kinv; std::array<float,16> c2w;
    const bool kok = inv3(K, Kinv);
    const bool eok = inv4(E4, c2w);
    double ki[9]; for (int k=0;k<9;++k) ki[k] = kok ? (double)Kinv[k] : 0.0;
    double cw[16]; for (int k=0;k<16;++k) cw[k] = eok ? (double)c2w[k] : 0.0;
    const size_t plane = (size_t)H*(size_t)W;
    const float* dF = view.depth.data();
    const float* cF = (have_conf && view.conf.size()==plane) ? view.conf.data() : nullptr;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            size_t pix = (size_t)v*W + u;
            float d = dF[pix];
            float cc = cF ? cF[pix] : 1.0f;
            conf[pix] = cc;
            bool ok = kok && eok && std::isfinite(d) && d > 0.0f;
            if (!ok) { valid[pix] = 0; xyz[3*pix+0]=xyz[3*pix+1]=xyz[3*pix+2]=0.0; continue; }
            double rx = ki[0]*u + ki[1]*v + ki[2];
            double ry = ki[3]*u + ki[4]*v + ki[5];
            double rz = ki[6]*u + ki[7]*v + ki[8];
            double xc = rx*d, yc = ry*d, zc = rz*d;
            xyz[3*pix+0] = cw[0]*xc + cw[1]*yc + cw[2]*zc + cw[3];
            xyz[3*pix+1] = cw[4]*xc + cw[5]*yc + cw[6]*zc + cw[7];
            xyz[3*pix+2] = cw[8]*xc + cw[9]*yc + cw[10]*zc + cw[11];
            valid[pix] = 1;
        }
    }
}

bool stream_points(Engine& eng, const std::vector<std::string>& frame_paths,
                   const Config& cfg, const StreamParams& p,
                   StreamCloud& out, std::string& err) {
    const int F = (int)frame_paths.size();
    if (F == 0) { err = "stream: no frames"; return false; }

    int chunk = p.chunk_size < 2 ? 2 : p.chunk_size;
    int overlap = p.overlap;
    if (overlap < 0) overlap = 0;
    if (overlap > chunk - 1) overlap = chunk - 1;
    int step = chunk - overlap;
    if (step < 1) step = 1;
    double conf_pct = p.conf_pct; if (conf_pct < 0) conf_pct = 0; if (conf_pct > 100) conf_pct = 100;
    float point_size = (p.point_size > 0.f) ? p.point_size : 1.f;

    out.xyz.clear(); out.rgb.clear(); out.radius.clear();
    out.counts.assign(F, 0);
    out.warnings = 0;

    Sim3 G;                 // current window LOCAL -> GLOBAL
    OverlapCache prev;      // previous window's tail-overlap geometry
    bool have_prev = false;

    for (int w0 = 0; w0 < F; w0 += step) {
        const int w1 = std::min(w0 + chunk, F);
        const int nv = w1 - w0;

        // --- load + fused inference for this window ---
        std::vector<Image> imgs(nv);
        for (int i = 0; i < nv; ++i)
            if (!load_image_rgb(frame_paths[w0+i], imgs[i])) {
                err = "stream: load image failed: " + frame_paths[w0+i]; return false; }

        std::vector<ViewResult> views; int H = 0, W = 0;
        if (!eng.depth_pose_multi(imgs, views, H, W) || (int)views.size() != nv) {
            err = "stream: depth_pose_multi failed"; return false; }
        const size_t plane = (size_t)H * (size_t)W;

        // K, E4, processed RGB per view.
        std::vector<std::array<float,9>>  K(nv);
        std::vector<std::array<float,16>> E4(nv);
        std::vector<std::vector<uint8_t>> rgb_store(nv);
        std::vector<const uint8_t*>       images_u8(nv);
        bool have_conf = true;
        std::vector<float> conf_all; conf_all.reserve((size_t)nv*plane);
        for (int i = 0; i < nv; ++i) {
            K[i] = views[i].intr;
            std::array<float,16> e4{}; for (int k=0;k<12;++k) e4[k]=views[i].ext[k];
            e4[12]=0.f; e4[13]=0.f; e4[14]=0.f; e4[15]=1.f; E4[i]=e4;
            Preprocessed pp;
            if (!preprocess_real(imgs[i], cfg, pp, &rgb_store[i]) || pp.H!=H || pp.W!=W) {
                err = "stream: preprocess color mismatch"; return false; }
            images_u8[i] = rgb_store[i].data();
            if (views[i].conf.size() == plane) conf_all.insert(conf_all.end(), views[i].conf.begin(), views[i].conf.end());
            else have_conf = false;
        }
        if (!have_conf) conf_all.clear();
        float conf_thr = -1e30f;
        if (have_conf && !conf_all.empty()) conf_thr = (float)percentile_linear(conf_all, conf_pct);

        // --- stitch: solve G (LOCAL -> GLOBAL) for this window ---
        if (!have_prev) {
            G = Sim3();  // window 0 defines the global frame
        } else {
            const int ov = std::min(prev.n, nv);
            std::vector<double> srcv, tgtv, wv;  // cur-local -> prev-local
            srcv.reserve((size_t)ov*plane*3);
            for (int o = 0; o < ov; ++o) {
                std::array<float,9> Kinv; std::array<float,16> c2w;
                if (!inv3(K[o], Kinv) || !inv4(E4[o], c2w)) continue;
                double ki[9]; for (int k=0;k<9;++k) ki[k]=(double)Kinv[k];
                double cw[16]; for (int k=0;k<16;++k) cw[k]=(double)c2w[k];
                const float* dcur = views[o].depth.data();
                const float* ccur = (have_conf && views[o].conf.size()==plane) ? views[o].conf.data() : nullptr;
                const uint8_t* pvalid = prev.valid.data() + (size_t)o*plane;
                const double*  pxyz   = prev.xyz.data()   + (size_t)o*plane*3;
                const float*   pconf  = prev.conf.data()  + (size_t)o*plane;
                for (int v = 0; v < H; ++v) for (int u = 0; u < W; ++u) {
                    size_t pix = (size_t)v*W + u;
                    if (!pvalid[pix]) continue;
                    float d = dcur[pix];
                    if (!std::isfinite(d) || d <= 0.0f) continue;
                    float cc = ccur ? ccur[pix] : 1.0f;   // weight only, no hard gate
                    double rx = ki[0]*u + ki[1]*v + ki[2];
                    double ry = ki[3]*u + ki[4]*v + ki[5];
                    double rz = ki[6]*u + ki[7]*v + ki[8];
                    double xc = rx*d, yc = ry*d, zc = rz*d;
                    srcv.push_back(cw[0]*xc + cw[1]*yc + cw[2]*zc + cw[3]);
                    srcv.push_back(cw[4]*xc + cw[5]*yc + cw[6]*zc + cw[7]);
                    srcv.push_back(cw[8]*xc + cw[9]*yc + cw[10]*zc + cw[11]);
                    tgtv.push_back(pxyz[3*pix+0]); tgtv.push_back(pxyz[3*pix+1]); tgtv.push_back(pxyz[3*pix+2]);
                    double wgt = std::min((double)cc, (double)pconf[pix]);
                    wv.push_back(wgt > 0.0 ? wgt : 1e-6);
                }
            }
            const int M = (int)wv.size();
            Sim3 S_rel; double rms = 0.0;
            bool ok = M >= p.min_overlap_pts &&
                      umeyama_sim3_weighted(srcv.data(), tgtv.data(), wv.data(), M, S_rel, rms, p.min_overlap_pts);
            if (!ok) {
                out.warnings++;
                DA_LOG("stream: window @%d degenerate stitch (M=%d) -> identity seam", w0, M);
                S_rel = Sim3();
            } else {
                DA_LOG("stream: window @%d stitch M=%d s=%.4f rms=%.4g", w0, M, S_rel.s, rms);
            }
            G = sim3_compose(prev.G, S_rel);  // cur-local -> global
        }

        // --- emit NEW frames only ---
        const int new_lo = have_prev ? std::min(prev.n, nv) : 0;
        const int newN = nv - new_lo;
        if (newN > 0) {
            std::vector<float> depth_slice; depth_slice.reserve((size_t)newN*plane);
            std::vector<float> conf_slice;  if (have_conf) conf_slice.reserve((size_t)newN*plane);
            std::vector<std::array<float,9>>  Ks(newN);
            std::vector<std::array<float,16>> Es(newN);
            std::vector<const uint8_t*>       Iu8(newN);
            for (int i = new_lo; i < nv; ++i) {
                int j = i - new_lo;
                depth_slice.insert(depth_slice.end(), views[i].depth.begin(), views[i].depth.end());
                if (have_conf) conf_slice.insert(conf_slice.end(), views[i].conf.begin(), views[i].conf.end());
                Ks[j] = K[i]; Es[j] = E4[i]; Iu8[j] = images_u8[i];
            }
            WorldPoints wp = back_project(depth_slice, have_conf ? conf_slice : std::vector<float>{},
                                          Ks, Es, Iu8, H, W, newN, conf_thr);
            const size_t np = wp.frame.size();

            // Per-window budget: uniform stride over the frame-major point list.
            size_t stride = 1;
            if (p.global_budget > 0 && F > 0) {
                double quota = (double)p.global_budget * (double)newN / (double)F;
                if (quota < 1.0) quota = 1.0;
                if ((double)np > quota) stride = (size_t)std::ceil((double)np / quota);
                if (stride < 1) stride = 1;
            }

            for (size_t k = 0; k < np; k += stride) {
                int f = wp.frame[k], u = wp.u[k], v = wp.v[k];
                double pL[3] = { wp.xyz[3*k+0], wp.xyz[3*k+1], wp.xyz[3*k+2] };
                double pG[3]; sim3_apply(G, pL, pG);
                out.xyz.push_back((float)pG[0]);
                out.xyz.push_back((float)pG[1]);
                out.xyz.push_back((float)pG[2]);
                out.rgb.push_back(wp.rgb[3*k+0]);
                out.rgb.push_back(wp.rgb[3*k+1]);
                out.rgb.push_back(wp.rgb[3*k+2]);
                float d = depth_slice[(size_t)f*plane + (size_t)v*W + u];
                float fx = Ks[f][0], fy = Ks[f][4];
                float rr = 0.5f * (d/fx + d/fy) * point_size * (float)G.s;
                if (!(rr > 0.f) || !std::isfinite(rr)) rr = 1e-4f;
                out.radius.push_back(rr);
                int gframe = w0 + new_lo + f;
                if (gframe >= 0 && gframe < F) out.counts[gframe]++;
            }
        }

        // --- snapshot this window's TAIL overlap (LOCAL frame) for the next stitch ---
        const int ovkeep = std::min(overlap, nv);
        prev.H = H; prev.W = W; prev.n = ovkeep; prev.G = G;
        prev.valid.assign((size_t)ovkeep*plane, 0);
        prev.xyz.assign((size_t)ovkeep*plane*3, 0.0);
        prev.conf.assign((size_t)ovkeep*plane, 0.f);
        for (int o = 0; o < ovkeep; ++o) {
            int vi = nv - ovkeep + o;  // local index of this tail frame
            backproject_frame_dense(views[vi], K[vi], E4[vi], H, W, have_conf,
                                    prev.valid.data() + (size_t)o*plane,
                                    prev.xyz.data()   + (size_t)o*plane*3,
                                    prev.conf.data()  + (size_t)o*plane);
        }
        have_prev = true;

        if (w1 >= F) break;  // last (possibly truncated) window processed
    }

    if (out.radius.empty()) { err = "stream: no points survived (raise conf_pct or check parallax)"; return false; }
    return true;
}

} // namespace da
