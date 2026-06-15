#include "winograd.hpp"
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace da {
namespace {

// ------------------------------------------------------------------------
// Winograd F(2x2,3x3) transform matrices (exact: halves + integers).
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   A^T = [[1,1,1,0],[0,1,-1,-1]]
// ------------------------------------------------------------------------

// U = G g G^T for a single 3x3 filter g (row=ky, col=kx). Result 4x4 -> u[16].
static inline void filter_transform(const float g[9], float u[16]) {
    // Gg : 4x3 (apply G on rows of g)
    float Gg[4][3];
    for (int j = 0; j < 3; ++j) {
        float c0 = g[0*3 + j], c1 = g[1*3 + j], c2 = g[2*3 + j];
        Gg[0][j] = c0;
        Gg[1][j] = 0.5f * (c0 + c1 + c2);
        Gg[2][j] = 0.5f * (c0 - c1 + c2);
        Gg[3][j] = c2;
    }
    // U = Gg G^T : 4x4 (apply G on columns of Gg)
    for (int i = 0; i < 4; ++i) {
        float c0 = Gg[i][0], c1 = Gg[i][1], c2 = Gg[i][2];
        u[i*4 + 0] = c0;
        u[i*4 + 1] = 0.5f * (c0 + c1 + c2);
        u[i*4 + 2] = 0.5f * (c0 - c1 + c2);
        u[i*4 + 3] = c2;
    }
}

// V = B^T d B for a 4x4 input patch d -> v[16].
static inline void input_transform(const float d[16], float v[16]) {
    // m = B^T d (combine rows)
    float m[16];
    for (int j = 0; j < 4; ++j) {
        float r0 = d[0*4 + j], r1 = d[1*4 + j], r2 = d[2*4 + j], r3 = d[3*4 + j];
        m[0*4 + j] = r0 - r2;
        m[1*4 + j] = r1 + r2;
        m[2*4 + j] = r2 - r1;
        m[3*4 + j] = r1 - r3;
    }
    // V = m B (combine columns the same way)
    for (int i = 0; i < 4; ++i) {
        float c0 = m[i*4 + 0], c1 = m[i*4 + 1], c2 = m[i*4 + 2], c3 = m[i*4 + 3];
        v[i*4 + 0] = c0 - c2;
        v[i*4 + 1] = c1 + c2;
        v[i*4 + 2] = c2 - c1;
        v[i*4 + 3] = c1 - c3;
    }
}

// Y = A^T m A for a 4x4 winograd-domain m -> 2x2 output y[4].
static inline void output_transform(const float m[16], float y[4]) {
    // p = A^T m : 2x4
    float p[8];
    for (int j = 0; j < 4; ++j) {
        float r0 = m[0*4 + j], r1 = m[1*4 + j], r2 = m[2*4 + j], r3 = m[3*4 + j];
        p[0*4 + j] = r0 + r1 + r2;
        p[1*4 + j] = r1 - r2 - r3;
    }
    // Y = p A : 2x2
    for (int i = 0; i < 2; ++i) {
        float c0 = p[i*4 + 0], c1 = p[i*4 + 1], c2 = p[i*4 + 2], c3 = p[i*4 + 3];
        y[i*2 + 0] = c0 + c1 + c2;
        y[i*2 + 1] = c1 - c2 - c3;
    }
}

// Persistent per-op state: caches the filter transform U (computed once from
// w->data; reused across forwards as long as the same filter pointer + shapes
// are seen). Scratch (V,M) is per-thread/per-tile and lives on the stack.
struct WinogradState {
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int Wout = 0, Hout = 0, tilesX = 0, tilesY = 0;
    const void* wdata = nullptr;
    // U layout: U[pos*IC*OC + ic*OC + oc], pos in 0..15. OC contiguous (innermost)
    // so the winograd-domain multiply vectorizes over OC.
    std::vector<float> U;
    std::once_flag once;
};

// Compute the filter transform into state->U from the filter weights w (F32,
// [3,3,IC,OC]). Called exactly once per state via std::call_once.
static void build_U(WinogradState* st, const float* w) {
    const int IC = st->IC, OC = st->OC;
    st->U.assign((size_t)16 * IC * OC, 0.0f);
    float u[16];
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            const float* g = w + ((size_t)oc * IC + ic) * 9;  // [3,3] for (oc,ic)
            filter_transform(g, u);
            for (int pos = 0; pos < 16; ++pos)
                st->U[(size_t)pos * IC * OC + (size_t)ic * OC + oc] = u[pos];
        }
    }
}

// Winograd-domain multiply for one tile + one position: M[oc] = sum_ic U[ic,oc]*V[ic]
//   Upos: [IC][OC] contiguous, Vpos: [IC], out: [OC].
static inline void wino_gemv(const float* Upos, const float* Vpos, float* out, int IC, int OC) {
#if defined(__AVX512F__)
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        }
        _mm512_storeu_ps(out + oc, acc);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic) {
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        }
        _mm512_mask_storeu_ps(out + oc, mask, acc);
    }
#else
    for (int oc = 0; oc < OC; ++oc) out[oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float vv = Vpos[ic];
        const float* up = Upos + (size_t)ic * OC;
        for (int oc = 0; oc < OC; ++oc) out[oc] += up[oc] * vv;
    }
#endif
}

static void winograd_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    WinogradState* st = (WinogradState*)userdata;
    const ggml_tensor* xt = dst->src[0];   // [W,H,IC,N]
    const ggml_tensor* wt = dst->src[1];   // [3,3,IC,OC]
    const float* x = (const float*)xt->data;
    const float* w = (const float*)wt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, N = st->N, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;

    // Filter transform: once across all threads/forwards.
    std::call_once(st->once, [&]{ build_U(st, w); });
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)N * ntiles;
    const int64_t beg = total * ith / nth;
    const int64_t end = total * (ith + 1) / nth;

    // Per-thread scratch.
    std::vector<float> Vbuf((size_t)16 * IC);   // V[pos*IC + ic]
    std::vector<float> Mbuf((size_t)16 * OC);   // M[pos*OC + oc]
    float dpatch[16];
    float vpatch[16];
    float mpatch[16];
    float ypatch[4];

    for (int64_t idx = beg; idx < end; ++idx) {
        const int n  = (int)(idx / ntiles);
        const int t  = (int)(idx % ntiles);
        const int ty = t / tilesX;
        const int tx = t % tilesX;
        const int oy0 = ty * 2;       // output row base
        const int ox0 = tx * 2;       // output col base
        const int iy0 = oy0 - pad;    // input row base of the 4x4 patch
        const int ix0 = ox0 - pad;

        const float* xn = x + (size_t)n * IC * H * W;

        // 1. Input transform per IC -> Vbuf[pos*IC + ic].
        for (int ic = 0; ic < IC; ++ic) {
            const float* xc = xn + (size_t)ic * H * W;
            // gather 4x4 patch with zero-pad
            for (int i = 0; i < 4; ++i) {
                const int yy = iy0 + i;
                const bool yok = (yy >= 0 && yy < H);
                const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                for (int j = 0; j < 4; ++j) {
                    const int xx = ix0 + j;
                    dpatch[i*4 + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                }
            }
            input_transform(dpatch, vpatch);
            for (int pos = 0; pos < 16; ++pos)
                Vbuf[(size_t)pos * IC + ic] = vpatch[pos];
        }

        // 2. Winograd-domain multiply: M[pos][oc] = sum_ic U[pos][ic][oc]*V[pos][ic].
        for (int pos = 0; pos < 16; ++pos) {
            const float* Upos = U + (size_t)pos * IC * OC;
            const float* Vpos = Vbuf.data() + (size_t)pos * IC;
            wino_gemv(Upos, Vpos, Mbuf.data() + (size_t)pos * OC, IC, OC);
        }

        // 3. Output transform per OC -> scatter 2x2 into dst (valid outputs only).
        float* yn = y + (size_t)n * OC * Hout * Wout;
        for (int oc = 0; oc < OC; ++oc) {
            for (int pos = 0; pos < 16; ++pos)
                mpatch[pos] = Mbuf[(size_t)pos * OC + oc];
            output_transform(mpatch, ypatch);
            float* yc = yn + (size_t)oc * Hout * Wout;
            for (int i = 0; i < 2; ++i) {
                const int oy = oy0 + i;
                if (oy >= Hout) continue;
                for (int j = 0; j < 2; ++j) {
                    const int ox = ox0 + j;
                    if (ox >= Wout) continue;
                    yc[(size_t)oy * Wout + ox] = ypatch[i*2 + j];
                }
            }
        }
    }
}

// Keyed cache of op states. The filter weight pointer is stable across forwards
// (model weights), so U is transformed once and reused. Bounded by the number of
// distinct (filter, shape) convs in the graph (~tens of entries).
struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad);
        return h;
    }
};

static std::mutex g_states_mtx;
static std::unordered_map<StateKey, WinogradState*, StateKeyHash> g_states;

static WinogradState* get_state(ggml_tensor* w, ggml_tensor* x, int pad) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    WinogradState* st = new WinogradState();
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->Wout = W + 2 * pad - 2;
    st->Hout = H + 2 * pad - 2;
    st->tilesX = (st->Wout + 1) / 2;
    st->tilesY = (st->Hout + 1) / 2;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

} // namespace

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    WinogradState* st = get_state(w, x, pad);
    ggml_tensor* args[2] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, 2, winograd_compute, GGML_N_TASKS_MAX, st);
}

} // namespace da
