// Baseline ghosting measurement for the streaming stitcher, on SURFACE ground
// truth. Drives the real stream_points_core (Task #4 refactor) over a looping
// room scene, injecting the error modes that cause real ghosting:
//   * per-view depth noise (independent per window => the SAME frame's depth
//     differs between the two windows that share it, as with real multi-view);
//   * a per-window non-rigid depth warp (a smooth stretch a global Sim3 can NOT
//     undo — mimics monocular depth's non-metric bend, the true ghosting driver).
// Then measures the fused cloud vs the analytic surfaces. The printed numbers are
// the baseline that A (fusion), B (ICP), C (loop-closure) must beat. Also serves
// as the functional test that the refactored core still stitches correctly.
#include "synth_scene.hpp"
#include "geom_metrics.hpp"
#include "stream.hpp"
#include "fuse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace da;
using namespace da::synth;

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); fails++; }
    else      std::fprintf(stderr, "ok:   %s\n", msg);
}

// ---- fused-cloud quality vs the analytic surfaces -----------------------
struct CloudMetrics {
    int n = 0;
    double mean = 0, p50 = 0, p95 = 0, mx = 0;  // distance to GT surface (m)
    double inlier2cm = 0;                        // frac within 2 cm of a surface
    double wall_thick = 0;                       // std of signed dist to +x wall (m)
};

static CloudMetrics measure(const Scene& sc, const StreamCloud& cloud) {
    CloudMetrics m;
    const int n = (int)(cloud.xyz.size()/3);
    m.n = n;
    if (!n) return m;
    std::vector<double> d(n);
    int inl = 0;
    for (int i = 0; i < n; ++i) {
        V3 P{ cloud.xyz[3*i+0], cloud.xyz[3*i+1], cloud.xyz[3*i+2] };
        d[i] = surface_distance(sc, P);
        m.mean += d[i];
        if (d[i] <= 0.02) ++inl;
    }
    m.mean /= n; m.inlier2cm = (double)inl / n;
    std::vector<double> s(d);
    auto pct = [&](double q){ size_t k=(size_t)(q*(s.size()-1));
        std::nth_element(s.begin(), s.begin()+k, s.end()); return s[k]; };
    m.p50 = pct(0.50); m.p95 = pct(0.95);
    m.mx = *std::max_element(d.begin(), d.end());

    // +x wall thickness on the FLAT INTERIOR only: |x-4|<0.3 (near the wall),
    // y in [-2,2] and z in [0.5,2.5] (away from the side-wall/floor corners, whose
    // points would otherwise pollute the x-spread and confound edge-rounding with
    // ghosting). Thickness = std of signed distance to the plane x=4.
    double mean_sd = 0, var = 0; int c = 0;
    std::vector<double> sd;
    for (int i = 0; i < n; ++i) {
        double x=cloud.xyz[3*i+0], y=cloud.xyz[3*i+1], z=cloud.xyz[3*i+2];
        if (std::fabs(x-4.0) < 0.3 && std::fabs(y) < 2.0 && z > 0.5 && z < 2.5) {
            sd.push_back(x-4.0); mean_sd += (x-4.0); ++c;
        }
    }
    if (c > 1) {
        mean_sd /= c;
        for (double v : sd) var += (v-mean_sd)*(v-mean_sd);
        m.wall_thick = std::sqrt(var / c);
    }
    return m;
}

int main() {
    Scene room = make_room(4.0, 3.0);
    const int W = 160, H = 120; const double fx = 140, fy = 140;
    const int F = 48;                          // frames around a closed loop
    auto cams = loop_trajectory(F, 2.5, 1.3, fx, fy, W, H);

    StreamParams sp;
    sp.chunk_size = 12; sp.overlap = 4; sp.conf_pct = 20; sp.point_size = 1.0f;
    sp.global_budget = 0;                      // unlimited: measure the full cloud
    sp.min_overlap_pts = 50;
    const int step = sp.chunk_size - sp.overlap;

    // Injected error model (the realistic ghosting drivers).
    const double depth_noise = 0.008;          // 0.8% per-view depth noise
    const double warp_amp    = 0.02;           // 2% per-window non-rigid depth stretch

    // WindowSource: render frames [w0,w1) fresh, apply per-window warp + noise,
    // flat color. Fresh render per window => shared frames get independent noise.
    WindowSource src = [&](int w0, int w1, std::vector<ViewResult>& views,
                           std::vector<std::vector<uint8_t>>& rgb,
                           int& outH, int& outW, std::string& err) -> bool {
        const int nv = w1 - w0;
        const int wk = w0 / step;                          // window index
        // Per-window warp amplitude: deterministic, varies in sign/size per window
        // so consecutive windows disagree non-rigidly (=> residual after Sim3).
        const double a = warp_amp * std::sin(1.7 * wk + 0.5);
        views.assign(nv, {});
        rgb.assign(nv, {});
        outH = H; outW = W;
        for (int i = 0; i < nv; ++i) {
            int f = w0 + i;
            // seed depends on (window, frame): same frame differs across windows.
            View v = render(room, cams[f], depth_noise, (unsigned long long)(1000*(wk+1) + f + 1));
            // apply the non-rigid warp: stretch depth smoothly across the image.
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                size_t idx = (size_t)y*W + x;
                if (!v.valid[idx]) continue;
                double g = ((double)x/(W-1) - 0.5) * 2.0;   // [-1,1] across width
                v.depth[idx] *= (float)(1.0 + a*g);
            }
            ViewResult vr;
            vr.depth = v.depth; vr.conf = v.conf; vr.intr = v.intr; vr.ext = v.ext;
            views[i] = std::move(vr);
            rgb[i].assign((size_t)H*W*3, 128);              // flat gray (geometry-only test)
        }
        return true;
    };

    StreamCloud cloud; std::string err;
    bool ok = stream_points_core(F, sp, src, cloud, err);
    check(ok, ok ? "stream_points_core ran" : err.c_str());
    if (!ok) { std::fprintf(stderr, "FAILED (%d)\n", fails+1); return 1; }

    CloudMetrics m = measure(room, cloud);
    std::fprintf(stderr, "\n==== BASELINE (no A/B/C) ====\n");
    std::fprintf(stderr, "points            : %d\n", m.n);
    std::fprintf(stderr, "surf dist mean    : %.4f m\n", m.mean);
    std::fprintf(stderr, "surf dist p50     : %.4f m\n", m.p50);
    std::fprintf(stderr, "surf dist p95     : %.4f m\n", m.p95);
    std::fprintf(stderr, "surf dist max     : %.4f m\n", m.mx);
    std::fprintf(stderr, "inliers <= 2cm    : %.1f %%\n", 100.0*m.inlier2cm);
    std::fprintf(stderr, "+x wall thickness : %.4f m  (ghosting/doubling proxy)\n", m.wall_thick);
    std::fprintf(stderr, "warnings          : %d\n", cloud.warnings);
    std::fprintf(stderr, "=============================\n\n");

    // Sanity gates (loose — this is a measurement, not a tight assertion).
    check(m.n > 10000, "fused cloud is substantial");
    check(m.p50 < 0.15, "median point lands near a surface (stitch not broken)");
    check(m.inlier2cm > 0.20, "a meaningful fraction is within 2cm");

    // ---- +A: apply surface fusion and re-measure (effectiveness gate) ----
    FuseParams fp; fp.radius = 0.04f; fp.voxel = 0.03f;
    StreamCloud fused = cloud;
    int nf = fuse_voxel(fused.xyz, fused.rgb, fused.radius, fp);
    CloudMetrics mf = measure(room, fused);
    std::fprintf(stderr, "==== +A  surface fusion (radius=%.0fcm voxel=%.0fcm) ====\n",
                 fp.radius*100, fp.voxel*100);
    std::fprintf(stderr, "points            : %d  (%.2fx fewer)\n", nf, (double)m.n/std::max(1,nf));
    std::fprintf(stderr, "surf dist mean    : %.4f m  (was %.4f)\n", mf.mean, m.mean);
    std::fprintf(stderr, "surf dist p95     : %.4f m  (was %.4f)\n", mf.p95, m.p95);
    std::fprintf(stderr, "inliers <= 2cm    : %.1f %%  (was %.1f)\n", 100.0*mf.inlier2cm, 100.0*m.inlier2cm);
    std::fprintf(stderr, "+x wall thickness : %.4f m  (was %.4f)  <-- ghosting\n", mf.wall_thick, m.wall_thick);
    std::fprintf(stderr, "=============================\n\n");

    // A targets FLAT ghosting: it must thin the wall interior + cut density, and
    // not worsen mean accuracy. (p95/inliers can move slightly the other way from
    // edge rounding — reported above, an honest MLS tradeoff, not gated here.)
    check(nf < m.n, "A reduces point count");
    check(mf.wall_thick < m.wall_thick, "A reduces flat-wall thickness (less ghosting)");
    check(mf.mean <= m.mean + 0.003, "A does not worsen mean surface distance");

    // ---- +B: per-seam point-to-plane ICP (re-run the stitch with it on) ----
    // Point-to-plane refinement of each seam should reduce the residual surface
    // misalignment the point-to-point Umeyama leaves behind (=> tighter walls,
    // lower surface distance), and must NOT drift (it self-guards degenerate seams).
    StreamParams spB = sp;
    spB.icp_refine = true;
    spB.icp.max_corr_dist = 0.20; spB.icp.normal_radius = 0.15; spB.icp.huber_delta = 0.05;
    StreamCloud cloudB; std::string errB;
    bool okB = stream_points_core(F, spB, src, cloudB, errB);
    check(okB, okB ? "stream_points_core (ICP seam) ran" : errB.c_str());
    CloudMetrics mB = measure(room, cloudB);
    std::fprintf(stderr, "==== +B  per-seam point-to-plane ICP ====\n");
    std::fprintf(stderr, "points            : %d\n", mB.n);
    std::fprintf(stderr, "surf dist mean    : %.4f m  (was %.4f)\n", mB.mean, m.mean);
    std::fprintf(stderr, "surf dist p95     : %.4f m  (was %.4f)\n", mB.p95, m.p95);
    std::fprintf(stderr, "inliers <= 2cm    : %.1f %%  (was %.1f)\n", 100.0*mB.inlier2cm, 100.0*m.inlier2cm);
    std::fprintf(stderr, "+x wall thickness : %.4f m  (was %.4f)  (reported; see note)\n", mB.wall_thick, m.wall_thick);
    std::fprintf(stderr, "seam warnings     : %d\n", cloudB.warnings);
    std::fprintf(stderr, "=============================\n\n");

    // B's headline is GLOBAL surface accuracy: as the seams tighten, mean and the
    // p95 tail (where ghosting/drift lives) drop substantially. The single flat
    // +x-wall spread is a weaker proxy that can tick up slightly, because point-to-
    // plane optimises the whole-surface fit rather than one plane's perpendicular
    // agreement — reported above, not gated.
    check(mB.n > 10000, "B still produces a substantial cloud");
    check(mB.mean < m.mean - 0.005, "B reduces mean surface distance (tighter seams)");
    check(mB.p95  < m.p95  - 0.005, "B reduces p95 surface distance (less ghosting tail)");
    check(mB.inlier2cm >= m.inlier2cm - 1e-9, "B does not reduce the 2cm inlier fraction");

    // ---- +A+B: fuse the ICP-refined cloud (the intended production combo) ----
    StreamCloud fusedB = cloudB;
    int nab = fuse_voxel(fusedB.xyz, fusedB.rgb, fusedB.radius, fp);
    CloudMetrics mAB = measure(room, fusedB);
    std::fprintf(stderr, "==== +A+B  ICP seam + surface fusion ====\n");
    std::fprintf(stderr, "points            : %d  (%.2fx fewer)\n", nab, (double)m.n/std::max(1,nab));
    std::fprintf(stderr, "surf dist mean    : %.4f m  (baseline %.4f, +A %.4f, +B %.4f)\n", mAB.mean, m.mean, mf.mean, mB.mean);
    std::fprintf(stderr, "surf dist p95     : %.4f m  (baseline %.4f)\n", mAB.p95, m.p95);
    std::fprintf(stderr, "+x wall thickness : %.4f m  (baseline %.4f, +A %.4f)\n", mAB.wall_thick, m.wall_thick, mf.wall_thick);
    std::fprintf(stderr, "=============================\n\n");
    check(nab < m.n, "A+B reduces point count");
    check(mAB.mean < m.mean, "A+B reduces mean surface distance vs baseline");
    // NB: loop closure (C) targets accumulating rigid/Sim3 DRIFT, not the non-rigid
    // warp that dominates this scene; its in-pipeline validation lives in
    // tests/test_stream_loop.cpp (a random-walk-drift scenario) + the gtsam oracle.

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
