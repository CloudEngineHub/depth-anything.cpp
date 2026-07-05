// In-pipeline validation for loop closure + Sim3 pose-graph (task C), on SURFACE
// ground truth. Unlike test_stream_synth (whose non-rigid warp is A/B's target),
// this drives the REAL stitcher over a CLOSED loop with the error mode loop
// closure actually fixes: per-window independent overlap noise, which makes the
// chained Sim3 seams RANDOM-WALK away from truth (open-loop drift). The last
// windows revisit the first windows' walls, so a detected loop closure + pose-
// graph optimization should redistribute that drift and cut the trajectory ATE.
#include "synth_scene.hpp"
#include "geom_metrics.hpp"
#include "stream.hpp"

#include <array>
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

static std::vector<std::array<double,3>> to_pos(const std::vector<float>& wp) {
    std::vector<std::array<double,3>> v(wp.size()/3);
    for (size_t i=0;i<v.size();++i) v[i]={wp[3*i],wp[3*i+1],wp[3*i+2]};
    return v;
}

// Displace a frame's back-projected WORLD points by a small rigid transform
// T=(Rt,tt): equivalently move the camera pose so back_project(ext') = T∘back_project(ext).
// ext is world->cam [R|t] 3x4 row-major. C=-R^T t; C'=Rt C+tt; R'=R Rt^T; t'=-R' C'.
static void perturb_ext(std::array<float,12>& ext, const double Rt[9], const double tt[3]) {
    double R[9]={ext[0],ext[1],ext[2], ext[4],ext[5],ext[6], ext[8],ext[9],ext[10]};
    double t[3]={ext[3],ext[7],ext[11]};
    double C[3]={ -(R[0]*t[0]+R[3]*t[1]+R[6]*t[2]),
                  -(R[1]*t[0]+R[4]*t[1]+R[7]*t[2]),
                  -(R[2]*t[0]+R[5]*t[1]+R[8]*t[2]) };
    double Cn[3]={ Rt[0]*C[0]+Rt[1]*C[1]+Rt[2]*C[2]+tt[0],
                   Rt[3]*C[0]+Rt[4]*C[1]+Rt[5]*C[2]+tt[1],
                   Rt[6]*C[0]+Rt[7]*C[1]+Rt[8]*C[2]+tt[2] };
    double Rn[9];                                  // R' = R * Rt^T
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
        Rn[r*3+c] = R[r*3+0]*Rt[c*3+0] + R[r*3+1]*Rt[c*3+1] + R[r*3+2]*Rt[c*3+2];
    double tn[3]={ -(Rn[0]*Cn[0]+Rn[1]*Cn[1]+Rn[2]*Cn[2]),
                   -(Rn[3]*Cn[0]+Rn[4]*Cn[1]+Rn[5]*Cn[2]),
                   -(Rn[6]*Cn[0]+Rn[7]*Cn[1]+Rn[8]*Cn[2]) };
    ext={ (float)Rn[0],(float)Rn[1],(float)Rn[2],(float)tn[0],
          (float)Rn[3],(float)Rn[4],(float)Rn[5],(float)tn[1],
          (float)Rn[6],(float)Rn[7],(float)Rn[8],(float)tn[2] };
}

int main() {
    Scene room = make_room(4.0, 3.0);
    const int W = 160, H = 120; const double fx = 140, fy = 140;
    // Orbit period 48 frames but capture 60: the last window (frames 48-59) re-
    // traverses the SAME angles as window 0 (frames 0-11), so the revisit overlap
    // is a full window — a substantial, geometrically-rich loop closure (not a
    // thin wrap sliver). Cameras look outward from a circle.
    const int F = 60, N_orbit = 48; const double radius = 2.5, zc = 1.3;
    const double TWO_PI = 6.283185307179586;
    std::vector<Camera> cams; cams.reserve(F);
    for (int i = 0; i < F; ++i) {
        double a = TWO_PI * i / N_orbit;
        V3 eye{ radius*std::cos(a), radius*std::sin(a), zc };
        V3 tgt{ 2*radius*std::cos(a), 2*radius*std::sin(a), zc*0.8 };
        cams.push_back(look_at(eye, tgt, V3{0,0,1}, fx, fy, W, H));
    }

    StreamParams sp;
    sp.chunk_size = 12; sp.overlap = 4; sp.conf_pct = 20; sp.point_size = 1.0f;
    sp.global_budget = 0; sp.min_overlap_pts = 50;
    const int step = sp.chunk_size - sp.overlap;

    const double depth_noise = 0.005;          // mild per-view noise (realism)
    // Systematic per-window overlap-frame pose bias gamma: each new window
    // estimates the SHARED (head) frames' pose slightly off — a small rigid error
    // (0.6deg yaw + 2cm) applied to the overlap frames only. The seam Umeyama then
    // reads this as the relative motion, so it enters G every window and
    // ACCUMULATES into rigid loop drift (exactly what loop closure removes). NOT
    // applied to window 0 (it defines the frame).
    const double gy = 0.6 * M_PI / 180.0;
    const double Rt[9] = { std::cos(gy),-std::sin(gy),0, std::sin(gy),std::cos(gy),0, 0,0,1 };
    const double tt[3] = { 0.02, 0.0, 0.0 };

    WindowSource src = [&](int w0, int w1, std::vector<ViewResult>& views,
                           std::vector<std::vector<uint8_t>>& rgb,
                           int& outH, int& outW, std::string& err) -> bool {
        const int nv = w1 - w0; const int wk = w0 / step;
        views.assign(nv, {}); rgb.assign(nv, {}); outH = H; outW = W;
        for (int i = 0; i < nv; ++i) {
            int f = w0 + i;
            View v = render(room, cams[f], depth_noise, (unsigned long long)(1000*(wk+1) + f + 1));
            ViewResult vr; vr.depth = v.depth; vr.conf = v.conf; vr.intr = v.intr; vr.ext = v.ext;
            if (wk > 0 && i < sp.overlap) perturb_ext(vr.ext, Rt, tt);   // bias the head/overlap frames
            views[i] = std::move(vr);
            rgb[i].assign((size_t)H*W*3, 128);
        }
        return true;
    };

    auto gt_pos = [&](const std::vector<int>& mid) {
        std::vector<std::array<double,3>> v(mid.size());
        for (size_t i=0;i<mid.size();++i){ const auto& c=cams[mid[i]].C; v[i]={c.x,c.y,c.z}; }
        return v;
    };

    // ---- open loop (chained seams only) ----
    StreamParams spOpen = sp; spOpen.loop_close = false;
    StreamCloud c0; std::string e0;
    bool ok0 = stream_points_core(F, spOpen, src, c0, e0);
    check(ok0, ok0 ? "open-loop stream ran" : e0.c_str());
    double ate_open = eval::ate_rmse(to_pos(c0.window_pos), gt_pos(c0.window_mid_frame));

    // ---- loop closure + pose graph ----
    // Generous proposal radius (drift pushes the revisit frames apart); measure_loop
    // verifies via a tight correspondence gate, so false proposals are rejected.
    StreamParams spLoop = sp; spLoop.loop_close = true; spLoop.loop_pos_thr = 3.0;
    StreamCloud c1; std::string e1;
    bool ok1 = stream_points_core(F, spLoop, src, c1, e1);
    check(ok1, ok1 ? "loop-closed stream ran" : e1.c_str());
    double ate_closed = eval::ate_rmse(to_pos(c1.window_pos), gt_pos(c1.window_mid_frame));

    std::fprintf(stderr, "\n==== loop closure (task C) ====\n");
    std::fprintf(stderr, "windows           : %d\n", (int)c0.window_mid_frame.size());
    std::fprintf(stderr, "loops accepted    : %d\n", c1.loops_found);
    std::fprintf(stderr, "trajectory ATE    : %.4f m -> %.4f m  (%.0f%% cut)\n",
                 ate_open, ate_closed, 100.0*(1.0 - ate_closed/std::max(ate_open,1e-9)));
    std::fprintf(stderr, "===============================\n\n");

    // A single revisit-at-the-end constrains the loop's far windows; the mid-loop
    // windows (farthest from both the anchor and the closure) still carry residual
    // drift, so the whole-trajectory ATE improvement is partial (this is correct
    // pose-graph behaviour, not a shortfall). Gate on a robust >15% cut.
    check(c1.loops_found > 0, "C: detected + accepted a loop closure on the revisit");
    check(ate_open > 0.02, "C: open-loop chaining actually drifts (measurable ATE)");
    check(ate_closed < 0.85 * ate_open, "C: loop closure cuts trajectory ATE by >15%");

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
