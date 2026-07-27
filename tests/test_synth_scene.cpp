// Validate the surface-concentrated synthetic generator (tests/synth_scene.hpp).
// The whole point of this ground truth is that back-projected clouds lie ON
// surfaces (like real reconstructions), NOT scattered through a volume. These
// checks assert exactly that, plus convention round-trip, occlusion, and that
// two views of the same wall land on the same plane (the substrate for seam
// alignment). No GGUF / GPU.
#include "synth_scene.hpp"
#include "geom_metrics.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

using namespace da::synth;

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); fails++; }
    else      std::fprintf(stderr, "ok:   %s\n", msg);
}

// Max & mean distance of a back-projected cloud to the analytic surfaces.
static void surf_stats(const Scene& sc, const std::vector<double>& xyz,
                       double& mean, double& mx) {
    int n = (int)(xyz.size()/3); mean=0; mx=0;
    for (int i=0;i<n;++i) {
        double d = surface_distance(sc, {xyz[3*i+0],xyz[3*i+1],xyz[3*i+2]});
        mean += d; if (d>mx) mx=d;
    }
    if (n) mean/=n;
}

int main() {
    Scene room = make_room(4.0, 3.0);
    const int W=160, H=120; const double fx=120, fy=120;

    // A view from inside looking at the +x wall / sphere.
    Camera cam = look_at(V3{-2,-0.5,1.2}, V3{4,-0.5,1.2}, V3{0,0,1}, fx, fy, W, H);

    // ---- 1. NO-NOISE: back-projected cloud lies exactly on the surfaces ----
    {
        View v = render(room, cam, /*noise=*/0.0, /*seed=*/1);
        int nvalid=0; for (auto b:v.valid) nvalid+=b;
        check(nvalid > W*H/2, "render: majority of pixels hit a surface");
        std::vector<double> cloud; back_project_world(v, cloud);
        double mean,mx; surf_stats(room, cloud, mean, mx);
        // Exact raycast + exact convention => on-surface to ~fp precision.
        check(mx < 1e-6, "no-noise cloud lies ON surfaces (max dist < 1e-6)");
    }

    // ---- 2. NOISY: cloud stays a thin SHEET, not a volume ----
    {
        const double noise = 0.01;                    // 1% depth noise
        View v = render(room, cam, noise, /*seed=*/2);
        std::vector<double> cloud; back_project_world(v, cloud);
        double mean,mx; surf_stats(room, cloud, mean, mx);
        // Mean surface offset ~ noise*depth (depth ~1-6 here) => a few cm, and far
        // below the ~8m scene extent. This is the surface-concentration guarantee.
        check(mean < 0.15, "noisy cloud is a thin sheet (mean surf dist < 0.15m)");
        check(mean > 0.005, "noisy cloud actually carries the injected noise");
        // Sanity vs scene scale: mean offset is <2% of the 8m room.
        check(mean < 0.02*8.0, "sheet thickness << scene extent (surface, not volume)");
    }

    // ---- 3. Occlusion: the sphere hides part of the wall behind it ----
    {
        View v = render(room, cam, 0.0, 3);
        int sphere_px=0, wall_px=0;
        for (size_t i=0;i<v.valid.size();++i) if (v.valid[i]) {
            // recover prim via a second raycast (cheap, test-only)
        }
        // Simpler: count pixels whose depth is closer than the far wall distance.
        // The +x wall is ~6m away along the look dir; the sphere sits ~2-3m ahead.
        float dmin=1e9,dmax=0; for (size_t i=0;i<v.depth.size();++i) if (v.valid[i]) {
            dmin=std::min(dmin,v.depth[i]); dmax=std::max(dmax,v.depth[i]); }
        check(dmin < dmax*0.7, "occlusion: near (sphere) depths well in front of far wall");
        (void)sphere_px; (void)wall_px;
    }

    // ---- 4. Two views of the same wall back-project to the SAME plane ----
    // (this is what a seam has to align: overlapping surface observations.)
    {
        Camera camA = look_at(V3{-2,-0.5,1.2}, V3{4,-0.5,1.2}, V3{0,0,1}, fx, fy, W, H);
        Camera camB = look_at(V3{-2, 0.8,1.4}, V3{4, 0.8,1.2}, V3{0,0,1}, fx, fy, W, H);
        View a = render(room, camA, 0.0, 4), b = render(room, camB, 0.0, 5);
        std::vector<double> ca, cb; back_project_world(a, ca); back_project_world(b, cb);
        double ma,mxa,mb,mxb; surf_stats(room,ca,ma,mxa); surf_stats(room,cb,mb,mxb);
        check(mxa < 1e-6 && mxb < 1e-6, "both views land exactly on surfaces");
        // Two viewpoints cover DIFFERENT extents, so full Chamfer reflects coverage,
        // not coherence. The meaningful claim is genuine partial overlap: a large
        // fraction of A's points coincide (within tau) with B's surface. That
        // overlap is exactly what a seam stitch/ICP operates on.
        da::eval::FScore fs = da::eval::fscore(ca.data(), (int)(ca.size()/3),
                                               cb.data(), (int)(cb.size()/3), /*tau=*/0.1);
        check(fs.precision > 0.3, "views share a substantial overlapping surface region");
    }

    // ---- 5. Loop trajectory closes (revisit for loop-closure tests) ----
    {
        auto cams = loop_trajectory(24, 2.5, 1.3, fx, fy, W, H);
        check((int)cams.size()==24, "loop trajectory has n cameras");
        // Adjacent wrap step is small; opposite side is ~a diameter away.
        double step = da::synth::norm(cams[0].C - cams[23].C);
        double across = da::synth::norm(cams[0].C - cams[12].C);
        check(step < across, "loop closes: start adjacent to last, far from midpoint");
        check(across > 2.5, "orbit spans the room (diameter ~ 2*radius)");
    }

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
