// Prototype/measurement for METRIC STREAMING (not a ctest).
// Mirrors the streaming path: ONE multi-view anyview pass (depth_pose_multi, ~504px)
// gives per-view relative depth+conf+pose; the metric ViT-L branch (metric_branch,
// same 504px) gives raw metric depth+sky per frame; NestedAligner::align then fits
// the per-frame scale_factor mapping THIS window's anyview depth -> metres.
// Reports per-frame scale_factor + metric-depth stats and the cross-frame coefficient
// of variation. Low CV => a single per-window scalar rescales the window to metric
// robustly. Also a VRAM feasibility check: if this fits, metric streaming fits.
//
//   metric_probe <anyview.gguf> <metric.gguf> <frames_dir> [n=12]
#include "engine.hpp"
#include "image_io.hpp"
#include "nested.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static double median(std::vector<float> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}
static double pctl(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::min((double)v.size()-1, std::max(0.0, q*(v.size()-1)));
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: metric_probe <anyview.gguf> <metric.gguf> <frames_dir> [n]\n"); return 2; }
    const char* anyv = argv[1];
    const char* metr = argv[2];
    const char* dir  = argv[3];
    const int n_want = (argc > 4) ? std::atoi(argv[4]) : 12;

    std::vector<std::string> all;
    for (auto& e : fs::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        if (ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".JPG"||ext==".PNG") all.push_back(e.path().string());
    }
    std::sort(all.begin(), all.end());
    if (all.empty()) { std::fprintf(stderr, "no frames in %s\n", dir); return 1; }
    std::vector<std::string> sel(all.begin(), all.begin() + std::min((size_t)n_want, all.size()));

    auto eng = da::Engine::load_nested(anyv, metr, 0);
    if (!eng) { std::fprintf(stderr, "load_nested failed\n"); return 1; }
    std::fprintf(stderr, "loaded nested; is_nested=%d\n", (int)eng->is_nested());

    std::vector<da::Image> imgs(sel.size());
    for (size_t i = 0; i < sel.size(); ++i)
        if (!da::load_image_rgb(sel[i], imgs[i])) { std::fprintf(stderr, "load %s failed\n", sel[i].c_str()); return 1; }

    // ---- ONE multi-view anyview pass (the streaming pass) ----
    std::vector<da::ViewResult> views; int H=0, W=0;
    if (!eng->depth_pose_multi(imgs, views, H, W)) { std::fprintf(stderr, "depth_pose_multi failed\n"); return 1; }
    std::printf("multi-view anyview: %d views @ %dx%d\n", (int)views.size(), W, H);

    std::printf("%-4s %-12s %-12s %-12s %-12s %-12s\n", "frm", "scale_fac", "any_med", "met_med", "met_p90", "met_max");
    std::vector<float> sfs, meds;
    for (size_t i = 0; i < views.size(); ++i) {
        // metric branch for this frame (same 504px preprocess_real -> pixel-aligned).
        std::vector<float> mraw, sky; int Hm=0, Wm=0;
        if (!eng->metric_branch(imgs[i], mraw, sky, Hm, Wm)) { std::fprintf(stderr, "metric_branch %zu failed\n", i); continue; }
        if (Hm != H || Wm != W || (int)mraw.size() != H*W) { std::fprintf(stderr, "frame %zu res mismatch (%dx%d vs %dx%d)\n", i, Wm, Hm, W, H); continue; }

        da::AnyviewOut any;
        any.depth = views[i].depth;
        any.depth_conf = views[i].conf;
        any.extrinsics = views[i].ext;
        any.intrinsics = views[i].intr;
        da::MetricOut m; m.depth = mraw; m.sky = sky;
        da::NestedOut no = da::NestedAligner::align(any, m, H, W);

        std::vector<float> ad; for (float x : views[i].depth) if (std::isfinite(x)&&x>0) ad.push_back(x);
        std::vector<float> md; for (float x : no.depth)       if (std::isfinite(x)&&x>0) md.push_back(x);
        std::printf("%-4zu %-12.5f %-12.4f %-12.4f %-12.4f %-12.4f\n",
                    i, no.scale_factor, median(ad), median(md), pctl(md,0.90), pctl(md,1.0));
        sfs.push_back(no.scale_factor);
        meds.push_back((float)median(md));
    }

    auto stats = [](const std::vector<float>& v, double& mean, double& cv){
        if (v.empty()){ mean=0; cv=0; return; }
        double s=0; for (float x:v) s+=x; mean=s/v.size();
        double var=0; for (float x:v) var+=(x-mean)*(x-mean); var/=v.size();
        cv = mean!=0 ? std::sqrt(var)/std::fabs(mean) : 0;
    };
    double sf_mean, sf_cv, md_mean, md_cv;
    stats(sfs, sf_mean, sf_cv);
    stats(meds, md_mean, md_cv);
    std::printf("\n== consistency across %zu frames ==\n", sfs.size());
    std::printf("scale_factor : mean=%.5f  CV=%.1f%%\n", sf_mean, 100*sf_cv);
    std::printf("metric median: mean=%.4f m  CV=%.1f%%   <-- low CV => per-window scalar is robust\n", md_mean, 100*md_cv);
    return 0;
}
