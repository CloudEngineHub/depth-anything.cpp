// Report the MAX metric depth the model predicts across one or more clips (not a
// ctest). Mirrors the streaming metric path: slide chunk/overlap windows, run the
// multi-view anyview pass + the metric ViT-L branch on ~6 frames/window -> median
// scale K, then metric_depth = anyview_depth * K per frame. Reports, per clip and
// overall: the absolute max, p99.9 (de-spiked "farthest real-ish surface"), and
// median metric depth (metres). Loads the nested model once, scans every clip dir.
//
//   metric_maxdepth <anyview.gguf> <metric.gguf> <frames_dir> [<frames_dir> ...]
#include "engine.hpp"
#include "image_io.hpp"
#include "nested.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static float pctl(std::vector<float>& v, double q) {
    if (v.empty()) return 0;
    size_t i = (size_t)std::min((double)v.size()-1, std::max(0.0, q*(v.size()-1)));
    std::nth_element(v.begin(), v.begin()+i, v.end());
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: metric_maxdepth <anyview.gguf> <metric.gguf> <dir> [<dir> ...]\n"); return 2; }
    auto eng = da::Engine::load_nested(argv[1], argv[2], 0);
    if (!eng) { std::fprintf(stderr, "load_nested failed\n"); return 1; }

    const int CHUNK = 12, OVERLAP = 3, STEP = CHUNK - OVERLAP;
    double overall_absmax = 0, overall_p999 = 0;

    std::printf("%-34s %8s %10s %10s %10s\n", "clip", "frames", "median_m", "p99.9_m", "MAX_m");
    std::printf("--------------------------------------------------------------------------------\n");
    for (int d = 3; d < argc; ++d) {
        std::vector<std::string> frames;
        for (auto& e : fs::directory_iterator(argv[d])) {
            auto x = e.path().extension().string();
            if (x==".jpg"||x==".jpeg"||x==".png"||x==".JPG"||x==".PNG") frames.push_back(e.path().string());
        }
        std::sort(frames.begin(), frames.end());
        const int F = (int)frames.size();
        if (F < 2) { std::fprintf(stderr, "skip %s (%d frames)\n", argv[d], F); continue; }

        double clip_absmax = 0;
        std::vector<float> clip_med, clip_p999;   // per-frame stats aggregated
        int used_frames = 0;

        for (int w0 = 0; w0 < F; w0 += STEP) {
            const int w1 = std::min(w0 + CHUNK, F);
            const int nv = w1 - w0;
            std::vector<da::Image> imgs(nv);
            bool ok = true;
            for (int i = 0; i < nv; ++i) if (!da::load_image_rgb(frames[w0+i], imgs[i])) { ok = false; break; }
            if (!ok) { std::fprintf(stderr, "  load fail @%d\n", w0); continue; }
            std::vector<da::ViewResult> views; int H=0, W=0;
            if (!eng->depth_pose_multi(imgs, views, H, W)) { std::fprintf(stderr, "  dpm fail @%d\n", w0); continue; }

            // per-window median metric scale (same as streaming)
            std::vector<double> scales;
            const int mstride = std::max(1, nv/6);
            for (int i = 0; i < nv; i += mstride) {
                std::vector<float> mraw, sky; int Hm=0, Wm=0;
                if (!eng->metric_branch(imgs[i], mraw, sky, Hm, Wm) || Hm!=H || Wm!=W) continue;
                da::AnyviewOut any; any.depth=views[i].depth; any.depth_conf=views[i].conf;
                any.extrinsics=views[i].ext; any.intrinsics=views[i].intr;
                da::MetricOut m; m.depth=std::move(mraw); m.sky=std::move(sky);
                da::NestedOut no = da::NestedAligner::align(any, m, H, W);
                if (no.scale_factor>0 && std::isfinite(no.scale_factor)) scales.push_back(no.scale_factor);
            }
            if (scales.empty()) continue;
            std::sort(scales.begin(), scales.end());
            const double K = scales[scales.size()/2];

            // only the NEW frames of this window (avoid double-counting the overlap)
            const int new_lo = (w0 == 0) ? 0 : OVERLAP;
            for (int i = new_lo; i < nv; ++i) {
                std::vector<float> md; md.reserve(views[i].depth.size());
                for (float x : views[i].depth) if (std::isfinite(x) && x > 0) md.push_back(x * (float)K);
                if (md.empty()) continue;
                float fmax = 0; for (float x : md) fmax = std::max(fmax, x);
                clip_absmax = std::max(clip_absmax, (double)fmax);
                clip_med.push_back(pctl(md, 0.50));
                clip_p999.push_back(pctl(md, 0.999));
                ++used_frames;
            }
            if (w1 >= F) break;
        }

        float med = clip_med.empty()?0:pctl(clip_med, 0.5);
        float p999 = 0; for (float x : clip_p999) p999 = std::max(p999, x);   // worst-case frame p99.9
        std::string name = fs::path(argv[d]).parent_path().filename().string();
        if (name.empty()) name = fs::path(argv[d]).filename().string();
        std::printf("%-34s %8d %10.2f %10.2f %10.2f\n", name.c_str(), used_frames, med, p999, clip_absmax);
        overall_absmax = std::max(overall_absmax, clip_absmax);
        overall_p999   = std::max(overall_p999, (double)p999);
    }
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("OVERALL: worst-frame p99.9 = %.2f m   absolute max = %.2f m\n", overall_p999, overall_absmax);
    return 0;
}
