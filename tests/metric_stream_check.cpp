// End-to-end check for METRIC STREAMING (not a ctest). Runs da_capi_points_stream
// twice on a nested engine — relative (metric=0) then metric (metric=1) — and reports
// each cloud's bbox diagonal + median point distance. The metric cloud should be the
// relative one uniformly rescaled to a plausible metre scale (bbox a few m, median
// distance ~1 m), and metric/relative bbox ratio ~ the applied scale_factor.
//
//   metric_stream_check <anyview.gguf> <metric.gguf> <frames_dir> [n=24]
#include "da_capi.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void cloud_stats(const float* xyz, int n, double& diag, double& med_dist) {
    if (n <= 0) { diag = med_dist = 0; return; }
    double lo[3] = {xyz[0],xyz[1],xyz[2]}, hi[3] = {xyz[0],xyz[1],xyz[2]}, c[3] = {0,0,0};
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], (double)xyz[3*i+k]); hi[k] = std::max(hi[k], (double)xyz[3*i+k]); c[k] += xyz[3*i+k]; }
    for (int k = 0; k < 3; ++k) c[k] /= n;
    diag = std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0])+(hi[1]-lo[1])*(hi[1]-lo[1])+(hi[2]-lo[2])*(hi[2]-lo[2]));
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) { double a=xyz[3*i]-c[0],b=xyz[3*i+1]-c[1],e=xyz[3*i+2]-c[2]; d[i]=std::sqrt(a*a+b*b+e*e); }
    std::nth_element(d.begin(), d.begin()+n/2, d.end());
    med_dist = d[n/2];
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: metric_stream_check <anyview.gguf> <metric.gguf> <frames_dir> [n]\n"); return 2; }
    const int n_want = (argc > 4) ? std::atoi(argv[4]) : 24;
    std::vector<std::string> all;
    for (auto& e : fs::directory_iterator(argv[3])) {
        auto x = e.path().extension().string();
        if (x==".jpg"||x==".jpeg"||x==".png"||x==".JPG"||x==".PNG") all.push_back(e.path().string());
    }
    std::sort(all.begin(), all.end());
    int stride = std::max(1, (int)all.size()/n_want);
    std::vector<std::string> sel; for (size_t i=0;i<all.size() && (int)sel.size()<n_want;i+=stride) sel.push_back(all[i]);
    std::vector<const char*> paths; for (auto& s : sel) paths.push_back(s.c_str());
    std::fprintf(stderr, "%d frames\n", (int)paths.size());

    da_ctx* ctx = da_capi_load_nested(argv[1], argv[2], 0);
    if (!ctx) { std::fprintf(stderr, "load_nested failed\n"); return 1; }

    for (int metric = 0; metric <= 1; ++metric) {
        int n = 0; float *xyz=nullptr, *rad=nullptr; unsigned char* rgb=nullptr;
        std::vector<int> counts(paths.size(), 0);
        int rc = da_capi_points_stream(ctx, paths.data(), (int)paths.size(),
                                       12, 3, 55.0, 1.2f, 4'000'000,
                                       0, 0, 0, metric, 0.0,
                                       &n, counts.data(), &xyz, &rgb, &rad);
        if (rc != 0) { std::fprintf(stderr, "metric=%d FAILED: %s\n", metric, da_capi_last_error(ctx)); continue; }
        double diag=0, med=0; cloud_stats(xyz, n, diag, med);
        // median radius
        std::vector<float> r(rad, rad+n); std::nth_element(r.begin(), r.begin()+n/2, r.end());
        std::printf("metric=%d : n=%d  bbox_diag=%.4f  median_dist=%.4f  median_radius=%.5f\n",
                    metric, n, diag, med, r[n/2]);
        da_capi_free_floats(xyz); da_capi_free_bytes(rgb); da_capi_free_floats(rad);
    }
    da_capi_free(ctx);
    return 0;
}
