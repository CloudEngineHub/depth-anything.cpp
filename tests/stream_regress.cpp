// Real-clip regression harness for the streaming de-ghosting toggles (tasks A/B/C).
// NOT a ctest — it needs a pose-capable DA3 model + a real frame sequence. Runs
// da_capi_points_stream over the frames with each A/B/C variant and reports, per
// variant: point count, wall-clock time, and a GROUND-TRUTH-FREE ghosting proxy —
// the mean LOCAL SURFACE THICKNESS. For each sampled point we PCA its radius
// neighbourhood; the smallest eigenvalue's sqrt is the local sheet thickness. A
// doubled/misaligned surface reads as thick; a single clean sheet reads as thin.
// So B (seam ICP) and C (loop closure) should THIN it by aligning the sheets, and
// A (fusion) should thin it (and cut the count) by merging them.
//
//   stream_regress <model.gguf> <frames_dir> [max_frames=48]
#include "da_capi.h"
#include "spatial_hash.hpp"
#include "linalg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Mean local surface thickness (m): sqrt of the smallest PCA eigenvalue over each
// sampled point's radius neighbourhood, averaged. GT-free ghosting/spread proxy.
static double surface_thickness(const float* xyz, int N, double radius, int max_q) {
    if (N < 16) return 0.0;
    std::vector<double> pts((size_t)3*N);
    for (int i = 0; i < 3*N; ++i) pts[i] = xyz[i];
    da::SpatialHash hash(pts.data(), N, radius);
    int stride = std::max(1, N / std::max(1, max_q));
    std::vector<int> nb;
    double sum = 0; long cnt = 0;
    for (int i = 0; i < N; i += stride) {
        nb.clear();
        hash.radius(&pts[3*i], radius, nb);
        if ((int)nb.size() < 8) continue;
        double mu[3] = {0,0,0};
        for (int j : nb) { mu[0]+=pts[3*j]; mu[1]+=pts[3*j+1]; mu[2]+=pts[3*j+2]; }
        double inv = 1.0/nb.size(); mu[0]*=inv; mu[1]*=inv; mu[2]*=inv;
        double C[9] = {0,0,0,0,0,0,0,0,0};
        for (int j : nb) {
            double a=pts[3*j]-mu[0], b=pts[3*j+1]-mu[1], c=pts[3*j+2]-mu[2];
            C[0]+=a*a; C[1]+=a*b; C[2]+=a*c; C[4]+=b*b; C[5]+=b*c; C[8]+=c*c;
        }
        C[3]=C[1]; C[6]=C[2]; C[7]=C[5];
        for (int k=0;k<9;++k) C[k]*=inv;
        double ev[3], evec[9]; da::linalg::jacobi_eigen_sym(C, 3, ev, evec);
        double lmin = std::min(ev[0], std::min(ev[1], ev[2]));
        sum += std::sqrt(std::max(0.0, lmin)); ++cnt;
    }
    return cnt ? sum / cnt : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: stream_regress <model.gguf> <frames_dir> [max_frames]\n"); return 2; }
    const char* model = argv[1];
    const char* dir = argv[2];
    const int max_frames = (argc > 3) ? std::atoi(argv[3]) : 48;

    // Enumerate frames (sorted); even stride down to max_frames.
    std::vector<std::string> all;
    for (auto& e : fs::directory_iterator(dir)) {
        auto p = e.path().string();
        auto ext = e.path().extension().string();
        if (ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".JPG"||ext==".PNG") all.push_back(p);
    }
    std::sort(all.begin(), all.end());
    if (all.size() < 2) { std::fprintf(stderr, "need >=2 frames in %s (found %zu)\n", dir, all.size()); return 1; }
    int stride = std::max(1, (int)(all.size() + max_frames - 1) / max_frames);
    std::vector<std::string> sel;
    for (size_t i = 0; i < all.size() && (int)sel.size() < max_frames; i += stride) sel.push_back(all[i]);
    std::vector<const char*> paths; for (auto& s : sel) paths.push_back(s.c_str());
    std::fprintf(stderr, "regress: %d frames from %s\n", (int)sel.size(), dir);

    da_ctx* ctx = da_capi_load(model, 0);
    if (!ctx) { std::fprintf(stderr, "load failed: %s\n", model); return 1; }

    struct Variant { const char* name; int icp, loop, fuse; };
    const Variant variants[] = {
        {"baseline   ", 0,0,0},
        {"+B icp     ", 1,0,0},
        {"+C loop    ", 0,1,0},
        {"+A fuse    ", 0,0,1},
        {"+A+B+C     ", 1,1,1},
    };

    // Monocular reconstructions live at an arbitrary metric scale, so the fusion
    // voxel and PCA radius must be SCENE-RELATIVE, not fixed centimetres. Derive
    // them from the baseline cloud's bounding-box diagonal (run the baseline once
    // up front to measure it).
    double thick_radius = 0.05, scene_voxel = 0.03, diag = 0.0;
    {
        int n=0; int* counts=(int*)std::calloc(sel.size(), sizeof(int));
        float *xyz=nullptr,*rad=nullptr; unsigned char* rgb=nullptr;
        if (da_capi_points_stream(ctx, paths.data(),(int)paths.size(), 12,3,55.0,1.2f,4'000'000,
                                  0,0,0,0,0.0, &n,counts,&xyz,&rgb,&rad)==0 && n>0){
            float lo[3]={xyz[0],xyz[1],xyz[2]}, hi[3]={xyz[0],xyz[1],xyz[2]};
            for (int i=0;i<n;++i) for(int k=0;k<3;++k){ lo[k]=std::min(lo[k],xyz[3*i+k]); hi[k]=std::max(hi[k],xyz[3*i+k]); }
            diag = std::sqrt((double)(hi[0]-lo[0])*(hi[0]-lo[0])+(hi[1]-lo[1])*(hi[1]-lo[1])+(hi[2]-lo[2])*(hi[2]-lo[2]));
            thick_radius = 0.010*diag;   // PCA neighbourhood ~1% of extent
            scene_voxel  = 0.004*diag;   // fusion cell ~0.4% of extent
            da_capi_free_floats(xyz); da_capi_free_bytes(rgb); da_capi_free_floats(rad);
        }
        std::free(counts);
    }
    std::fprintf(stderr, "scene bbox diag=%.3f  =>  fuse voxel=%.4f  thickness radius=%.4f\n",
                 diag, scene_voxel, thick_radius);

    // Report absolute thickness AND thickness normalised by each variant's own bbox
    // diagonal. A pose-graph Sim3 can globally rescale the cloud, which would thin
    // neighbourhoods trivially; the normalised column (thickness/diag, in per-mille)
    // is rescale-invariant, so a drop there is genuine local de-ghosting, not shrink.
    std::fprintf(stderr, "\n%-12s %10s %10s %13s %13s %9s\n",
                 "variant", "points", "bbox_diag", "thick(x1e3)", "norm(x1e3)", "time(s)");
    std::fprintf(stderr, "--------------------------------------------------------------------------\n");
    for (const auto& v : variants) {
        int n = 0; int* counts = (int*)std::calloc(sel.size(), sizeof(int));
        float *xyz=nullptr, *rad=nullptr; unsigned char* rgb=nullptr;
        auto t0 = std::chrono::steady_clock::now();
        int rc = da_capi_points_stream(ctx, paths.data(), (int)paths.size(),
                                       12, 3, 55.0, 1.2f, 4'000'000,
                                       v.icp, v.loop, v.fuse, 0, scene_voxel,
                                       &n, counts, &xyz, &rgb, &rad);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1-t0).count();
        if (rc != 0) { std::fprintf(stderr, "%-12s FAILED: %s\n", v.name, da_capi_last_error(ctx)); std::free(counts); continue; }
        double thick = surface_thickness(xyz, n, thick_radius, 8000);
        double vlo[3]={xyz[0],xyz[1],xyz[2]}, vhi[3]={xyz[0],xyz[1],xyz[2]};
        for (int i=0;i<n;++i) for(int k=0;k<3;++k){ vlo[k]=std::min(vlo[k],(double)xyz[3*i+k]); vhi[k]=std::max(vhi[k],(double)xyz[3*i+k]); }
        double vdiag = std::sqrt((vhi[0]-vlo[0])*(vhi[0]-vlo[0])+(vhi[1]-vlo[1])*(vhi[1]-vlo[1])+(vhi[2]-vlo[2])*(vhi[2]-vlo[2]));
        double norm = vdiag > 0 ? thick / vdiag * 1000.0 : 0.0;   // per-mille of extent
        std::fprintf(stderr, "%-12s %10d %10.3f %13.3f %13.4f %9.1f\n",
                     v.name, n, vdiag, thick*1000.0, norm, secs);
        da_capi_free_floats(xyz); da_capi_free_bytes(rgb); da_capi_free_floats(rad); std::free(counts);
    }
    std::fprintf(stderr, "--------------------------------------------------------------------------\n");
    da_capi_free(ctx);
    return 0;
}
