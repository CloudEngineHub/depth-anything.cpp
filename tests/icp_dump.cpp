// Standalone harness for scripts/oracle_icp.py (NOT a ctest): reads a source and
// target point cloud, estimates target normals, runs src/icp.cpp point-to-plane
// ICP from an identity init, and writes the recovered transform so the Python
// oracle can diff it against open3d.pipelines.registration.registration_icp
// (TransformationEstimationPointToPlane).
//
// Cloud format (little-endian): int32 N, then 3N float32 xyz.
// Output format: 13 float64 [s, R0..R8, t0..t2], then 2 float64 [rms_before,
// rms_after], then 2 int32 [rank, iters].
//   icp_dump <src.bin> <tgt.bin> <out.bin> [max_corr=0.1] [normal_radius=0.06]
#include "icp.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool read_cloud(const char* path, std::vector<double>& xyz) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    int32_t N = 0;
    if (std::fread(&N, sizeof(int32_t), 1, f) != 1 || N < 0) { std::fclose(f); return false; }
    std::vector<float> tmp((size_t)3*N);
    if (N > 0 && std::fread(tmp.data(), sizeof(float), (size_t)3*N, f) != (size_t)3*N) { std::fclose(f); return false; }
    std::fclose(f);
    xyz.resize((size_t)3*N);
    for (size_t i = 0; i < tmp.size(); ++i) xyz[i] = tmp[i];
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: icp_dump <src> <tgt> <out> [max_corr] [normal_radius]\n"); return 2; }
    std::vector<double> src, tgt;
    if (!read_cloud(argv[1], src) || !read_cloud(argv[2], tgt)) return 1;

    da::IcpParams p;
    p.max_corr_dist = (argc > 4) ? std::atof(argv[4]) : 0.10;
    p.normal_radius = (argc > 5) ? std::atof(argv[5]) : 0.06;

    std::vector<double> nrm;
    da::estimate_normals(tgt, p.normal_radius, nullptr, nrm);
    da::IcpResult r = da::icp_point_to_plane(src, tgt, nrm, da::Sim3(), p);

    FILE* o = std::fopen(argv[3], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    double hdr[13] = { r.T.s, r.T.R[0],r.T.R[1],r.T.R[2],r.T.R[3],r.T.R[4],r.T.R[5],r.T.R[6],r.T.R[7],r.T.R[8],
                       r.T.t[0], r.T.t[1], r.T.t[2] };
    double rms[2] = { r.rms_before, r.rms_after };
    int32_t meta[2] = { r.rank, r.iters };
    std::fwrite(hdr, sizeof(double), 13, o);
    std::fwrite(rms, sizeof(double), 2, o);
    std::fwrite(meta, sizeof(int32_t), 2, o);
    std::fclose(o);
    return 0;
}
