// Standalone harness for scripts/oracle_fuse.py (NOT a ctest): reads a binary
// point cloud, runs src/fuse.cpp, writes the result, so the Python oracle can
// diff our fusion against open3d (voxel stage) and scipy (radius stage).
//
// Binary format (little-endian): int32 N, then 3N float32 xyz. Same for output.
//   fuse_dump <in> <out> <voxel> [radius]
#include "fuse.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: fuse_dump <in> <out> <voxel> [radius]\n"); return 2; }
    const double voxel  = std::atof(argv[3]);
    const double radius = (argc > 4) ? std::atof(argv[4]) : 0.0;

    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int32_t N = 0;
    if (std::fread(&N, sizeof(int32_t), 1, f) != 1 || N < 0) { std::fclose(f); return 1; }
    std::vector<float> xyz((size_t)3*N);
    if (N > 0 && std::fread(xyz.data(), sizeof(float), (size_t)3*N, f) != (size_t)3*N) { std::fclose(f); return 1; }
    std::fclose(f);

    std::vector<uint8_t> rgb; std::vector<float> rad;
    da::FuseParams p; p.voxel = (float)voxel; p.radius = (float)radius;
    int M = da::fuse_voxel(xyz, rgb, rad, p);

    FILE* o = std::fopen(argv[2], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    int32_t m = M;
    std::fwrite(&m, sizeof(int32_t), 1, o);
    std::fwrite(xyz.data(), sizeof(float), (size_t)3*M, o);
    std::fclose(o);
    return 0;
}
