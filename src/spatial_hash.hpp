#pragma once
// Uniform spatial hash over a point cloud for nearest-neighbour / radius queries.
// Cell size == the widest query radius run against it, so a 27-cell (3x3x3) scan
// is exact for any query radius <= cell. Used by point-to-plane ICP (src/icp.cpp)
// and loop-closure correspondence (src/stream.cpp). Points are borrowed (3N
// interleaved doubles), not copied — keep them alive for the hash's lifetime.
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace da {

struct SpatialHash {
    double cell = 1.0;
    const double* pts = nullptr;
    int n = 0;
    std::unordered_map<uint64_t, std::vector<int>> map;

    SpatialHash(const double* p, int count, double cell_size)
        : cell(cell_size > 0 ? cell_size : 1.0), pts(p), n(count) {
        map.reserve((size_t)count * 2 + 16);
        for (int i = 0; i < count; ++i) map[key_of(p + 3*i)].push_back(i);
    }
    static uint64_t pack(int a, int b, int c) {
        const uint64_t M = (1u << 21) - 1;
        return ((uint64_t)(a & M)) | ((uint64_t)(b & M) << 21) | ((uint64_t)(c & M) << 42);
    }
    uint64_t key_of(const double q[3]) const {
        return pack((int)std::floor(q[0]/cell), (int)std::floor(q[1]/cell), (int)std::floor(q[2]/cell));
    }
    // Nearest point to q within max_dist. Returns index (-1 if none); fills d2_out.
    int nearest(const double q[3], double max_dist, double& d2_out) const {
        const int cx = (int)std::floor(q[0]/cell), cy = (int)std::floor(q[1]/cell), cz = (int)std::floor(q[2]/cell);
        const double md2 = max_dist * max_dist;
        int best = -1; double best2 = md2;
        for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            auto it = map.find(pack(cx+dx, cy+dy, cz+dz));
            if (it == map.end()) continue;
            for (int j : it->second) {
                double ex=q[0]-pts[3*j+0], ey=q[1]-pts[3*j+1], ez=q[2]-pts[3*j+2];
                double e2 = ex*ex+ey*ey+ez*ez;
                if (e2 < best2) { best2 = e2; best = j; }
            }
        }
        if (best >= 0) d2_out = best2;
        return best;
    }
    // All points within r of q (r <= cell) appended to out (as indices).
    void radius(const double q[3], double r, std::vector<int>& out) const {
        const int cx = (int)std::floor(q[0]/cell), cy = (int)std::floor(q[1]/cell), cz = (int)std::floor(q[2]/cell);
        const double r2 = r*r;
        for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            auto it = map.find(pack(cx+dx, cy+dy, cz+dz));
            if (it == map.end()) continue;
            for (int j : it->second) {
                double ex=q[0]-pts[3*j+0], ey=q[1]-pts[3*j+1], ez=q[2]-pts[3*j+2];
                if (ex*ex+ey*ey+ez*ez <= r2) out.push_back(j);
            }
        }
    }
};

} // namespace da
