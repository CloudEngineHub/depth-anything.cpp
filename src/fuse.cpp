#include "fuse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace da {

// Integer grid coordinate (fixed origin at world 0).
struct VKey {
    int32_t x, y, z;
    bool operator==(const VKey& o) const { return x==o.x && y==o.y && z==o.z; }
};
struct VKeyHash {
    size_t operator()(const VKey& k) const {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B185EBCA87ULL;
        h ^= (uint64_t)(uint32_t)k.y * 0xC2B2AE3D27D4EB4FULL + (h<<6) + (h>>2);
        h ^= (uint64_t)(uint32_t)k.z * 0x165667B19E3779F9ULL + (h<<6) + (h>>2);
        return (size_t)h;
    }
};

static inline VKey key_of(const float* p, double cell) {
    return { (int32_t)std::floor(p[0]/cell), (int32_t)std::floor(p[1]/cell), (int32_t)std::floor(p[2]/cell) };
}

// Bucket point indices into a hash grid at the given cell size.
static void build_grid(const std::vector<float>& xyz, int N, double cell,
                       std::unordered_map<VKey, std::vector<int>, VKeyHash>& grid) {
    grid.reserve((size_t)N);
    for (int i = 0; i < N; ++i) grid[key_of(&xyz[3*i], cell)].push_back(i);
}

int fuse_voxel(std::vector<float>& xyz, std::vector<uint8_t>& rgb,
               std::vector<float>& radius, const FuseParams& p,
               const std::vector<float>* weights) {
    const int N = (int)(xyz.size()/3);
    if (N <= 0 || (!(p.radius > 0.f) && !(p.voxel > 0.f))) return N;
    const bool have_rgb = ((int)rgb.size() == 3*N);
    const bool have_rad = ((int)radius.size() == N);
    const bool have_w   = (weights && (int)weights->size() == N);
    auto wof = [&](int i){ double w = have_w ? (double)(*weights)[i] : 1.0; return (w>0.0)?w:1e-6; };

    // ---- Stage 1: radius smoothing (normal-direction thinning) ----
    // Move each point to the weighted centroid of neighbours within `radius`.
    // Grid cell == radius, so all neighbours within radius lie in the 27-cell
    // block around a point (|Δindex| <= 1 per axis when cell == radius).
    std::vector<float> pos;                 // smoothed positions (3N)
    std::vector<uint8_t> col;               // smoothed colours (3N)
    if (p.radius > 0.f) {
        const double r = p.radius, r2 = r*r;
        std::unordered_map<VKey, std::vector<int>, VKeyHash> grid;
        build_grid(xyz, N, r, grid);
        pos.resize((size_t)3*N);
        if (have_rgb) col.resize((size_t)3*N);
        // Each point writes only its own pos[i]/col[i] from the const grid, so this
        // is embarrassingly parallel and bitwise-identical to the serial version.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const float* ci = &xyz[3*i];
            VKey k0 = key_of(ci, r);
            double wx=0,wy=0,wz=0,wsum=0,wr=0,wg=0,wb=0;
            for (int dx=-1; dx<=1; ++dx) for (int dy=-1; dy<=1; ++dy) for (int dz=-1; dz<=1; ++dz) {
                auto it = grid.find({k0.x+dx,k0.y+dy,k0.z+dz});
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    double ex=xyz[3*j+0]-ci[0], ey=xyz[3*j+1]-ci[1], ez=xyz[3*j+2]-ci[2];
                    if (ex*ex+ey*ey+ez*ez > r2) continue;
                    double w = wof(j);
                    wx += w*xyz[3*j+0]; wy += w*xyz[3*j+1]; wz += w*xyz[3*j+2]; wsum += w;
                    if (have_rgb) { wr += w*rgb[3*j+0]; wg += w*rgb[3*j+1]; wb += w*rgb[3*j+2]; }
                }
            }
            double iw = (wsum>0.0)?1.0/wsum:0.0;
            pos[3*i+0]=(float)(wx*iw); pos[3*i+1]=(float)(wy*iw); pos[3*i+2]=(float)(wz*iw);
            if (have_rgb) {
                auto c8=[](double v){ double c=std::max(0.0,std::min(255.0,v)); return (uint8_t)std::lround(c); };
                col[3*i+0]=c8(wr*iw); col[3*i+1]=c8(wg*iw); col[3*i+2]=c8(wb*iw);
            }
        }
    } else {
        pos = xyz;
        if (have_rgb) col = rgb;
    }

    // ---- Stage 2: voxel downsample (density) ----
    if (!(p.voxel > 0.f)) {                 // smoothing only: keep N points
        xyz.swap(pos);
        if (have_rgb) rgb.swap(col);
        return N;                           // radius[] unchanged (still per-point)
    }

    struct Accum { double wx=0,wy=0,wz=0,wr=0,wg=0,wb=0,rsum=0,wsum=0; int n=0; };
    std::unordered_map<VKey, Accum, VKeyHash> cells;
    cells.reserve((size_t)N);
    const double vox = p.voxel;
    for (int i = 0; i < N; ++i) {
        Accum& a = cells[key_of(&pos[3*i], vox)];
        double w = wof(i);
        a.wx += w*pos[3*i+0]; a.wy += w*pos[3*i+1]; a.wz += w*pos[3*i+2]; a.wsum += w;
        if (have_rgb) { a.wr += w*col[3*i+0]; a.wg += w*col[3*i+1]; a.wb += w*col[3*i+2]; }
        if (have_rad) a.rsum += radius[i];
        ++a.n;
    }
    std::vector<std::pair<VKey,Accum>> vc(cells.begin(), cells.end());
    std::sort(vc.begin(), vc.end(), [](const auto& a, const auto& b){
        if (a.first.x != b.first.x) return a.first.x < b.first.x;
        if (a.first.y != b.first.y) return a.first.y < b.first.y;
        return a.first.z < b.first.z;
    });
    const int M = (int)vc.size();
    std::vector<float> ox(3*M), orad; std::vector<uint8_t> orgb;
    if (have_rgb) orgb.resize(3*M);
    if (have_rad) orad.resize(M);
    for (int i = 0; i < M; ++i) {
        const Accum& a = vc[i].second;
        double iw = (a.wsum>0.0)?1.0/a.wsum:0.0;
        ox[3*i+0]=(float)(a.wx*iw); ox[3*i+1]=(float)(a.wy*iw); ox[3*i+2]=(float)(a.wz*iw);
        if (have_rgb) {
            auto c8=[](double v){ double c=std::max(0.0,std::min(255.0,v)); return (uint8_t)std::lround(c); };
            orgb[3*i+0]=c8(a.wr*iw); orgb[3*i+1]=c8(a.wg*iw); orgb[3*i+2]=c8(a.wb*iw);
        }
        if (have_rad) orad[i]=(float)(a.rsum/std::max(1,a.n));
    }
    xyz.swap(ox);
    if (have_rgb) rgb.swap(orgb);
    if (have_rad) radius.swap(orad);
    return M;
}

} // namespace da
