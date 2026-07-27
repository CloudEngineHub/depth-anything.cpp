#pragma once
// Global voxel-hash surface fusion (task A). The streaming stitcher deposits one
// point per pixel per emitted frame, so overlapping observations of a surface
// pile up as several near-coincident (and, under residual misalignment, doubled)
// sheets — which read as ghosting. fuse_voxel() collapses all points sharing a
// voxel into ONE point (weighted-centroid position, mean color, representative
// radius), turning those stacked sheets into a single surface and shrinking the
// point count. This is the classic voxel-downsample / TSDF-lite operation and is
// cross-checked against open3d's voxel_down_sample.
#include <cstdint>
#include <vector>

namespace da {

struct FuseParams {
    // Surface-smoothing / merge radius (world units): each point is moved to the
    // (weighted) centroid of its neighbours within this radius. Because it works
    // in a metric radius, not a grid cell, it thins sheets and collapses doubles
    // regardless of voxel-boundary alignment — the normal-direction thinning that
    // naive voxel downsampling can't guarantee. <= 0 disables smoothing.
    float radius = 0.03f;
    // Output downsample cell (world units): after smoothing, keep one point per
    // occupied voxel (centroid). Reduces density/file size. <= 0 keeps every
    // (smoothed) point. With radius<=0 this is a pure voxel_down_sample.
    float voxel = 0.03f;
};

// In-place surface fusion of a parallel point cloud (xyz: 3N, rgb: 3N, radius: N).
// Two stages: (1) radius smoothing — move each point to its neighbourhood
// centroid (thins/denoises sheets, collapses doubles within `radius`); (2) voxel
// downsample — one centroid point per occupied voxel (density reduction). If
// `weights` is non-null (length N, e.g. per-point confidence) it weights both
// stages; otherwise uniform. Output sorted by voxel key for determinism. Returns
// the output point count. The pure voxel stage (radius<=0) is idempotent.
int fuse_voxel(std::vector<float>& xyz, std::vector<uint8_t>& rgb,
               std::vector<float>& radius, const FuseParams& p,
               const std::vector<float>* weights = nullptr);

} // namespace da
