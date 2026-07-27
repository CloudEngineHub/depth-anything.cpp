#pragma once
#include <vector>
#include <array>

namespace da {

// World-space 3D Gaussians produced by the GaussianAdapter (host geometry).
// All arrays are flattened row-major; N = H*W (single view, B=V=1).
struct Gaussians {
    int N = 0;
    std::vector<float> means;      // [N*3]        (pixel*3 + xyz)
    std::vector<float> scales;     // [N*3]        (pixel*3 + xyz)
    std::vector<float> rotations;  // [N*4]   wxyz (pixel*4 + c)
    std::vector<float> harmonics;  // [N*3*9]      (pixel*3 + color)*9 + coeff
    std::vector<float> opacities;  // [N]
    // Optional per-gaussian input-photo RGB in [0,1] (pixel*3 + color), filled by
    // Engine::reconstruct from the processed image. DA3's learned SH-DC colour is a
    // deliberately flat, near-grey base (higher SH bands masked x0.025), so for the
    // single-image input-view presentation we colour each gaussian by its own source
    // pixel -- same convention as the point cloud. Empty => fall back to SH-DC.
    std::vector<float> colors;     // [N*3]
    // Input camera (from the model's pose head), so the demo can render the
    // gaussians from the exact viewpoint they composite into.
    std::array<float,12> ext{};    // world->camera 3x4 (w2c), row-major
    std::array<float,9>  intr{};   // K 3x3 (pixel units at processed W,H), row-major
    int H = 0, W = 0;              // processed resolution the gaussians live at
};

// GaussianAdapter: pure host math (NO learned weights). Converts the GSDPT
// raw gaussian channels (37) + confidence + the giant depth + camera pose into
// world-space 3D Gaussians, exactly mirroring depth_anything_3.model.gs_adapter
// .GaussianAdapter.forward (sh_degree=2, pred_offset_depth/xy, no pred_color).
class GsAdapter {
public:
    int   sh_degree         = 2;
    int   d_sh              = 9;     // (sh_degree+1)^2
    bool  pred_offset_depth = true;
    bool  pred_offset_xy    = true;
    bool  pred_color        = false;
    float scale_min         = 1e-5f;
    float scale_max         = 30.0f;

    // raw_gs  : [H*W*37], channels-LAST: (h*W+w)*37 + c.
    // depth   : [H*W], row-major (h*W+w).
    // gs_conf : [H*W], row-major; mapped to opacity via map_pdf_to_opacity.
    // ext     : 3x4 row-major, the world->camera extrinsics (w2c).
    // intr    : 3x3 row-major camera intrinsics K (pixel units, H/W image).
    // Returns false on malformed input (size / singular intrinsics).
    bool build(const std::vector<float>& raw_gs, const std::vector<float>& depth,
               const std::vector<float>& gs_conf,
               const std::array<float,12>& ext, const std::array<float,9>& intr,
               int H, int W, Gaussians& out) const;
};

} // namespace da
