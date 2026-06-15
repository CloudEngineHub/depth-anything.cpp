#include "depth_export.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace da {

bool write_pfm(const std::string& path, const std::vector<float>& depth, int H, int W){
    if (H <= 0 || W <= 0 || depth.size() != (size_t)H * (size_t)W) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    // "Pf" = single-channel float. scale < 0 means little-endian samples.
    std::fprintf(f, "Pf\n%d %d\n-1.0\n", W, H);
    // PFM stores rows bottom-to-top by spec: our depth is row-major top-to-bottom
    // (row 0 = top), so emit rows in reverse (row H-1 first) for upright viewers.
    bool ok = true;
    for (int y = H - 1; y >= 0 && ok; --y){
        size_t n = std::fwrite(depth.data() + (size_t)y * W, sizeof(float), (size_t)W, f);
        ok = (n == (size_t)W);
    }
    std::fclose(f);
    return ok;
}

bool write_depth_png(const std::string& path, const std::vector<float>& depth, int H, int W, bool invert){
    if (H <= 0 || W <= 0 || depth.size() != (size_t)H * (size_t)W) return false;
    float dmin = depth[0], dmax = depth[0];
    for (float v : depth){ dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
    const float range = dmax - dmin;
    std::vector<uint8_t> px((size_t)H * W);
    for (size_t i = 0; i < px.size(); ++i){
        float t = (range > 0.0f) ? (depth[i] - dmin) / range : 0.0f;
        if (invert) t = 1.0f - t;
        int v = (int)(t * 255.0f + 0.5f);
        px[i] = (uint8_t)std::min(255, std::max(0, v));
    }
    return stbi_write_png(path.c_str(), W, H, 1, px.data(), W) != 0;
}

} // namespace da
