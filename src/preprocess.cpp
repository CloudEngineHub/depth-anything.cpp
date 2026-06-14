#include "preprocess.hpp"
#include <cmath>
#include <algorithm>
namespace da {
static void resize_bilinear(const Image& s, int dw, int dh, std::vector<float>& dst_hwc){
    dst_hwc.assign((size_t)dw*dh*3, 0.f);
    const float sx = (float)s.w / dw, sy = (float)s.h / dh;
    for (int y=0;y<dh;++y){
        float fy = (y+0.5f)*sy - 0.5f; int y0 = (int)std::floor(fy); float wy = fy-y0;
        int y0c = std::clamp(y0,0,s.h-1), y1c = std::clamp(y0+1,0,s.h-1);
        for (int x=0;x<dw;++x){
            float fx = (x+0.5f)*sx - 0.5f; int x0 = (int)std::floor(fx); float wx = fx-x0;
            int x0c = std::clamp(x0,0,s.w-1), x1c = std::clamp(x0+1,0,s.w-1);
            for (int c=0;c<3;++c){
                auto P=[&](int yy,int xx){ return (float)s.rgb[((size_t)yy*s.w+xx)*3+c]; };
                float top = P(y0c,x0c)*(1-wx)+P(y0c,x1c)*wx;
                float bot = P(y1c,x0c)*(1-wx)+P(y1c,x1c)*wx;
                dst_hwc[((size_t)y*dw+x)*3+c] = top*(1-wy)+bot*wy;
            }
        }
    }
}
bool preprocess(const Image& img, const Config& cfg, Preprocessed& out){
    if (img.w<=0 || img.h<=0 || cfg.img_mean.size()<3 || cfg.img_std.size()<3) return false;
    const int patch = (int)cfg.patch_size;
    int dw = (img.w/patch)*patch, dh = (img.h/patch)*patch;
    if (dw==0) dw=patch; if (dh==0) dh=patch;
    std::vector<float> hwc; resize_bilinear(img, dw, dh, hwc);
    out.W = dw; out.H = dh; out.chw.assign((size_t)3*dh*dw, 0.f);
    for (int c=0;c<3;++c) for (int y=0;y<dh;++y) for (int x=0;x<dw;++x){
        float v = hwc[((size_t)y*dw+x)*3+c] / 255.f;
        out.chw[((size_t)c*dh+y)*dw+x] = (v - cfg.img_mean[c]) / cfg.img_std[c];
    }
    return true;
}
}
