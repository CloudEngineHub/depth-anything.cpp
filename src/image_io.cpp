#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image_io.hpp"
namespace da {
static bool from_stb(unsigned char* data, int w, int h, Image& out){
    if (!data) return false;
    out.w = w; out.h = h; out.rgb.assign(data, data + (size_t)w*h*3);
    stbi_image_free(data); return true;
}
bool load_image_rgb(const std::string& path, Image& out){
    int w,h,c; unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 3);
    return from_stb(d, w, h, out);
}
bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out){
    int w,h,c; unsigned char* d = stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3);
    return from_stb(d, w, h, out);
}
}
