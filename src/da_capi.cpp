#include "da_capi.h"
#include "engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct da_ctx { std::unique_ptr<da::Engine> engine; std::string last_error; };

static char* dup_cstr(const std::string& s){
    char* p = (char*)std::malloc(s.size()+1);
    if (p) std::memcpy(p, s.c_str(), s.size()+1);
    return p;
}
// Minimal JSON string escaping for interpolated values (quotes, backslash, controls).
static std::string json_escape(const std::string& s){
    std::string o; o.reserve(s.size()+2);
    for (char ch : s){
        switch (ch){
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)ch < 0x20){ char b[8]; std::snprintf(b,sizeof(b),"\\u%04x",ch); o += b; }
                else o += ch;
        }
    }
    return o;
}
extern "C" {
int da_capi_abi_version(void){ return 1; }
da_ctx* da_capi_load(const char* path, int n_threads){
    if (!path) return nullptr;
    auto e = da::Engine::load(path, n_threads);
    if (!e) return nullptr;
    auto* c = new da_ctx(); c->engine = std::move(e); return c;
}
void da_capi_free(da_ctx* c){ delete c; }
char* da_capi_info_json(da_ctx* c){
    if (!c || !c->engine) return nullptr;
    const auto& cfg = c->engine->config();
    std::string j = "{\"checkpoint\":\"" + json_escape(cfg.checkpoint_name) + "\",\"embed_dim\":" +
        std::to_string(cfg.embed_dim) + ",\"depth\":" + std::to_string(cfg.depth) +
        ",\"num_heads\":" + std::to_string(cfg.num_heads) + "}";
    return dup_cstr(j);
}
void da_capi_free_string(char* s){ std::free(s); }
const char* da_capi_last_error(da_ctx* c){ return c ? c->last_error.c_str() : ""; }
float* da_capi_depth_path(da_ctx* c, const char* image_path, int* out_h, int* out_w){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "depth: bad args"; return nullptr; }
    std::vector<float> depth, conf; int H = 0, W = 0;
    if (!c->engine->depth(image_path, depth, conf, H, W)){ c->last_error = "depth: failed"; return nullptr; }
    float* p = (float*)std::malloc(depth.size() * sizeof(float));
    if (!p){ c->last_error = "depth: oom"; return nullptr; }
    std::memcpy(p, depth.data(), depth.size() * sizeof(float));
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    return p;
}
void da_capi_free_floats(float* p){ std::free(p); }
}
