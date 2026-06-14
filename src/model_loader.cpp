#include "model_loader.hpp"
#include "da_gguf_keys.h"
#include "common.hpp"

namespace da {
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_u32(g,id);
}
static int32_t kv_i32(gguf_context* g, const char* k, int32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::vector<float> kv_f32_arr(gguf_context* g, const char* k){
    std::vector<float> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_FLOAT32){
        size_t n = gguf_get_arr_n(g,id);
        const float* a = (const float*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}

ModelLoader::~ModelLoader(){
    if (gguf_) gguf_free(gguf_);
    if (ctx_)  ggml_free(ctx_);
    if (device_ctx_) ggml_free(device_ctx_);
}

bool ModelLoader::load(const std::string& path){
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_){ DA_LOG("gguf_init_from_file failed: %s", path.c_str()); return false; }
    cfg_.patch_size      = kv_u32(gguf_, DA_KV_PATCH_SIZE, 14);
    cfg_.embed_dim       = kv_u32(gguf_, DA_KV_VIT_EMBED_DIM);
    cfg_.depth           = kv_u32(gguf_, DA_KV_VIT_DEPTH);
    cfg_.num_heads       = kv_u32(gguf_, DA_KV_VIT_NUM_HEADS);
    cfg_.head_dim        = kv_u32(gguf_, DA_KV_VIT_HEAD_DIM);
    cfg_.mlp_hidden      = kv_u32(gguf_, DA_KV_VIT_MLP_HIDDEN);
    cfg_.num_register    = kv_u32(gguf_, DA_KV_VIT_NUM_REGISTER);
    cfg_.pos_embed_grid  = kv_u32(gguf_, DA_KV_VIT_POS_EMBED_GRID);
    cfg_.alt_start       = kv_i32(gguf_, DA_KV_VIT_ALT_START, -1);
    cfg_.rope_start      = kv_i32(gguf_, DA_KV_VIT_ROPE_START, -1);
    cfg_.qknorm_start    = kv_i32(gguf_, DA_KV_VIT_QKNORM_START, -1);
    cfg_.init_values     = kv_f32(gguf_, DA_KV_VIT_INIT_VALUES, 0.f);
    cfg_.rope_freq       = kv_f32(gguf_, DA_KV_VIT_ROPE_FREQ, 100.f);
    cfg_.ln_eps          = kv_f32(gguf_, DA_KV_VIT_LN_EPS, 1e-6f);
    cfg_.interp_offset   = kv_f32(gguf_, DA_KV_VIT_INTERP_OFFSET, 0.1f);
    cfg_.cat_token       = kv_bool(gguf_, DA_KV_VIT_CAT_TOKEN, true);
    cfg_.qkv_bias        = kv_bool(gguf_, DA_KV_VIT_QKV_BIAS, true);
    cfg_.interp_antialias= kv_bool(gguf_, DA_KV_VIT_INTERP_ANTIALIAS, false);
    cfg_.out_layers      = kv_i32_arr(gguf_, DA_KV_VIT_OUT_LAYERS);
    cfg_.img_mean        = kv_f32_arr(gguf_, DA_KV_IMG_MEAN);
    cfg_.img_std         = kv_f32_arr(gguf_, DA_KV_IMG_STD);

    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (t) tensors_[nm] = t;
    }
    return cfg_.embed_dim>0 && cfg_.depth>0;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it==tensors_.end() ? nullptr : it->second;
}

bool ModelLoader::offload_weights(Backend&){ return true; }  // CPU zero-copy for M0/M1; GPU mirror added in M7
} // namespace da
