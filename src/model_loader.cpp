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
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}

ModelLoader::~ModelLoader(){
    if (gguf_) gguf_free(gguf_);
    if (ctx_)  ggml_free(ctx_);
    if (device_ctx_) ggml_free(device_ctx_);
}

bool ModelLoader::load(const std::string& path){
    // Guard against re-entry: a second load() would otherwise leak the prior
    // gguf/ctx and accumulate stale tensor-map entries.
    if (gguf_) { gguf_free(gguf_); gguf_ = nullptr; }
    if (ctx_)  { ggml_free(ctx_);  ctx_  = nullptr; }
    tensors_.clear();
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_){ DA_LOG("gguf_init_from_file failed: %s", path.c_str()); return false; }
    // Nested metric branch: config + tensors live under m_vit.*/m_head.* prefixes.
    // Detect it and read the m_* KV; tensors are aliased to vit.*/head.* below so the
    // existing backbone/DPT graph code (which hard-codes vit./head.) works unchanged.
    const bool metric = gguf_find_key(gguf_, DA_KV_M_VIT_EMBED_DIM) >= 0;
    cfg_.patch_size      = kv_u32(gguf_, DA_KV_PATCH_SIZE, 14);
    if (metric){
        cfg_.embed_dim       = kv_u32(gguf_, DA_KV_M_VIT_EMBED_DIM);
        cfg_.depth           = kv_u32(gguf_, DA_KV_M_VIT_DEPTH);
        cfg_.num_heads       = kv_u32(gguf_, DA_KV_M_VIT_NUM_HEADS);
        cfg_.head_dim        = kv_u32(gguf_, DA_KV_M_VIT_HEAD_DIM);
        cfg_.mlp_hidden      = kv_u32(gguf_, DA_KV_M_VIT_MLP_HIDDEN);
        cfg_.num_register    = kv_u32(gguf_, DA_KV_M_VIT_NUM_REGISTER);
        cfg_.pos_embed_grid  = kv_u32(gguf_, DA_KV_M_VIT_POS_EMBED_GRID);
        cfg_.alt_start       = kv_i32(gguf_, DA_KV_M_VIT_ALT_START, -1);
        cfg_.rope_start      = kv_i32(gguf_, DA_KV_M_VIT_ROPE_START, -1);
        cfg_.qknorm_start    = kv_i32(gguf_, DA_KV_M_VIT_QKNORM_START, -1);
        cfg_.init_values     = kv_f32(gguf_, DA_KV_M_VIT_INIT_VALUES, 0.f);
        cfg_.rope_freq       = kv_f32(gguf_, DA_KV_M_VIT_ROPE_FREQ, 100.f);
        cfg_.ln_eps          = kv_f32(gguf_, DA_KV_M_VIT_LN_EPS, 1e-6f);
        cfg_.interp_offset   = kv_f32(gguf_, DA_KV_M_VIT_INTERP_OFFSET, 0.1f);
        cfg_.cat_token       = kv_bool(gguf_, DA_KV_M_VIT_CAT_TOKEN, true);
        cfg_.qkv_bias        = kv_bool(gguf_, DA_KV_M_VIT_QKV_BIAS, true);
        cfg_.interp_antialias= kv_bool(gguf_, DA_KV_M_VIT_INTERP_ANTIALIAS, false);
        cfg_.ffn_type        = kv_str(gguf_, DA_KV_M_VIT_FFN_TYPE, "mlp");
        cfg_.head_features   = kv_u32(gguf_, DA_KV_M_HEAD_FEATURES, 0);
        cfg_.out_layers      = kv_i32_arr(gguf_, DA_KV_M_VIT_OUT_LAYERS);
        cfg_.head_out_channels = kv_i32_arr(gguf_, DA_KV_M_HEAD_OUT_CHANNELS);
        // Metric DPT pos_embed defaults to False (DPT.__init__ default); the nested
        // converter omits the KV, so it is absent here -> no UV pos-embed in the head.
        cfg_.head_pos_embed  = kv_bool(gguf_, "depthanything3.m_head.pos_embed", false);
    } else {
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
        cfg_.ffn_type        = kv_str(gguf_, DA_KV_VIT_FFN_TYPE, "mlp");
        cfg_.head_features   = kv_u32(gguf_, DA_KV_HEAD_FEATURES, 0);
        cfg_.out_layers      = kv_i32_arr(gguf_, DA_KV_VIT_OUT_LAYERS);
        cfg_.head_out_channels = kv_i32_arr(gguf_, DA_KV_HEAD_OUT_CHANNELS);
        cfg_.head_pos_embed  = kv_bool(gguf_, DA_KV_HEAD_POS_EMBED, true);
    }
    cfg_.img_mean        = kv_f32_arr(gguf_, DA_KV_IMG_MEAN);
    cfg_.img_std         = kv_f32_arr(gguf_, DA_KV_IMG_STD);
    cfg_.img_resize_target = kv_u32(gguf_, DA_KV_IMG_RESIZE_TARGET, 504);
    cfg_.img_resize_mode = kv_str(gguf_, DA_KV_IMG_RESIZE_MODE, "upper_bound");
    cfg_.checkpoint_name = kv_str(gguf_, DA_KV_CHECKPOINT_NAME);

    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (!t) continue;
        tensors_[nm] = t;
        // Alias metric-branch tensors so vit./head. lookups resolve to m_vit./m_head.
        std::string s(nm);
        if (s.rfind("m_vit.", 0) == 0)  tensors_["vit."  + s.substr(6)] = t;
        else if (s.rfind("m_head.", 0) == 0) tensors_["head." + s.substr(7)] = t;
    }
    // All four structural dims are converter-written and required by the graph
    // (head_dim/num_heads feed attention reshapes). Gate on all of them so a
    // malformed external GGUF fails here rather than dividing by zero later.
    return cfg_.embed_dim>0 && cfg_.depth>0 && cfg_.num_heads>0 && cfg_.head_dim>0;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it==tensors_.end() ? nullptr : it->second;
}

bool ModelLoader::offload_weights(Backend&){ return true; }  // CPU zero-copy for M0/M1; GPU mirror added in M7
} // namespace da
