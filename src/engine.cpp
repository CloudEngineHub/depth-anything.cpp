#include "engine.hpp"
#include "common.hpp"
#include "dino_backbone.hpp"
#include "image_io.hpp"
#include "preprocess.hpp"
#include "dpt_head.hpp"
#include "cam_pose.hpp"
#include "gs_head.hpp"
#include "gs_adapter.hpp"
#include "nested.hpp"

namespace da {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads){
    std::unique_ptr<Engine> e(new Engine());
    if (!e->ml_.load(path)) { DA_LOG("engine: load failed"); return nullptr; }
    e->be_.set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) { DA_LOG("engine: offload failed"); return nullptr; }
    return e;
}
std::unique_ptr<Engine> Engine::load_nested(const std::string& anyview_gguf,
                                            const std::string& metric_gguf, int n_threads){
    auto e = load(anyview_gguf, n_threads);
    if (!e) { DA_LOG("engine: anyview load failed"); return nullptr; }
    e->metric_ml_.reset(new ModelLoader());
    e->metric_be_.reset(new Backend());
    if (!e->metric_ml_->load(metric_gguf)) { DA_LOG("engine: metric load failed"); return nullptr; }
    e->metric_be_->set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->metric_ml_->offload_weights(*e->metric_be_)) { DA_LOG("engine: metric offload failed"); return nullptr; }
    return e;
}
bool Engine::backbone_features(const std::vector<float>& input_chw, int H, int W,
                               std::vector<std::vector<float>>& feats_out){
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> cams;
    return bb.forward(input_chw, H, W, feats_out, cams);
}
bool Engine::depth(const std::string& image_path, std::vector<float>& depth_out,
                   std::vector<float>& conf_out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth: load image failed"); return false; }
    return depth_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_image(const Image& img, std::vector<float>& depth_out,
                         std::vector<float>& conf_out, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth: preprocess failed"); return false; }
    H = p.H; W = p.W;
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) { DA_LOG("depth: backbone failed"); return false; }
    DptHead head(ml_, be_);
    return head.depth(feats, H, W, depth_out, conf_out);
}
bool Engine::depth_native(const std::string& image_path, std::vector<float>& depth_out,
                          std::vector<float>& conf_out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_native: load image failed"); return false; }
    return depth_native_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_native_image(const Image& img, std::vector<float>& depth_out,
                                std::vector<float>& conf_out, int& H, int& W){
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_native: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) { DA_LOG("depth_native: backbone failed"); return false; }
    DptHead head(ml_, be_);
    return head.depth(feats, H, W, depth_out, conf_out);
}
bool Engine::depth_pose_native(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                               std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_pose_native: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose_native: backbone failed"); return false; }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("depth_pose_native: depth head failed"); return false; }
    if (cam_tokens.size() < 4) { DA_LOG("depth_pose_native: missing layer-11 cam token"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("depth_pose_native: cam pose failed"); return false; }
    return true;
}
bool Engine::depth_pose_native_path(const std::string& image_path, std::vector<float>& depth,
                                    std::vector<float>& conf, std::array<float,12>& ext,
                                    std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose_native: load image failed"); return false; }
    return depth_pose_native(img, depth, conf, ext, intr, H, W);
}
bool Engine::depth_pose(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                        std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth_pose: preprocess failed"); return false; }
    H = p.H; W = p.W;
    // Run the backbone ONCE; reuse feats for depth and cam_tokens[3] for pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose: backbone failed"); return false; }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("depth_pose: depth head failed"); return false; }
    if (cam_tokens.size() < 4) { DA_LOG("depth_pose: missing layer-11 cam token"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("depth_pose: cam pose failed"); return false; }
    return true;
}
bool Engine::depth_pose_multi(const std::vector<Image>& imgs, std::vector<ViewResult>& out, int& H, int& W){
    out.clear();
    if (imgs.empty()) { DA_LOG("depth_pose_multi: no images"); return false; }
    // Preprocess every image; all must yield identical H,W.
    std::vector<std::vector<float>> views_chw;
    views_chw.reserve(imgs.size());
    H = 0; W = 0;
    for (size_t i = 0; i < imgs.size(); ++i){
        Preprocessed p;
        if (!preprocess(imgs[i], ml_.config(), p)) { DA_LOG("depth_pose_multi: preprocess failed"); return false; }
        if (i == 0) { H = p.H; W = p.W; }
        else if (p.H != H || p.W != W) { DA_LOG("depth_pose_multi: views differ in H,W"); return false; }
        views_chw.push_back(std::move(p.chw));
    }
    const int S = (int)views_chw.size();
    // One backbone pass over all views (cross-view global attention).
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<std::vector<float>>> feats, cam_tokens;  // [L=4][S][...]
    if (!bb.forward_mv(views_chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose_multi: backbone failed"); return false; }
    if (feats.size() < 4 || cam_tokens.size() < 4) { DA_LOG("depth_pose_multi: missing out layers"); return false; }
    out.resize(S);
    for (int v = 0; v < S; ++v){
        ViewResult& r = out[v];
        std::vector<std::vector<float>> feats4_v = {
            feats[0][v], feats[1][v], feats[2][v], feats[3][v] };
        DptHead head(ml_, be_);
        if (!head.depth(feats4_v, H, W, r.depth, r.conf)) { DA_LOG("depth_pose_multi: depth head failed"); return false; }
        CamPose cam(ml_, be_);
        std::array<float,9> pe;
        if (!cam.pose(cam_tokens[3][v], H, W, pe, r.ext, r.intr)) { DA_LOG("depth_pose_multi: cam pose failed"); return false; }
    }
    return true;
}
bool Engine::reconstruct(const Image& img, Gaussians& g, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("reconstruct: preprocess failed"); return false; }
    H = p.H; W = p.W;
    // One backbone pass: feats[4] feed depth + gs_head; cam_tokens[3] feeds pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("reconstruct: backbone failed"); return false; }
    if (feats.size() < 4 || cam_tokens.size() < 4) { DA_LOG("reconstruct: missing out layers"); return false; }
    DptHead head(ml_, be_);
    std::vector<float> depth, conf;
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("reconstruct: depth head failed"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe; std::array<float,12> ext; std::array<float,9> intr;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("reconstruct: cam pose failed"); return false; }
    GsHead gs(ml_, be_);
    std::vector<float> raw_gs, gs_conf;
    if (!gs.raw_gaussians(feats, p.chw, H, W, raw_gs, gs_conf)) { DA_LOG("reconstruct: gs_head failed"); return false; }
    GsAdapter ad;
    if (!ad.build(raw_gs, depth, gs_conf, ext, intr, H, W, g)) { DA_LOG("reconstruct: gs_adapter failed"); return false; }
    return true;
}
bool Engine::reconstruct_path(const std::string& image_path, Gaussians& g, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("reconstruct: load image failed"); return false; }
    return reconstruct(img, g, H, W);
}
bool Engine::depth_pose_path(const std::string& image_path, std::vector<float>& depth, std::vector<float>& conf,
                             std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose: load image failed"); return false; }
    return depth_pose(img, depth, conf, ext, intr, H, W);
}
bool Engine::depth_metric(const Image& img, NestedOut& out, int& H, int& W){
    if (!metric_ml_ || !metric_be_) { DA_LOG("depth_metric: engine not loaded via load_nested"); return false; }
    // Both branches consume the SAME preprocessed input x (da3.py NestedDepthAnything3Net).
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth_metric: preprocess failed"); return false; }
    H = p.H; W = p.W;

    // --- anyview (GIANT): backbone once -> depth + conf + cam pose ---
    AnyviewOut any;
    {
        DinoBackbone bb(ml_, be_);
        std::vector<std::vector<float>> feats, cam_tokens;
        if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_metric: anyview backbone failed"); return false; }
        if (cam_tokens.size() < 4) { DA_LOG("depth_metric: missing cam token"); return false; }
        DptHead head(ml_, be_);
        if (!head.depth(feats, H, W, any.depth, any.depth_conf)) { DA_LOG("depth_metric: anyview depth head failed"); return false; }
        CamPose cam(ml_, be_);
        std::array<float,9> pe;
        if (!cam.pose(cam_tokens[3], H, W, pe, any.extrinsics, any.intrinsics)) { DA_LOG("depth_metric: cam pose failed"); return false; }
    }

    // --- metric (ViT-L + DPT/sky): backbone + depth_sky head ---
    MetricOut metric;
    {
        DinoBackbone bb(*metric_ml_, *metric_be_);
        std::vector<std::vector<float>> feats_m, cams_m;
        if (!bb.forward(p.chw, H, W, feats_m, cams_m)) { DA_LOG("depth_metric: metric backbone failed"); return false; }
        DptHead head(*metric_ml_, *metric_be_);
        if (!head.depth_sky(feats_m, H, W, metric.depth, metric.sky)) { DA_LOG("depth_metric: metric depth_sky failed"); return false; }
    }
    // The metric branch applies its own sky-fill inside da3_metric(x) before alignment.
    NestedAligner::process_mono_sky(metric.depth, metric.sky);

    out = NestedAligner::align(any, metric, H, W);
    return true;
}
bool Engine::depth_metric_path(const std::string& image_path, NestedOut& out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_metric: load image failed"); return false; }
    return depth_metric(img, out, H, W);
}
} // namespace da
