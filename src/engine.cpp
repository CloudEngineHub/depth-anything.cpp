#include "engine.hpp"
#include "common.hpp"
#include "dino_backbone.hpp"
#include "image_io.hpp"
#include "preprocess.hpp"
#include "dpt_head.hpp"
#include "cam_pose.hpp"

namespace da {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads){
    std::unique_ptr<Engine> e(new Engine());
    if (!e->ml_.load(path)) { DA_LOG("engine: load failed"); return nullptr; }
    e->be_.set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) { DA_LOG("engine: offload failed"); return nullptr; }
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
bool Engine::depth_pose_path(const std::string& image_path, std::vector<float>& depth, std::vector<float>& conf,
                             std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose: load image failed"); return false; }
    return depth_pose(img, depth, conf, ext, intr, H, W);
}
} // namespace da
