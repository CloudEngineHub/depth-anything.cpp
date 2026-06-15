#include "cli.hpp"
#include "engine.hpp"
#include "depth_export.hpp"
#include "pose_export.hpp"
#include "ply_export.hpp"
#include "quantize.hpp"
#include <cstdio>
#include <algorithm>
#include <array>
#include <vector>
static int cmd_info(const std::string& model){
    auto eng = da::Engine::load(model, 1);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    const auto& c = eng->config();
    std::printf("checkpoint: %s\nembed_dim: %u\ndepth: %u\nnum_heads: %u\nstatus: loaded ok\n",
                c.checkpoint_name.c_str(), c.embed_dim, c.depth, c.num_heads);
    return 0;
}
static int cmd_depth_multi(const da::cli::Parsed& p, da::Engine& eng){
    // Load all input images.
    std::vector<da::Image> imgs(p.inputs.size());
    for (size_t i = 0; i < p.inputs.size(); ++i){
        if (!da::load_image_rgb(p.inputs[i], imgs[i])){
            std::fprintf(stderr, "error: load image failed: %s\n", p.inputs[i].c_str()); return 1;
        }
    }
    std::vector<da::ViewResult> views; int H, W;
    if (!eng.depth_pose_multi(imgs, views, H, W)){ std::fprintf(stderr, "error: depth_pose_multi failed\n"); return 1; }
    // Output prefix: --out-prefix, else --pfm, else --png, else "out".
    std::string prefix = !p.out_prefix.empty() ? p.out_prefix
                       : !p.output_pfm.empty() ? p.output_pfm
                       : !p.output_png.empty() ? p.output_png : std::string("out");
    for (size_t i = 0; i < views.size(); ++i){
        const auto& r = views[i];
        float dmin = r.depth[0], dmax = r.depth[0];
        for (float v : r.depth){ dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
        std::printf("view %zu: depth %dx%d min=%.4f max=%.4f fx=%.4f fy=%.4f\n",
                    i, W, H, dmin, dmax, r.intr[0], r.intr[4]);
        std::string base = prefix + "_view" + std::to_string(i);
        da::write_pfm(base + ".pfm", r.depth, H, W);
        da::write_depth_png(base + ".png", r.depth, H, W, p.invert);
        da::write_pose_json(base + ".json", r.ext, r.intr);
    }
    return 0;
}
static int cmd_depth_metric(const da::cli::Parsed& p){
    auto eng = da::Engine::load_nested(p.model, p.metric_model, 0);
    if (!eng){ std::fprintf(stderr, "error: load_nested failed\n"); return 1; }
    da::NestedOut out; int H, W;
    if (!eng->depth_metric_path(p.input, out, H, W)){ std::fprintf(stderr, "error: depth_metric failed\n"); return 1; }
    float dmin=out.depth[0], dmax=out.depth[0]; for(float v:out.depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("metric depth %dx%d min=%.4f max=%.4f scale_factor=%.6f\n", W, H, dmin, dmax, out.scale_factor);
    if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, out.depth, H, W);
    if(!p.output_png.empty()) da::write_depth_png(p.output_png, out.depth, H, W, p.invert);
    if(!p.output_pose.empty()){
        da::write_pose_json(p.output_pose, out.extrinsics, out.intrinsics);
    }
    return 0;
}
static int cmd_depth(const da::cli::Parsed& p){
    if (!p.metric_model.empty()) return cmd_depth_metric(p);
    auto eng = da::Engine::load(p.model, 0);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    if (p.inputs.size() > 1) return cmd_depth_multi(p, *eng);
    std::vector<float> depth, conf; int H,W;
    // Default: native-resolution real DA3 resize. --legacy-resize forces the old floor path.
    const bool native = !p.legacy_resize;
    if (!p.output_pose.empty()){
        std::array<float,12> ext; std::array<float,9> intr;
        bool ok = native ? eng->depth_pose_native_path(p.input, depth, conf, ext, intr, H, W)
                         : eng->depth_pose_path(p.input, depth, conf, ext, intr, H, W);
        if (!ok){ std::fprintf(stderr, "error: depth_pose failed\n"); return 1; }
        std::printf("pose: fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n", intr[0], intr[4], intr[2], intr[5]);
        if (!da::write_pose_json(p.output_pose, ext, intr)){ std::fprintf(stderr, "error: write pose json failed\n"); return 1; }
    } else {
        bool ok = native ? eng->depth_native(p.input, depth, conf, H, W)
                         : eng->depth(p.input, depth, conf, H, W);
        if (!ok){ std::fprintf(stderr, "error: depth failed\n"); return 1; }
    }
    float dmin=depth[0], dmax=depth[0]; for(float v:depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("depth %dx%d min=%.4f max=%.4f\n", W, H, dmin, dmax);
    if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
    if(!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
    return 0;
}
static int cmd_reconstruct(const da::cli::Parsed& p){
    auto eng = da::Engine::load(p.model, 0);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    da::Gaussians g; int H, W;
    if (!eng->reconstruct_path(p.input, g, H, W)){ std::fprintf(stderr, "error: reconstruct failed\n"); return 1; }
    std::printf("reconstructed %d gaussians (%dx%d)\n", g.N, W, H);
    if (!da::write_gaussian_ply(p.output_ply, g)){ std::fprintf(stderr, "error: write ply failed\n"); return 1; }
    std::printf("wrote %s\n", p.output_ply.c_str());
    return 0;
}
static int cmd_quantize(const da::cli::Parsed& p){
    if(!da::quantize_gguf(p.q_in, p.q_out, p.q_type)){ std::fprintf(stderr,"error: quantize failed\n"); return 1; }
    std::printf("wrote %s (%s)\n", p.q_out.c_str(), p.q_type.c_str());
    return 0;
}
int main(int argc, char** argv){
    auto p = da::cli::parse(argc, argv);
    if (!p.error.empty()){ std::fprintf(stderr, "error: %s\n", p.error.c_str()); da::cli::print_help(); return 1; }
    using S = da::cli::Sub;
    switch (p.sub){
        case S::Info: return cmd_info(p.model);
        case S::Depth: return cmd_depth(p);
        case S::Reconstruct: return cmd_reconstruct(p);
        case S::Quantize: return cmd_quantize(p);
        case S::Help: da::cli::print_help(); return 0;
        default: da::cli::print_help(); return 1;
    }
}
