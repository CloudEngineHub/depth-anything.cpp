#include "cli.hpp"
#include "engine.hpp"
#include "depth_export.hpp"
#include "pose_export.hpp"
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
static int cmd_depth(const da::cli::Parsed& p){
    auto eng = da::Engine::load(p.model, 0);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    std::vector<float> depth, conf; int H,W;
    if (!p.output_pose.empty()){
        std::array<float,12> ext; std::array<float,9> intr;
        if (!eng->depth_pose_path(p.input, depth, conf, ext, intr, H, W)){ std::fprintf(stderr, "error: depth_pose failed\n"); return 1; }
        std::printf("pose: fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n", intr[0], intr[4], intr[2], intr[5]);
        if (!da::write_pose_json(p.output_pose, ext, intr)){ std::fprintf(stderr, "error: write pose json failed\n"); return 1; }
    } else {
        if (!eng->depth(p.input, depth, conf, H, W)){ std::fprintf(stderr, "error: depth failed\n"); return 1; }
    }
    float dmin=depth[0], dmax=depth[0]; for(float v:depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("depth %dx%d min=%.4f max=%.4f\n", W, H, dmin, dmax);
    if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
    if(!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
    return 0;
}
int main(int argc, char** argv){
    auto p = da::cli::parse(argc, argv);
    if (!p.error.empty()){ std::fprintf(stderr, "error: %s\n", p.error.c_str()); da::cli::print_help(); return 1; }
    using S = da::cli::Sub;
    switch (p.sub){
        case S::Info: return cmd_info(p.model);
        case S::Depth: return cmd_depth(p);
        case S::Help: da::cli::print_help(); return 0;
        default: da::cli::print_help(); return 1;
    }
}
