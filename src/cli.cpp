#include "cli.hpp"
#include <cstdio>
namespace da { namespace cli {
void print_help(){
    std::printf(
        "usage:\n"
        "  da3-cli info  --model <gguf>\n"
        "  da3-cli depth --model <gguf> --input <img> [--pfm <out.pfm>] [--png <out.png>] [--pose <out.json>] [--no-invert]\n"
        "  da3-cli depth --model <anyview.gguf> --metric-model <metric.gguf> --input <img> [--pfm <out.pfm>]   (nested metric-scale depth)\n"
        "  da3-cli depth --model <gguf> --input a.png --input b.png [--out-prefix out] [--no-invert]   (multi-view)\n"
        "  da3-cli reconstruct --model <giant.gguf> --input <img> --ply <out.ply> [--pose <out.json>]\n"
        "  da3-cli quantize <in.gguf> <out.gguf> <type>   (type: f16|q8_0|q6_k|q5_k|q4_k)\n");
}
Parsed parse(int argc, char** argv){
    Parsed r;
    if (argc < 2){ r.sub = Sub::Help; return r; }
    std::string first = argv[1];
    if (first == "info"){
        r.sub = Sub::Info;
        for (int i=2;i<argc;++i){
            std::string a = argv[i];
            if (a == "--model" && i+1<argc){ r.model = argv[++i]; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.model.empty()) r.error = "info: --model required";
        return r;
    }
    if (first == "depth"){
        r.sub = Sub::Depth;
        for (int i=2;i<argc;++i){
            std::string a = argv[i];
            if (a == "--model" && i+1<argc){ r.model = argv[++i]; }
            else if (a == "--metric-model" && i+1<argc){ r.metric_model = argv[++i]; }
            else if (a == "--input" && i+1<argc){ std::string in = argv[++i]; r.inputs.push_back(in); if (r.input.empty()) r.input = in; }
            else if (a == "--pfm" && i+1<argc){ r.output_pfm = argv[++i]; }
            else if (a == "--png" && i+1<argc){ r.output_png = argv[++i]; }
            else if (a == "--pose" && i+1<argc){ r.output_pose = argv[++i]; }
            else if (a == "--out-prefix" && i+1<argc){ r.out_prefix = argv[++i]; }
            else if (a == "--no-invert"){ r.invert = false; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.model.empty()) r.error = "depth: --model required";
        else if (r.inputs.empty()) r.error = "depth: --input required";
        return r;
    }
    if (first == "reconstruct"){
        r.sub = Sub::Reconstruct;
        for (int i=2;i<argc;++i){
            std::string a = argv[i];
            if (a == "--model" && i+1<argc){ r.model = argv[++i]; }
            else if (a == "--input" && i+1<argc){ r.input = argv[++i]; }
            else if (a == "--ply" && i+1<argc){ r.output_ply = argv[++i]; }
            else if (a == "--pose" && i+1<argc){ r.output_pose = argv[++i]; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.model.empty()) r.error = "reconstruct: --model required";
        else if (r.input.empty()) r.error = "reconstruct: --input required";
        else if (r.output_ply.empty()) r.error = "reconstruct: --ply required";
        return r;
    }
    if (first == "quantize"){
        r.sub = Sub::Quantize;
        std::vector<std::string> pos;
        for (int i=2;i<argc;++i) pos.push_back(argv[i]);
        if (pos.size() != 3){ r.error = "quantize: usage: da3-cli quantize <in.gguf> <out.gguf> <type>"; return r; }
        r.q_in = pos[0]; r.q_out = pos[1]; r.q_type = pos[2];
        return r;
    }
    if (first == "help" || first == "-h" || first == "--help"){ r.sub = Sub::Help; return r; }
    r.error = "unknown subcommand: " + first;
    return r;
}
}}
