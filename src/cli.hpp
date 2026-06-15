#pragma once
#include <string>
#include <vector>
namespace da { namespace cli {
enum class Sub { Info, Depth, Reconstruct, Quantize, Help, None };
struct Parsed {
    Sub sub = Sub::None;
    std::string model;
    std::string metric_model;          // depth: --metric-model => nested metric-scale depth
    std::string input, output_pfm, output_png;
    std::string output_pose;
    std::string output_ply;            // reconstruct: --ply out.ply
    std::vector<std::string> inputs;   // accumulates repeated --input; >1 => multi-view mode
    std::string out_prefix;            // --out-prefix for multi-view outputs
    std::string q_in, q_out, q_type;   // quantize: <in.gguf> <out.gguf> <type>
    bool invert = true;
    // Single-image depth uses the REAL DA3 native-resolution resize by default.
    // --legacy-resize forces the old floor-to-patch path (fixture parity only).
    bool legacy_resize = false;
    std::string error;
};
Parsed parse(int argc, char** argv);
void print_help();
}}
