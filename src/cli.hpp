#pragma once
#include <string>
#include <vector>
namespace da { namespace cli {
enum class Sub { Info, Depth, Quantize, Help, None };
struct Parsed {
    Sub sub = Sub::None;
    std::string model;
    std::string input, output_pfm, output_png;
    std::string output_pose;
    std::vector<std::string> inputs;   // accumulates repeated --input; >1 => multi-view mode
    std::string out_prefix;            // --out-prefix for multi-view outputs
    std::string q_in, q_out, q_type;   // quantize: <in.gguf> <out.gguf> <type>
    bool invert = true;
    std::string error;
};
Parsed parse(int argc, char** argv);
void print_help();
}}
