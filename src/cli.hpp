#pragma once
#include <string>
namespace da { namespace cli {
enum class Sub { Info, Help, None };
struct Parsed { Sub sub = Sub::None; std::string model; std::string error; };
Parsed parse(int argc, char** argv);
void print_help();
}}
