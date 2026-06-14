#include "cli.hpp"
#include <cstdio>
namespace da { namespace cli {
void print_help(){ std::printf("usage: da3-cli info --model <gguf>\n"); }
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
    if (first == "help" || first == "-h" || first == "--help"){ r.sub = Sub::Help; return r; }
    r.error = "unknown subcommand: " + first;
    return r;
}
}}
