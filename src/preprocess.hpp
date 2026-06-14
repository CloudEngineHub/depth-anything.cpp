#pragma once
#include "image_io.hpp"
#include "model_loader.hpp"
#include <vector>
namespace da {
struct Preprocessed { int H=0, W=0; std::vector<float> chw; };   // [3,H,W] f32, C-major
bool preprocess(const Image& img, const Config& cfg, Preprocessed& out);
}
