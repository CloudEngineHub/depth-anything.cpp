#pragma once
#include "ggml.h"

// Winograd F(2x2,3x3) convolution for the DPT head's 3x3 stride-1 convs.
//
// Uses 2.25x fewer multiplies than direct conv. Implemented as a CPU custom op
// (ggml_custom_4d) with an AVX-512 inner GEMV over the winograd domain.
//
// Tensor layout (ggml ne, fastest dim first):
//   x : [W, H, IC, N]    input feature map (F32)
//   w : [3, 3, IC, OC]   filter (torch (OC,IC,KH,KW) reversed)  (F32)
//   out: [Wout, Hout, OC, N]  with Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
//
// Only valid for KW==KH==3, stride==1, F32 inputs. `pad` is arbitrary (the DPT
// head always uses pad=1 -> same-size output). Bias is NOT applied here; add it
// after with ggml_add (matching the direct-conv path).
namespace da {

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

} // namespace da
