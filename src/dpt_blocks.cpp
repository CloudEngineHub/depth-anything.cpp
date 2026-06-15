#include "dpt_blocks.hpp"

namespace da {

// Reshape an [OC] bias to [1,1,OC,1] so it broadcasts over spatial dims of a
// conv output [W,H,OC,N].
static ggml_tensor* bias_chw(ggml_context* ctx, ggml_tensor* b) {
    return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
}

ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x,
                    int stride, int pad) {
    // w:[KW,KH,IC,OC] x:[W,H,IC,N] -> [W_out,H_out,OC,N]
    ggml_tensor* y = ggml_conv_2d(ctx, w, x, stride, stride, pad, pad, 1, 1);
    if (b) y = ggml_add(ctx, y, bias_chw(ctx, b));
    return y;
}

ggml_tensor* conv_transpose2d_p0(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                 ggml_tensor* x, int stride) {
    // ggml_conv_transpose_2d_p0 expects kernel layout (KW,KH,Cout,Cin) == [KW,KH,OC,IC],
    // which is exactly the straight dim-reversal of torch ConvTranspose2d (IC,OC,KH,KW).
    // x:[W,H,IC,N] -> [W_out,H_out,OC,N]
    ggml_tensor* y = ggml_conv_transpose_2d_p0(ctx, w, x, stride);
    if (b) y = ggml_add(ctx, y, bias_chw(ctx, b));
    return y;
}

ggml_tensor* interp_bilinear_ac(ggml_context* ctx, ggml_tensor* x, int out_w, int out_h) {
    return ggml_interpolate(ctx, x, out_w, out_h, x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
}

ggml_tensor* residual_conv_unit(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* c1w, ggml_tensor* c1b,
                                ggml_tensor* c2w, ggml_tensor* c2b) {
    // out = relu(x); out = conv1(out); out = relu(out); out = conv2(out); return out + x
    ggml_tensor* out = ggml_relu(ctx, x);
    out = conv2d(ctx, c1w, c1b, out, 1, 1);     // 3x3 pad1 stride1
    out = ggml_relu(ctx, out);
    out = conv2d(ctx, c2w, c2b, out, 1, 1);     // 3x3 pad1 stride1
    return ggml_add(ctx, out, x);
}

ggml_tensor* feature_fusion(ggml_context* ctx, ggml_tensor* top, ggml_tensor* lateral,
                            ggml_tensor* rc1c1w, ggml_tensor* rc1c1b, ggml_tensor* rc1c2w, ggml_tensor* rc1c2b,
                            ggml_tensor* rc2c1w, ggml_tensor* rc2c1b, ggml_tensor* rc2c2w, ggml_tensor* rc2c2b,
                            ggml_tensor* outw, ggml_tensor* outb, int out_w, int out_h) {
    ggml_tensor* y = top;
    // Fuse lateral skip connection through resConfUnit1 (only when present).
    if (lateral && rc1c1w) {
        ggml_tensor* res = residual_conv_unit(ctx, lateral, rc1c1w, rc1c1b, rc1c2w, rc1c2b);
        y = ggml_add(ctx, y, res);
    }
    // resConfUnit2
    y = residual_conv_unit(ctx, y, rc2c1w, rc2c1b, rc2c2w, rc2c2b);
    // interpolate: explicit target size, else scale_factor 2.
    int ow = out_w > 0 ? out_w : (int)(y->ne[0] * 2);
    int oh = out_h > 0 ? out_h : (int)(y->ne[1] * 2);
    y = interp_bilinear_ac(ctx, y, ow, oh);
    // out_conv: 1x1 stride1 pad0 WITH bias
    y = conv2d(ctx, outw, outb, y, 1, 0);
    return y;
}

} // namespace da
