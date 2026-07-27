#pragma once
// Weighted Umeyama Sim3 alignment (NO ggml, NO engine). Used by the sliding-window
// streaming stitcher to map each window's local point cloud into a single global
// frame. Mirrors da3_streaming/loop_utils/alignment_torch.py: weighted centroids,
// rotation from a cross-covariance, RMS-ratio scale, translation, Huber IRLS.
namespace da {

// Similarity transform: y = s * R * x + t.  R is row-major 3x3.
struct Sim3 {
    double s = 1.0;
    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double t[3] = {0,0,0};
};

// out = s*R*p + t. Safe when out aliases p.
void sim3_apply(const Sim3& T, const double p[3], double out[3]);

// Returns A∘B with (A∘B)(x) = A(B(x)).
Sim3 sim3_compose(const Sim3& A, const Sim3& B);

// Returns T^{-1}: if y = s*R*x + t then x = (1/s)*R^T*(y - t).
Sim3 sim3_inverse(const Sim3& T);

// Weighted Umeyama similarity fitting src -> tgt, refined by Huber IRLS.
//   src,tgt : length 3*M (x,y,z interleaved).
//   w       : length M, per-correspondence weight (>=0; e.g. min confidence).
// On success fills `out` (maps src->tgt) and `rms` (weighted RMS residual) and
// returns true. Returns false if degenerate: M < min_pts, all weights ~0, or a
// non-finite result. When the source points carry no scale information (nearly
// coincident) the scale is pinned to 1; when the rotation is ill-posed (collinear/
// coincident) R is left at identity (translation+scale only).
bool umeyama_sim3_weighted(const double* src, const double* tgt, const double* w,
                           int M, Sim3& out, double& rms, int min_pts = 50);

} // namespace da
