#pragma once
// Geometry evaluation metrics for the streaming-alignment validation suite
// (tasks A/B/C). These are the MEASURING INSTRUMENTS: reconstruction quality
// (Chamfer, F-score), trajectory accuracy (ATE/RPE), and ghosting proxies
// (surface thickness, duplicate ratio). Header-only so any test can include it;
// each metric is exercised by test_geom_metrics.cpp against known answers.
//
// Point sets are flat double[3N] (x,y,z interleaved), matching src/stream.cpp's
// working representation. NN queries are brute-force O(na*nb) — fine for the
// few-thousand-point synthetic scenes used in tests; swap in a KD-tree only if a
// test ever needs >~1e5 points.
#include "sim3.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace da { namespace eval {

// ---- nearest-neighbour helpers ------------------------------------------
// Squared distance from point a[3] to its nearest neighbour in B[3*nb].
inline double nn_dist2(const double* a, const double* B, int nb) {
    double best = 1e300;
    for (int j = 0; j < nb; ++j) {
        double dx = a[0]-B[3*j+0], dy = a[1]-B[3*j+1], dz = a[2]-B[3*j+2];
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best) best = d2;
    }
    return best;
}

// ---- Chamfer distance ----------------------------------------------------
// a_to_b = mean over A of distance to nearest B point; symmetric average is the
// reported chamfer. Zero iff the sets coincide; equals the offset for a rigid
// shift. Lower = closer to ground-truth surface.
struct ChamferResult { double a_to_b = 0, b_to_a = 0, chamfer = 0; };
inline ChamferResult chamfer(const double* A, int na, const double* B, int nb) {
    ChamferResult r;
    if (na <= 0 || nb <= 0) return r;
    double sab = 0.0;
    for (int i = 0; i < na; ++i) sab += std::sqrt(nn_dist2(A+3*i, B, nb));
    double sba = 0.0;
    for (int j = 0; j < nb; ++j) sba += std::sqrt(nn_dist2(B+3*j, A, na));
    r.a_to_b = sab / na;
    r.b_to_a = sba / nb;
    r.chamfer = 0.5 * (r.a_to_b + r.b_to_a);
    return r;
}

// ---- F-score @ tau -------------------------------------------------------
// precision = frac of A within tau of B; recall = frac of B within tau of A;
// f = harmonic mean. The standard reconstruction completeness/accuracy score;
// 1.0 for coincident sets, degrades as they separate past tau.
struct FScore { double precision = 0, recall = 0, f = 0; };
inline FScore fscore(const double* A, int na, const double* B, int nb, double tau) {
    FScore r;
    if (na <= 0 || nb <= 0 || !(tau > 0)) return r;
    const double t2 = tau * tau;
    int pa = 0; for (int i = 0; i < na; ++i) if (nn_dist2(A+3*i, B, nb) <= t2) ++pa;
    int pb = 0; for (int j = 0; j < nb; ++j) if (nn_dist2(B+3*j, A, na) <= t2) ++pb;
    r.precision = (double)pa / na;
    r.recall    = (double)pb / nb;
    double s = r.precision + r.recall;
    r.f = (s > 0) ? 2.0 * r.precision * r.recall / s : 0.0;
    return r;
}

// ---- surface thickness ---------------------------------------------------
// Std-dev of signed distances of P to a known plane (point p0, unit normal n).
// A single clean sheet returns ~its noise sigma; a doubled sheet separated by
// delta returns ~delta/2 (the std of {0, delta} about their mean). This is the
// direct GHOSTING metric when the true surface is known.
inline double thickness_to_plane(const double* P, int n,
                                 const double p0[3], const double nrm[3]) {
    if (n <= 0) return 0.0;
    double nn = std::sqrt(nrm[0]*nrm[0] + nrm[1]*nrm[1] + nrm[2]*nrm[2]);
    if (!(nn > 0)) return 0.0;
    double ux = nrm[0]/nn, uy = nrm[1]/nn, uz = nrm[2]/nn;
    double mean = 0.0;
    std::vector<double> sd(n);
    for (int i = 0; i < n; ++i) {
        double dx = P[3*i+0]-p0[0], dy = P[3*i+1]-p0[1], dz = P[3*i+2]-p0[2];
        sd[i] = dx*ux + dy*uy + dz*uz;
        mean += sd[i];
    }
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) { double e = sd[i]-mean; var += e*e; }
    return std::sqrt(var / n);
}

// ---- duplicate ratio -----------------------------------------------------
// Fraction of points that have ANOTHER point within eps. ~1.0 for a doubled
// (un-merged) cloud, ~0 for a cloud whose spacing exceeds eps. A GT-free
// ghosting proxy usable on real clips.
inline double duplicate_ratio(const double* P, int n, double eps) {
    if (n <= 1 || !(eps > 0)) return 0.0;
    const double e2 = eps * eps;
    int dup = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            double dx = P[3*i+0]-P[3*j+0], dy = P[3*i+1]-P[3*j+1], dz = P[3*i+2]-P[3*j+2];
            if (dx*dx + dy*dy + dz*dz <= e2) { ++dup; break; }
        }
    }
    return (double)dup / n;
}

// ---- trajectory error (ATE / RPE) ---------------------------------------
// ATE: align estimated positions to GT with a single Umeyama Sim3 (removes the
// arbitrary global frame — exactly what a fused reconstruction is free to
// choose), then RMSE of residual translations. This is the standard SLAM
// absolute-trajectory metric. Returns <0 on failure (degenerate alignment).
inline double ate_rmse(const std::vector<std::array<double,3>>& est,
                       const std::vector<std::array<double,3>>& gt) {
    const int M = (int)std::min(est.size(), gt.size());
    if (M < 3) return -1.0;
    std::vector<double> src(3*M), tgt(3*M), w(M, 1.0);
    for (int i = 0; i < M; ++i) {
        src[3*i+0]=est[i][0]; src[3*i+1]=est[i][1]; src[3*i+2]=est[i][2];
        tgt[3*i+0]=gt[i][0];  tgt[3*i+1]=gt[i][1];  tgt[3*i+2]=gt[i][2];
    }
    Sim3 T; double rms = 0;
    // min_pts=3: a Sim3 needs only 3 non-collinear correspondences; trajectories
    // are short (the solver's default of 50 is for dense overlap point clouds).
    if (!umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, T, rms, /*min_pts=*/3))
        return -1.0;
    double se = 0.0;
    for (int i = 0; i < M; ++i) {
        double p[3] = { src[3*i+0], src[3*i+1], src[3*i+2] }, q[3];
        sim3_apply(T, p, q);
        double dx=q[0]-tgt[3*i+0], dy=q[1]-tgt[3*i+1], dz=q[2]-tgt[3*i+2];
        se += dx*dx + dy*dy + dz*dz;
    }
    return std::sqrt(se / M);
}

// RPE (translational): RMSE of the difference between consecutive relative
// motions of est vs gt. Insensitive to the global frame AND to accumulated
// drift — it isolates per-step consistency. Returns <0 on failure.
inline double rpe_trans_rmse(const std::vector<std::array<double,3>>& est,
                             const std::vector<std::array<double,3>>& gt) {
    const int M = (int)std::min(est.size(), gt.size());
    if (M < 2) return -1.0;
    double se = 0.0; int cnt = 0;
    for (int i = 0; i + 1 < M; ++i) {
        double ex=est[i+1][0]-est[i][0], ey=est[i+1][1]-est[i][1], ez=est[i+1][2]-est[i][2];
        double gx=gt[i+1][0]-gt[i][0],  gy=gt[i+1][1]-gt[i][1],  gz=gt[i+1][2]-gt[i][2];
        double dx=ex-gx, dy=ey-gy, dz=ez-gz;
        se += dx*dx + dy*dy + dz*dz; ++cnt;
    }
    return cnt ? std::sqrt(se / cnt) : -1.0;
}

}} // namespace da::eval
