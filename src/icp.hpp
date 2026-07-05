#pragma once
// Point-to-plane ICP for per-seam refinement (task B). Refines the weighted-
// Umeyama Sim3 seam that stream.cpp solves between consecutive windows: Umeyama
// (point-to-point) is biased by the along-ray depth noise of the overlap
// correspondences; point-to-plane instead minimises distance to the target's
// tangent planes, so a source point that is correct-on-the-surface but shifted
// tangentially is not penalised, tightening the surface fit that drives ghosting.
//
// The linearised 6-DOF increment (Low 2004) is solved in the well-conditioned
// eigen-subspace of the normal matrix: a rank-deficient geometry (e.g. a single
// flat wall, where tangential slide + in-plane spin are unobservable — the
// aperture problem) leaves those DOFs at the prior instead of drifting. Scale is
// inherited from the Umeyama init (point-to-plane refines rigid pose only).
//
// NO ggml / NO engine — pure host geometry, so it is validated against known
// transforms and open3d.registration_icp (scripts/oracle_icp.py).
#include "sim3.hpp"

#include <vector>

namespace da {

struct IcpParams {
    int    max_iter      = 30;
    double max_corr_dist = 0.10;  // reject correspondences beyond this (m)
    double tol           = 1e-7;  // convergence: L2 norm of the (omega,t) step
    double rel_rmse      = 1e-6;  // convergence: relative point-to-plane RMSE improvement
    double huber_delta   = 0.05;  // robust band on point-to-plane residual (m); <=0 disables
    double rank_tol      = 1e-3;  // eigval/eigval_max below which a DOF is deemed unconstrained
    double normal_radius = 0.15;  // neighbourhood radius for target-normal PCA (m)
    int    min_corr      = 20;    // bail (return the init unchanged) below this many matches
};

struct IcpResult {
    Sim3   T;                       // refined source->target (scale inherited from init)
    double rms_before   = 0;        // point-to-plane RMS at the init transform
    double rms_after    = 0;        // point-to-plane RMS at convergence
    int    iters        = 0;
    int    corr         = 0;        // # correspondences at the last iteration
    int    rank         = 0;        // effective rank of the last 6x6 normal system (6=full)
    bool   ok           = false;    // false only on bad/degenerate input (init returned)
    std::vector<double> rms_iters;  // point-to-plane RMS after each iteration (monotonicity)
};

// Per-point normals via local PCA over a radius neighbourhood (smallest-variance
// eigenvector). xyz is 3N (x,y,z interleaved); normals is filled 3N. If viewpoint
// is non-null, each normal is flipped to face it (cosmetic — point-to-plane is
// sign-invariant). Points with too few neighbours get a zero normal (skipped by ICP).
void estimate_normals(const std::vector<double>& xyz, double radius,
                      const double* viewpoint, std::vector<double>& normals);

// Point-to-plane ICP refining `init` (maps src_xyz -> tgt_xyz). tgt_normals is
// 3*|tgt| (e.g. from estimate_normals); a zero normal marks a skipped target.
// Correspondences are re-established by nearest-neighbour each iteration. Returns
// the refined transform (or `init` unchanged if it bailed) in result.T.
IcpResult icp_point_to_plane(const std::vector<double>& src_xyz,
                             const std::vector<double>& tgt_xyz,
                             const std::vector<double>& tgt_normals,
                             const Sim3& init, const IcpParams& p);

} // namespace da
