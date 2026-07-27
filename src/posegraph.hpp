#pragma once
// Sim3 pose-graph optimization (task C). The streaming stitcher chains window
// poses open-loop (G_k = G_{k-1} ∘ S_rel_k), so per-seam error accumulates as
// drift; when the trajectory revisits an area (loop closure) the same surface is
// placed twice => double walls that no local fusion (A) or seam ICP (B) can fix.
//
// This is a global fix: treat each window's Sim3 pose G_k as a graph node, each
// relative alignment (sequential seams + loop closures) as an edge measurement Z
// constraining G_i^{-1} ∘ G_j = Z, and optimize the nodes so the loop-closure
// residual is spread consistently around the whole trajectory (Levenberg-
// Marquardt on the Sim3 manifold). Scale is a first-class node DOF (windows drift
// in monocular scale, not just pose).
//
// NO ggml / NO engine — pure host geometry, validated against known poses and
// gtsam's Pose3 pose-graph + Similarity3.Align (scripts/oracle_posegraph.py).
#include "sim3.hpp"

#include <vector>

namespace da {

// Edge measurement: the relative Sim3 from node i's frame to node j's frame,
// i.e. the constraint G_i^{-1} ∘ G_j == z (residual is zero when it holds).
struct PoseEdge {
    int    i = 0, j = 0;
    Sim3   z;             // measured relative transform (i -> j)
    double info = 1.0;    // isotropic information weight (>=0)
    bool   loop = false;  // cosmetic: marks a loop-closure edge (vs sequential)
};

struct PoseGraph {
    std::vector<Sim3>     nodes;   // node poses (local_k -> global); optimized in place
    std::vector<PoseEdge> edges;
    std::vector<char>     fixed;   // per-node gauge lock; empty => node 0 fixed
};

struct PoseGraphParams {
    int    max_iter = 50;
    double lambda0  = 1e-4;   // initial LM damping
    double tol      = 1e-10;  // stop when relative cost improvement drops below this
    double step_eps = 1e-6;   // finite-difference step for edge Jacobians
};

struct PoseGraphResult {
    bool   ok        = false;
    int    iters     = 0;
    double cost0     = 0;     // 0.5 * Σ info ||r||²  before optimization
    double cost      = 0;     // ... after
    double max_dx    = 0;     // L∞ of the last accepted increment
    int    free_dofs = 0;
};

// Optimize graph.nodes in place. Returns convergence info. A graph with only
// sequential edges whose measurements are already consistent with the node
// estimates is a no-op (cost stays ~0).
PoseGraphResult optimize_pose_graph(PoseGraph& graph, const PoseGraphParams& p = {});

// Edge residual r[7] = [ E.t(3), so3_log(E.R)(3), log(E.s)(1) ] with
// E = z^{-1} ∘ G_i^{-1} ∘ G_j. Zero iff G_i^{-1}∘G_j == z. Exposed for tests.
void pose_edge_residual(const Sim3& Gi, const Sim3& Gj, const Sim3& z, double r[7]);

} // namespace da
