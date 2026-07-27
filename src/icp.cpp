#include "icp.hpp"

#include "linalg.hpp"
#include "spatial_hash.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace da {

namespace {

// Rodrigues: rotation vector omega (row-major 3x3 R). Exact (no small-angle assumption).
void rodrigues(const double w[3], double R[9]) {
    double th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (th < 1e-12) { R[0]=R[4]=R[8]=1; R[1]=R[2]=R[3]=R[5]=R[6]=R[7]=0; return; }
    double kx=w[0]/th, ky=w[1]/th, kz=w[2]/th;
    double c=std::cos(th), s=std::sin(th), v=1-c;
    R[0]=c+kx*kx*v;    R[1]=kx*ky*v-kz*s; R[2]=kx*kz*v+ky*s;
    R[3]=ky*kx*v+kz*s; R[4]=c+ky*ky*v;    R[5]=ky*kz*v-kx*s;
    R[6]=kz*kx*v-ky*s; R[7]=kz*ky*v+kx*s; R[8]=c+kz*kz*v;
}

} // namespace

void estimate_normals(const std::vector<double>& xyz, double radius,
                      const double* viewpoint, std::vector<double>& normals) {
    const int n = (int)(xyz.size()/3);
    normals.assign((size_t)3*n, 0.0);
    if (n == 0 || !(radius > 0)) return;
    SpatialHash hash(xyz.data(), n, radius);
    // Per-point PCA into normals[i] from const read-only hash queries — independent
    // per point, so parallel is bitwise-identical. nb is declared inside the loop so
    // each thread/iteration gets its own neighbour buffer.
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; ++i) {
        const double* q = xyz.data() + 3*i;
        std::vector<int> nb;
        hash.radius(q, radius, nb);
        if ((int)nb.size() < 3) continue;               // too few -> leave zero (skipped)
        double mu[3] = {0,0,0};
        for (int j : nb) { mu[0]+=xyz[3*j]; mu[1]+=xyz[3*j+1]; mu[2]+=xyz[3*j+2]; }
        double inv = 1.0/nb.size(); mu[0]*=inv; mu[1]*=inv; mu[2]*=inv;
        double C[9] = {0,0,0,0,0,0,0,0,0};
        for (int j : nb) {
            double a=xyz[3*j]-mu[0], b=xyz[3*j+1]-mu[1], c=xyz[3*j+2]-mu[2];
            C[0]+=a*a; C[1]+=a*b; C[2]+=a*c;
            C[3]+=a*b; C[4]+=b*b; C[5]+=b*c;
            C[6]+=a*c; C[7]+=b*c; C[8]+=c*c;
        }
        double nrm[3]; linalg::smallest_eigvec(C, 3, nrm);
        double ln = std::sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2]);
        if (!(ln > 0)) continue;
        nrm[0]/=ln; nrm[1]/=ln; nrm[2]/=ln;
        if (viewpoint) {   // orient toward viewpoint (cosmetic; point-to-plane is sign-invariant)
            double vx=viewpoint[0]-q[0], vy=viewpoint[1]-q[1], vz=viewpoint[2]-q[2];
            if (nrm[0]*vx + nrm[1]*vy + nrm[2]*vz < 0) { nrm[0]=-nrm[0]; nrm[1]=-nrm[1]; nrm[2]=-nrm[2]; }
        }
        normals[3*i+0]=nrm[0]; normals[3*i+1]=nrm[1]; normals[3*i+2]=nrm[2];
    }
}

IcpResult icp_point_to_plane(const std::vector<double>& src_xyz,
                             const std::vector<double>& tgt_xyz,
                             const std::vector<double>& tgt_normals,
                             const Sim3& init, const IcpParams& p) {
    IcpResult res;
    res.T = init;
    const int ns = (int)(src_xyz.size()/3);
    const int nt = (int)(tgt_xyz.size()/3);
    if (ns == 0 || nt == 0 || (int)tgt_normals.size() != 3*nt) return res;  // ok stays false

    // Working source points, pre-transformed by the Umeyama init into the target
    // frame. ICP then finds a rigid increment dT (target frame); T_final = dT ∘ init.
    std::vector<double> P((size_t)3*ns);
    for (int i = 0; i < ns; ++i) sim3_apply(init, &src_xyz[3*i], &P[3*i]);

    const double corr_cell = std::max(p.max_corr_dist, 1e-6);
    SpatialHash hash(tgt_xyz.data(), nt, corr_cell);

    Sim3 dT;  // accumulated rigid increment (s=1), target frame
    res.ok = true;

    // point-to-plane RMS of the current P against the target planes.
    auto plane_rms = [&](int& n_corr) -> double {
        double se = 0.0; long c = 0;
        #pragma omp parallel for reduction(+:se,c) schedule(dynamic, 512)
        for (int i = 0; i < ns; ++i) {
            double d2;
            int j = hash.nearest(&P[3*i], p.max_corr_dist, d2);
            if (j < 0) continue;
            const double* nj = &tgt_normals[3*j];
            if (nj[0]==0 && nj[1]==0 && nj[2]==0) continue;
            double r = (P[3*i]-tgt_xyz[3*j])*nj[0] + (P[3*i+1]-tgt_xyz[3*j+1])*nj[1] + (P[3*i+2]-tgt_xyz[3*j+2])*nj[2];
            se += r*r; ++c;
        }
        n_corr = (int)c;
        return c ? std::sqrt(se / c) : 0.0;
    };

    int c0 = 0; res.rms_before = plane_rms(c0);
    double prev_rms = res.rms_before;

    for (int it = 0; it < p.max_iter; ++it) {
        // Accumulate the linearised normal system H x = -g, x = [omega(3), t(3)].
        // Fused correspondence-find + Huber-weighted normal-equation accumulation:
        // each source point's Huber weight depends only on its own residual (the
        // delta is a constant), so this is a clean per-point reduction into H/g
        // with no cross-point dependency — parallel over the ns source points.
        double H[36] = {0}; double g[6] = {0};
        long c = 0;
        const double w_delta = p.huber_delta;
        #pragma omp parallel for reduction(+:H[:36], g[:6], c) schedule(dynamic, 512)
        for (int i = 0; i < ns; ++i) {
            double d2;
            int j = hash.nearest(&P[3*i], p.max_corr_dist, d2);
            if (j < 0) continue;
            const double* nj = &tgt_normals[3*j];
            if (nj[0]==0 && nj[1]==0 && nj[2]==0) continue;
            double px=P[3*i], py=P[3*i+1], pz=P[3*i+2];
            double r = (px-tgt_xyz[3*j])*nj[0] + (py-tgt_xyz[3*j+1])*nj[1] + (pz-tgt_xyz[3*j+2])*nj[2];
            // a = [ p x n , n ]
            double a[6] = { py*nj[2]-pz*nj[1], pz*nj[0]-px*nj[2], px*nj[1]-py*nj[0], nj[0], nj[1], nj[2] };
            double w = 1.0;
            if (w_delta > 0 && std::fabs(r) > w_delta) w = w_delta / std::fabs(r);
            for (int u=0; u<6; ++u) {
                g[u] += w * a[u] * r;
                for (int v=0; v<6; ++v) H[6*u+v] += w * a[u] * a[v];
            }
            ++c;
        }
        res.corr = (int)c;
        if (c < p.min_corr) break;   // too little overlap -> keep the best-so-far (init if it=0)

        // Solve in the well-conditioned eigen-subspace of H (rank-truncated).
        // Unconstrained DOFs (small eigenvalue) get zero increment => prior kept.
        double eval[6], evec[36];
        linalg::jacobi_eigen_sym(H, 6, eval, evec);
        double lmax = 0; for (int k=0;k<6;++k) lmax = std::max(lmax, eval[k]);
        double x[6] = {0,0,0,0,0,0};
        int rank = 0;
        if (lmax > 0) {
            for (int k = 0; k < 6; ++k) {
                if (eval[k] <= p.rank_tol * lmax || eval[k] <= 1e-15) continue;
                // coeff = (v_k . (-g)) / lambda_k ; eigenvector k is column k.
                double vg = 0; for (int u=0; u<6; ++u) vg += evec[6*u+k] * (-g[u]);
                double coeff = vg / eval[k];
                for (int u=0; u<6; ++u) x[u] += coeff * evec[6*u+k];
                ++rank;
            }
        }
        res.rank = rank;

        // Build the incremental transform (R from omega, t) and apply to P.
        double dR[9]; double w3[3] = {x[0], x[1], x[2]};
        rodrigues(w3, dR);
        double dt[3] = {x[3], x[4], x[5]};
        for (int i = 0; i < ns; ++i) {
            double px=P[3*i], py=P[3*i+1], pz=P[3*i+2];
            P[3*i+0] = dR[0]*px + dR[1]*py + dR[2]*pz + dt[0];
            P[3*i+1] = dR[3]*px + dR[4]*py + dR[5]*pz + dt[1];
            P[3*i+2] = dR[6]*px + dR[7]*py + dR[8]*pz + dt[2];
        }
        Sim3 step; for (int k=0;k<9;++k) step.R[k]=dR[k]; step.t[0]=dt[0]; step.t[1]=dt[1]; step.t[2]=dt[2];
        dT = sim3_compose(step, dT);

        int cc = 0; double rms = plane_rms(cc);
        res.rms_iters.push_back(rms);
        res.iters = it + 1;

        double step_norm = 0; for (int k=0;k<6;++k) step_norm += x[k]*x[k];
        if (std::sqrt(step_norm) < p.tol) break;
        // relative-RMSE early stop (open3d-style): the residual has stopped
        // improving meaningfully, so further iterations only burn NN queries.
        if (prev_rms > 0 && (prev_rms - rms) <= p.rel_rmse * prev_rms) break;
        prev_rms = rms;
    }

    int cf = 0; res.rms_after = plane_rms(cf);
    res.T = sim3_compose(dT, init);   // src -> tgt, refined (init's scale preserved)
    return res;
}

} // namespace da
