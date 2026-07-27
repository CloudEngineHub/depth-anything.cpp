#include "sim3.hpp"
#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace da {

void sim3_apply(const Sim3& T, const double p[3], double out[3]) {
    double x = p[0], y = p[1], z = p[2];
    double rx = T.R[0]*x + T.R[1]*y + T.R[2]*z;
    double ry = T.R[3]*x + T.R[4]*y + T.R[5]*z;
    double rz = T.R[6]*x + T.R[7]*y + T.R[8]*z;
    out[0] = T.s*rx + T.t[0];
    out[1] = T.s*ry + T.t[1];
    out[2] = T.s*rz + T.t[2];
}

Sim3 sim3_compose(const Sim3& A, const Sim3& B) {
    // (A∘B)(x) = sA RA (sB RB x + tB) + tA
    //          = (sA sB)(RA RB) x + (sA RA tB + tA)
    Sim3 C;
    C.s = A.s * B.s;
    linalg::mat3_mul(A.R, B.R, C.R);
    double rb[3] = {
        A.R[0]*B.t[0] + A.R[1]*B.t[1] + A.R[2]*B.t[2],
        A.R[3]*B.t[0] + A.R[4]*B.t[1] + A.R[5]*B.t[2],
        A.R[6]*B.t[0] + A.R[7]*B.t[1] + A.R[8]*B.t[2],
    };
    C.t[0] = A.s*rb[0] + A.t[0];
    C.t[1] = A.s*rb[1] + A.t[1];
    C.t[2] = A.s*rb[2] + A.t[2];
    return C;
}

Sim3 sim3_inverse(const Sim3& T) {
    // y = s R x + t  =>  x = (1/s) R^T (y - t).
    Sim3 inv;
    inv.s = (T.s != 0.0) ? 1.0 / T.s : 0.0;
    linalg::mat3_transpose(T.R, inv.R);
    // t' = -(1/s) R^T t
    double rtt[3] = {
        inv.R[0]*T.t[0] + inv.R[1]*T.t[1] + inv.R[2]*T.t[2],
        inv.R[3]*T.t[0] + inv.R[4]*T.t[1] + inv.R[5]*T.t[2],
        inv.R[6]*T.t[0] + inv.R[7]*T.t[1] + inv.R[8]*T.t[2],
    };
    inv.t[0] = -inv.s * rtt[0];
    inv.t[1] = -inv.s * rtt[1];
    inv.t[2] = -inv.s * rtt[2];
    return inv;
}

// Unit quaternion (w,x,y,z) -> row-major active rotation matrix.
static void quat_to_R(const double q[4], double R[9]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2*(y*y + z*z); R[1] = 2*(x*y - w*z);     R[2] = 2*(x*z + w*y);
    R[3] = 2*(x*y + w*z);     R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - w*x);
    R[6] = 2*(x*z - w*y);     R[7] = 2*(y*z + w*x);     R[8] = 1 - 2*(x*x + y*y);
}

// One weighted closed-form solve given raw (un-normalized) weights W[M].
// Returns false if total weight <= 0. On return: out filled; den = Σ w||a||²,
// rot_ok=false if the rotation was ill-posed (kept identity).
static bool solve_once(const double* src, const double* tgt, const double* W, int M,
                       Sim3& out, double& den_out, bool& rot_ok) {
    double Wsum = 0.0;
    for (int i = 0; i < M; ++i) Wsum += W[i];
    if (!(Wsum > 0.0) || !std::isfinite(Wsum)) return false;

    // Weighted centroids.
    double mu_s[3] = {0,0,0}, mu_t[3] = {0,0,0};
    for (int i = 0; i < M; ++i) {
        double w = W[i];
        mu_s[0] += w*src[3*i+0]; mu_s[1] += w*src[3*i+1]; mu_s[2] += w*src[3*i+2];
        mu_t[0] += w*tgt[3*i+0]; mu_t[1] += w*tgt[3*i+1]; mu_t[2] += w*tgt[3*i+2];
    }
    for (int k = 0; k < 3; ++k) { mu_s[k] /= Wsum; mu_t[k] /= Wsum; }

    // Cross-covariance Cxx = Σ w a b^T (a=src centered, b=tgt centered) and the
    // weighted variances of a (den) and b (num) for the RMS-ratio scale.
    double C[9] = {0,0,0, 0,0,0, 0,0,0};
    double den = 0.0;
    for (int i = 0; i < M; ++i) {
        double w = W[i];
        double a0 = src[3*i+0]-mu_s[0], a1 = src[3*i+1]-mu_s[1], a2 = src[3*i+2]-mu_s[2];
        double b0 = tgt[3*i+0]-mu_t[0], b1 = tgt[3*i+1]-mu_t[1], b2 = tgt[3*i+2]-mu_t[2];
        C[0] += w*a0*b0; C[1] += w*a0*b1; C[2] += w*a0*b2;
        C[3] += w*a1*b0; C[4] += w*a1*b1; C[5] += w*a1*b2;
        C[6] += w*a2*b0; C[7] += w*a2*b1; C[8] += w*a2*b2;
        den += w*(a0*a0 + a1*a1 + a2*a2);
    }
    den_out = den;

    // Rotation via Horn's quaternion method: largest-eigenvalue eigenvector of the
    // symmetric 4x4 N built from C. Reflection-free by construction.
    double Sxx=C[0], Sxy=C[1], Sxz=C[2];
    double Syx=C[3], Syy=C[4], Syz=C[5];
    double Szx=C[6], Szy=C[7], Szz=C[8];
    double N[16] = {
        Sxx+Syy+Szz,  Syz-Szy,       Szx-Sxz,       Sxy-Syx,
        Syz-Szy,      Sxx-Syy-Szz,   Sxy+Syx,       Szx+Sxz,
        Szx-Sxz,      Sxy+Syx,      -Sxx+Syy-Szz,   Syz+Szy,
        Sxy-Syx,      Szx+Sxz,       Syz+Szy,      -Sxx-Syy+Szz,
    };
    double evals[4], evecs[16];
    linalg::jacobi_eigen_sym(N, 4, evals, evecs);
    int best = 0, second = -1;
    for (int i = 1; i < 4; ++i) if (evals[i] > evals[best]) best = i;
    for (int i = 0; i < 4; ++i) {
        if (i == best) continue;
        if (second < 0 || evals[i] > evals[second]) second = i;
    }
    rot_ok = true;
    Sim3 r;  // identity by default
    double gap = evals[best] - (second >= 0 ? evals[second] : 0.0);
    double mag = std::fabs(evals[best]) + 1e-30;
    if (gap < 1e-9 * mag) {
        rot_ok = false;  // ill-posed (collinear / coincident) -> keep R = I
    } else {
        double q[4] = { evecs[0*4+best], evecs[1*4+best], evecs[2*4+best], evecs[3*4+best] };
        double qn = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (!(qn > 0.0) || !std::isfinite(qn)) { rot_ok = false; }
        else { for (int k = 0; k < 4; ++k) q[k] /= qn; quat_to_R(q, r.R); }
    }

    // Optimal Umeyama scale s = tr(R·C) / Σ w||a||². Unlike the RMS-ratio form it
    // is linear (not quadratic) in the target points, so gross outliers downweighted
    // by Huber can't inflate it; identical to the RMS ratio on clean data. Pinned to
    // 1 when the source carries no spread.
    double trRC = 0.0;
    for (int rr = 0; rr < 3; ++rr)
        for (int kk = 0; kk < 3; ++kk)
            trRC += r.R[rr*3+kk] * C[kk*3+rr];
    r.s = (den > 1e-30) ? trRC / den : 1.0;
    if (!std::isfinite(r.s) || r.s <= 0.0) r.s = 1.0;

    // Translation: t = mu_tgt - s*R*mu_src.
    double rs[3];
    rs[0] = r.R[0]*mu_s[0] + r.R[1]*mu_s[1] + r.R[2]*mu_s[2];
    rs[1] = r.R[3]*mu_s[0] + r.R[4]*mu_s[1] + r.R[5]*mu_s[2];
    rs[2] = r.R[6]*mu_s[0] + r.R[7]*mu_s[1] + r.R[8]*mu_s[2];
    r.t[0] = mu_t[0] - r.s*rs[0];
    r.t[1] = mu_t[1] - r.s*rs[1];
    r.t[2] = mu_t[2] - r.s*rs[2];
    out = r;
    return true;
}

bool umeyama_sim3_weighted(const double* src, const double* tgt, const double* w,
                           int M, Sim3& out, double& rms, int min_pts) {
    rms = 0.0;
    if (M < min_pts) return false;

    std::vector<double> W(w, w + M);          // current IRLS weights
    std::vector<double> resid(M, 0.0);
    Sim3 T;
    double den = 0.0;
    bool rot_ok = false;

    const int max_iter = 8;
    Sim3 prev;
    for (int it = 0; it < max_iter; ++it) {
        if (!solve_once(src, tgt, W.data(), M, T, den, rot_ok)) return false;

        // residuals r_i = || s*R*src_i + t - tgt_i ||
        for (int i = 0; i < M; ++i) {
            double p[3] = { src[3*i+0], src[3*i+1], src[3*i+2] };
            double q[3]; sim3_apply(T, p, q);
            double dx = q[0]-tgt[3*i+0], dy = q[1]-tgt[3*i+1], dz = q[2]-tgt[3*i+2];
            resid[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        // Huber band from the median residual; floor relative to scene extent.
        std::vector<double> tmp(resid);
        size_t mid = tmp.size()/2;
        std::nth_element(tmp.begin(), tmp.begin()+mid, tmp.end());
        double med = tmp[mid];
        double scene = (den > 0.0) ? std::sqrt(den / std::max(1.0, (double)M)) : 1.0;
        double delta = std::max(1.345 * med, 1e-6 * scene + 1e-30);
        // Reweight: conf * huber(r,delta).
        for (int i = 0; i < M; ++i) {
            double hu = (resid[i] <= delta) ? 1.0 : (delta / resid[i]);
            W[i] = w[i] * hu;
        }
        // Convergence: small change in (s,R,t).
        if (it > 0) {
            double d = std::fabs(T.s - prev.s);
            for (int k = 0; k < 9; ++k) d += std::fabs(T.R[k] - prev.R[k]);
            for (int k = 0; k < 3; ++k) d += std::fabs(T.t[k] - prev.t[k]);
            if (d < 1e-12) break;
        }
        prev = T;
    }

    // Final weighted RMS with the converged (conf*huber) weights.
    double sw = 0.0, se = 0.0;
    for (int i = 0; i < M; ++i) { sw += W[i]; se += W[i]*resid[i]*resid[i]; }
    rms = (sw > 0.0) ? std::sqrt(se / sw) : 0.0;

    for (int k = 0; k < 9; ++k) if (!std::isfinite(T.R[k])) return false;
    for (int k = 0; k < 3; ++k) if (!std::isfinite(T.t[k])) return false;
    if (!std::isfinite(T.s)) return false;
    out = T;
    return true;
}

} // namespace da
