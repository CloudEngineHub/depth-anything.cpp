#include "posegraph.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace da {

namespace {

// ---- SO3 exp/log ---------------------------------------------------------
void so3_exp(const double w[3], double R[9]) {
    double th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (th < 1e-12) { R[0]=R[4]=R[8]=1; R[1]=R[2]=R[3]=R[5]=R[6]=R[7]=0; return; }
    double kx=w[0]/th, ky=w[1]/th, kz=w[2]/th;
    double c=std::cos(th), s=std::sin(th), v=1-c;
    R[0]=c+kx*kx*v;    R[1]=kx*ky*v-kz*s; R[2]=kx*kz*v+ky*s;
    R[3]=ky*kx*v+kz*s; R[4]=c+ky*ky*v;    R[5]=ky*kz*v-kx*s;
    R[6]=kz*kx*v-ky*s; R[7]=kz*ky*v+kx*s; R[8]=c+kz*kz*v;
}

void so3_log(const double R[9], double w[3]) {
    double tr = R[0]+R[4]+R[8];
    double c = (tr-1.0)*0.5; c = std::max(-1.0, std::min(1.0, c));
    double th = std::acos(c);
    double vx = R[7]-R[5], vy = R[2]-R[6], vz = R[3]-R[1];  // 2*sin(th)*axis
    if (th < 1e-8) {                                        // small angle: w ~ 0.5*v
        w[0]=0.5*vx; w[1]=0.5*vy; w[2]=0.5*vz; return;
    }
    if (M_PI - th < 1e-6) {                                 // near pi: axis from (R+I) diag
        int k = 0; double bd = R[0];
        if (R[4] > bd) { bd = R[4]; k = 1; }
        if (R[8] > bd) { bd = R[8]; k = 2; }
        double ax[3];
        ax[k] = std::sqrt(std::max(0.0, (R[4*k] + 1.0) * 0.5));
        int a=(k+1)%3, b=(k+2)%3;
        ax[a] = R[k*3+a] / (2.0*ax[k]);
        ax[b] = R[k*3+b] / (2.0*ax[k]);
        double n = std::sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]) + 1e-30;
        w[0]=th*ax[0]/n; w[1]=th*ax[1]/n; w[2]=th*ax[2]/n; return;
    }
    double kf = th / (2.0*std::sin(th));
    w[0]=kf*vx; w[1]=kf*vy; w[2]=kf*vz;
}

// Node retraction: G⁺ = ( R·exp(dω), s·exp(dσ), t + dt ), d = [dt(3), dω(3), dσ(1)].
Sim3 retract(const Sim3& G, const double d[7]) {
    Sim3 out = G;
    double dR[9], w[3] = { d[3], d[4], d[5] };
    so3_exp(w, dR);
    // out.R = G.R * dR (row-major 3x3 multiply)
    for (int r=0;r<3;++r) for (int cc=0;cc<3;++cc)
        out.R[r*3+cc] = G.R[r*3+0]*dR[0*3+cc] + G.R[r*3+1]*dR[1*3+cc] + G.R[r*3+2]*dR[2*3+cc];
    out.s = G.s * std::exp(d[6]);
    out.t[0]=G.t[0]+d[0]; out.t[1]=G.t[1]+d[1]; out.t[2]=G.t[2]+d[2];
    return out;
}

// Dense SPD Cholesky solve A x = b (A row-major n×n, overwritten). Returns false
// if not positive-definite.
bool chol_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int i=0;i<n;++i) {
        for (int j=0;j<=i;++j) {
            double sum = A[i*n+j];
            for (int k=0;k<j;++k) sum -= A[i*n+k]*A[j*n+k];
            if (i==j) {
                if (sum <= 0) return false;
                A[i*n+j] = std::sqrt(sum);
            } else {
                A[i*n+j] = sum / A[j*n+j];
            }
        }
    }
    for (int i=0;i<n;++i) {                       // forward: L y = b
        double sum=b[i];
        for (int k=0;k<i;++k) sum -= A[i*n+k]*b[k];
        b[i]=sum/A[i*n+i];
    }
    for (int i=n-1;i>=0;--i) {                     // back: L^T x = y
        double sum=b[i];
        for (int k=i+1;k<n;++k) sum -= A[k*n+i]*b[k];
        b[i]=sum/A[i*n+i];
    }
    return true;
}

} // namespace

void pose_edge_residual(const Sim3& Gi, const Sim3& Gj, const Sim3& z, double r[7]) {
    // E = z^{-1} ∘ Gi^{-1} ∘ Gj ; zero iff Gi^{-1}∘Gj == z.
    Sim3 E = sim3_compose(sim3_inverse(z), sim3_compose(sim3_inverse(Gi), Gj));
    r[0]=E.t[0]; r[1]=E.t[1]; r[2]=E.t[2];
    double w[3]; so3_log(E.R, w);
    r[3]=w[0]; r[4]=w[1]; r[5]=w[2];
    r[6]=std::log(E.s > 0 ? E.s : 1e-30);
}

PoseGraphResult optimize_pose_graph(PoseGraph& g, const PoseGraphParams& p) {
    PoseGraphResult res;
    const int N = (int)g.nodes.size();
    if (N == 0) return res;

    // Gauge: default-fix node 0 if no explicit fix set.
    std::vector<char> fixed = g.fixed;
    if ((int)fixed.size() != N) { fixed.assign(N, 0); if (N>0) fixed[0]=1; }

    // Variable layout: 7 DOF per free node.
    std::vector<int> off(N, -1);
    int V = 0;
    for (int i=0;i<N;++i) if (!fixed[i]) { off[i]=V; V+=7; }
    res.free_dofs = V;
    if (V == 0) { res.ok = true; return res; }

    auto cost_of = [&](const std::vector<Sim3>& nodes) -> double {
        double c = 0; double r[7];
        for (const PoseEdge& e : g.edges) {
            pose_edge_residual(nodes[e.i], nodes[e.j], e.z, r);
            double s2=0; for (int k=0;k<7;++k) s2+=r[k]*r[k];
            c += 0.5 * e.info * s2;
        }
        return c;
    };

    res.cost0 = cost_of(g.nodes);
    double lambda = p.lambda0;
    double prev_cost = res.cost0;

    for (int it=0; it<p.max_iter; ++it) {
        // Assemble H (V×V) and rhs -g (b).
        std::vector<double> H((size_t)V*V, 0.0), b(V, 0.0);
        double r0[7];
        for (const PoseEdge& e : g.edges) {
            pose_edge_residual(g.nodes[e.i], g.nodes[e.j], e.z, r0);
            // numerical 7x7 jacobians wrt each free endpoint (central differences)
            struct EndJ { int node, base; double J[49]; bool used=false; } ends[2];
            ends[0].node=e.i; ends[0].base=off[e.i];
            ends[1].node=e.j; ends[1].base=off[e.j];
            for (int s=0;s<2;++s) {
                if (fixed[ends[s].node]) continue;
                ends[s].used=true;
                for (int k=0;k<7;++k) {
                    double dp[7]={0}, dm[7]={0}; dp[k]=p.step_eps; dm[k]=-p.step_eps;
                    Sim3 gp = retract(g.nodes[ends[s].node], dp);
                    Sim3 gm = retract(g.nodes[ends[s].node], dm);
                    double rp[7], rm[7];
                    if (s==0) { pose_edge_residual(gp, g.nodes[e.j], e.z, rp);
                               pose_edge_residual(gm, g.nodes[e.j], e.z, rm); }
                    else      { pose_edge_residual(g.nodes[e.i], gp, e.z, rp);
                               pose_edge_residual(g.nodes[e.i], gm, e.z, rm); }
                    for (int q=0;q<7;++q) ends[s].J[q*7+k] = (rp[q]-rm[q])/(2*p.step_eps);
                }
            }
            // Accumulate info * J_a^T J_b into H, info * J^T r0 into b.
            for (int a=0;a<2;++a) {
                if (!ends[a].used) continue;
                int ba = ends[a].base;
                for (int q=0;q<7;++q) for (int c=0;c<7;++c) {  // b_a += info * J_a^T r0
                    b[ba+q] -= e.info * ends[a].J[c*7+q] * r0[c];
                }
                for (int bkt=0;bkt<2;++bkt) {
                    if (!ends[bkt].used) continue;
                    int bb = ends[bkt].base;
                    for (int q=0;q<7;++q) for (int c=0;c<7;++c) {
                        double v=0; for (int m=0;m<7;++m) v += ends[a].J[m*7+q]*ends[bkt].J[m*7+c];
                        H[(size_t)(ba+q)*V + (bb+c)] += e.info * v;
                    }
                }
            }
        }

        // LM inner loop: damp, solve, trial, accept/reject.
        bool stepped = false;
        for (int tries=0; tries<12; ++tries) {
            std::vector<double> A(H), rhs(b);
            for (int i=0;i<V;++i) A[(size_t)i*V+i] += lambda * (H[(size_t)i*V+i] + 1e-12);
            if (!chol_solve(A, rhs, V)) { lambda*=4; continue; }
            std::vector<Sim3> trial = g.nodes;
            double mdx=0;
            for (int i=0;i<N;++i) if (off[i]>=0) {
                const double* d=&rhs[off[i]];
                trial[i]=retract(g.nodes[i], d);
                for (int k=0;k<7;++k) mdx=std::max(mdx,std::fabs(d[k]));
            }
            double c_new = cost_of(trial);
            if (c_new < prev_cost) {
                g.nodes.swap(trial);
                res.max_dx = mdx;
                lambda = std::max(lambda*0.5, 1e-12);
                double improve = (prev_cost - c_new);
                prev_cost = c_new;
                stepped = true;
                res.iters = it+1;
                if (improve <= p.tol * (res.cost0 + 1e-30)) { res.cost=prev_cost; res.ok=true; return res; }
                break;
            } else {
                lambda *= 4;
            }
        }
        res.cost = prev_cost;
        if (!stepped) { res.ok = true; return res; }   // couldn't improve => converged/stuck
    }
    res.cost = prev_cost;
    res.ok = true;
    return res;
}

} // namespace da
