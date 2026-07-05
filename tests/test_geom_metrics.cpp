// Self-test for the geometry metrics (tests/geom_metrics.hpp). The metrics are
// the measuring instruments for the A/B/C validation suite, so they must be
// correct BEFORE we trust any A/B/C number. Each check uses inputs with a known
// analytic answer. No GGUF / GPU.
#include "geom_metrics.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>

using namespace da::eval;

static unsigned long long g_seed = 0x243f6a8885a308d3ULL;
static double urand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)(g_seed >> 11) / 9007199254740992.0;
}
static double sym() { return urand() * 2.0 - 1.0; }

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); fails++; }
    else      std::fprintf(stderr, "ok:   %s\n", msg);
}

int main() {
    // A random cloud reused across checks.
    const int N = 1500;
    std::vector<double> A(3*N);
    for (int i = 0; i < 3*N; ++i) A[i] = sym() * 5.0;

    // ---- Chamfer ----
    {
        ChamferResult c0 = chamfer(A.data(), N, A.data(), N);
        check(c0.chamfer < 1e-12, "chamfer(A,A) == 0");

        // Rigid shift on a spacing-1 lattice by d < 0.5: the self-correspondence
        // is provably the nearest neighbour (any other lattice point is >= 1-d
        // away), so Chamfer == d exactly. (A random cloud would give < d because
        // a non-corresponding point can be closer than the shift.)
        const int G = 10; const double d = 0.3;
        std::vector<double> L(3*G*G*G), Ls(3*G*G*G);
        for (int a=0;a<G;++a) for (int b=0;b<G;++b) for (int cc=0;cc<G;++cc) {
            int i = (a*G+b)*G+cc;
            L[3*i+0]=a; L[3*i+1]=b; L[3*i+2]=cc;
            Ls[3*i+0]=a+d; Ls[3*i+1]=b; Ls[3*i+2]=cc;
        }
        ChamferResult c1 = chamfer(L.data(), G*G*G, Ls.data(), G*G*G);
        check(std::fabs(c1.chamfer - d) < 1e-9, "chamfer(lattice, lattice+d) == d");
    }

    // ---- F-score ----
    {
        FScore f0 = fscore(A.data(), N, A.data(), N, 0.1);
        check(std::fabs(f0.f - 1.0) < 1e-12, "fscore(A,A,tau) == 1");

        // Shift far past tau -> no matches either way -> f == 0.
        std::vector<double> B(A);
        for (int i = 0; i < N; ++i) B[3*i+0] += 10.0;
        FScore f1 = fscore(A.data(), N, B.data(), N, 0.1);
        check(f1.precision < 1e-12 && f1.recall < 1e-12 && f1.f < 1e-12,
              "fscore separated-by-10 @tau=0.1 == 0");
    }

    // ---- surface thickness ----
    {
        const double p0[3] = {0,0,0}, nz[3] = {0,0,1};
        // Single clean sheet at z=0 with gaussian-ish jitter sigma along z.
        const double sigma = 0.02;
        std::vector<double> sheet(3*N);
        for (int i = 0; i < N; ++i) {
            sheet[3*i+0] = sym()*3; sheet[3*i+1] = sym()*3;
            // sum of 3 uniforms ~ approx normal, scaled to ~sigma std.
            sheet[3*i+2] = (sym()+sym()+sym())/std::sqrt(3.0) * sigma;
        }
        double th1 = thickness_to_plane(sheet.data(), N, p0, nz);
        check(th1 < 3.0*sigma && th1 > 0.3*sigma, "thickness single sheet ~ sigma");

        // Doubled sheet separated by delta -> thickness ~ delta/2 (std of {0,delta}).
        const double delta = 0.10;
        std::vector<double> dbl(3*2*N);
        for (int i = 0; i < N; ++i) {
            double x = sym()*3, y = sym()*3;
            dbl[3*i+0]=x; dbl[3*i+1]=y; dbl[3*i+2]=0.0;
            dbl[3*(N+i)+0]=x; dbl[3*(N+i)+1]=y; dbl[3*(N+i)+2]=delta;
        }
        double th2 = thickness_to_plane(dbl.data(), 2*N, p0, nz);
        check(std::fabs(th2 - delta/2.0) < 1e-9, "thickness doubled sheet == delta/2");
    }

    // ---- duplicate ratio ----
    {
        // Exact duplicates: every point has a coincident twin -> ratio 1.
        std::vector<double> dbl(3*2*N);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < 3; ++k) { dbl[3*i+k]=A[3*i+k]; dbl[3*(N+i)+k]=A[3*i+k]; }
        double dr = duplicate_ratio(dbl.data(), 2*N, 1e-6);
        check(std::fabs(dr - 1.0) < 1e-12, "duplicate_ratio(doubled) == 1");

        // A well-separated lattice with spacing 1: eps=0.4 finds no neighbour.
        const int G = 8; std::vector<double> lat(3*G*G);
        for (int a = 0; a < G; ++a) for (int b = 0; b < G; ++b) {
            int i = a*G+b; lat[3*i+0]=a; lat[3*i+1]=b; lat[3*i+2]=0;
        }
        double dr2 = duplicate_ratio(lat.data(), G*G, 0.4);
        check(dr2 < 1e-12, "duplicate_ratio(spacing-1 lattice, eps=0.4) == 0");
    }

    // ---- ATE / RPE ----
    {
        // A GT trajectory (a curve, not a line, so Umeyama alignment is well posed).
        const int T = 40;
        std::vector<std::array<double,3>> gt(T), est(T);
        for (int i = 0; i < T; ++i) {
            double t = i * 0.2;
            gt[i] = { std::cos(t)*2.0, std::sin(t)*2.0, t*0.1 };
        }

        // est == gt -> ATE 0. (>=0 guard: a degenerate solve returns -1, which must
        // NOT masquerade as "== 0".)
        est = gt;
        double a_id = ate_rmse(est, gt);
        check(a_id >= 0 && a_id < 1e-9, "ATE(gt,gt) == 0");
        check(rpe_trans_rmse(est, gt) >= 0 && rpe_trans_rmse(est, gt) < 1e-9, "RPE(gt,gt) == 0");

        // est = a global Sim3 of gt -> ATE ~0 (frame-invariant by construction).
        da::Sim3 Gg; Gg.s = 1.7;
        { double ax=0.3,ay=-0.6,az=0.5,n=std::sqrt(ax*ax+ay*ay+az*az); ax/=n;ay/=n;az/=n;
          double ang=0.8,c=std::cos(ang),s=std::sin(ang),C=1-c;
          Gg.R[0]=c+ax*ax*C; Gg.R[1]=ax*ay*C-az*s; Gg.R[2]=ax*az*C+ay*s;
          Gg.R[3]=ay*ax*C+az*s; Gg.R[4]=c+ay*ay*C; Gg.R[5]=ay*az*C-ax*s;
          Gg.R[6]=az*ax*C-ay*s; Gg.R[7]=az*ay*C+ax*s; Gg.R[8]=c+az*az*C; }
        Gg.t[0]=1; Gg.t[1]=-2; Gg.t[2]=0.5;
        for (int i = 0; i < T; ++i) {
            double p[3]={gt[i][0],gt[i][1],gt[i][2]}, q[3]; da::sim3_apply(Gg,p,q);
            est[i] = {q[0],q[1],q[2]};
        }
        double a_sim = ate_rmse(est, gt);
        check(a_sim >= 0 && a_sim < 1e-6, "ATE invariant to global Sim3");

        // Per-pose drift of known magnitude m -> ATE grows to ~m (not exactly, since
        // alignment absorbs a little), but must be within [0.3m, 1.5m].
        const double m = 0.15;
        for (int i = 0; i < T; ++i) {
            double dx=sym(),dy=sym(),dz=sym(),nn=std::sqrt(dx*dx+dy*dy+dz*dz)+1e-12;
            est[i] = { gt[i][0]+m*dx/nn, gt[i][1]+m*dy/nn, gt[i][2]+m*dz/nn };
        }
        double ate = ate_rmse(est, gt);
        check(ate > 0.3*m && ate < 1.5*m, "ATE(gt + drift m) ~ m");
    }

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
