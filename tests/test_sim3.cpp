// Host-only gate for the weighted-Umeyama Sim3 solver (src/sim3.cpp): exact
// recovery, the reflection trap (proper rotation only), Huber robustness to gross
// outliers, and compose associativity. No GGUF needed.
#include "sim3.hpp"
#include "linalg.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace da;

static unsigned long long g_seed = 88172645463325252ULL;
static double urand() {  // xorshift64 -> [0,1)
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)(g_seed >> 11) / 9007199254740992.0;
}
static double sym() { return urand() * 2.0 - 1.0; }

static void axis_angle_R(double ax, double ay, double az, double ang, double R[9]) {
    double n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n < 1e-12) { R[0]=R[4]=R[8]=1; R[1]=R[2]=R[3]=R[5]=R[6]=R[7]=0; return; }
    ax/=n; ay/=n; az/=n;
    double c = std::cos(ang), s = std::sin(ang), C = 1 - c;
    R[0]=c+ax*ax*C;    R[1]=ax*ay*C-az*s; R[2]=ax*az*C+ay*s;
    R[3]=ay*ax*C+az*s; R[4]=c+ay*ay*C;    R[5]=ay*az*C-ax*s;
    R[6]=az*ax*C-ay*s; R[7]=az*ay*C+ax*s; R[8]=c+az*az*C;
}
static double rdiff(const double A[9], const double B[9]) {
    double s = 0; for (int i = 0; i < 9; ++i) { double d = A[i]-B[i]; s += d*d; } return std::sqrt(s);
}
static double tdiff(const double a[3], const double b[3]) {
    double s = 0; for (int i = 0; i < 3; ++i) { double d = a[i]-b[i]; s += d*d; } return std::sqrt(s);
}

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); fails++; }
    else      std::fprintf(stderr, "ok:   %s\n", msg);
}

// Build M correspondences tgt = GT(src) for random src in [-5,5]^3.
static void make(int M, const Sim3& GT, std::vector<double>& src, std::vector<double>& tgt) {
    src.resize(3*M); tgt.resize(3*M);
    for (int i = 0; i < M; ++i) {
        double p[3] = { sym()*5, sym()*5, sym()*5 };
        double q[3]; sim3_apply(GT, p, q);
        for (int k = 0; k < 3; ++k) { src[3*i+k] = p[k]; tgt[3*i+k] = q[k]; }
    }
}

int main() {
    // (a) exact recovery.
    {
        Sim3 GT; GT.s = 1.7; axis_angle_R(0.3,-0.7,0.5, 0.9, GT.R);
        GT.t[0]=2.0; GT.t[1]=-1.0; GT.t[2]=0.5;
        int M = 300; std::vector<double> src, tgt; make(M, GT, src, tgt);
        std::vector<double> w(M, 1.0);
        Sim3 out; double rms = 0;
        bool ok = umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, out, rms);
        check(ok, "exact: solver returned true");
        check(rms < 1e-6, "exact: rms ~ 0");
        check(std::fabs(out.s - GT.s) < 1e-6, "exact: scale");
        check(rdiff(out.R, GT.R) < 1e-6, "exact: rotation");
        check(tdiff(out.t, GT.t) < 1e-6, "exact: translation");
    }
    // (b) reflection trap: R = diag(-1,-1,1) (180 deg about z) must stay proper.
    {
        Sim3 GT; GT.s = 1.0;
        GT.R[0]=-1; GT.R[4]=-1; GT.R[8]=1;
        int M = 250; std::vector<double> src, tgt; make(M, GT, src, tgt);
        std::vector<double> w(M, 1.0);
        Sim3 out; double rms = 0;
        bool ok = umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, out, rms);
        check(ok, "reflect: solver returned true");
        check(std::fabs(linalg::det3(out.R) - 1.0) < 1e-6, "reflect: det(R) == +1 (proper)");
        check(rdiff(out.R, GT.R) < 1e-5, "reflect: recovered diag(-1,-1,1)");
    }
    // (c) Huber robustness: 20% gross outliers, all weights 1.
    {
        Sim3 GT; GT.s = 1.3; axis_angle_R(-0.2,0.5,0.8, 1.4, GT.R);
        GT.t[0]=-3.0; GT.t[1]=0.7; GT.t[2]=4.0;
        int M = 400; std::vector<double> src, tgt; make(M, GT, src, tgt);
        std::vector<double> w(M, 1.0);
        for (int i = 0; i < M; i += 5) {            // ~20% corrupted
            tgt[3*i+0] += sym()*30; tgt[3*i+1] += sym()*30; tgt[3*i+2] += sym()*30;
        }
        Sim3 out; double rms = 0;
        bool ok = umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, out, rms);
        check(ok, "huber: solver returned true");
        check(std::fabs(out.s - GT.s) < 5e-2, "huber: scale within 5e-2");
        check(rdiff(out.R, GT.R) < 5e-2, "huber: rotation within 5e-2");
        check(tdiff(out.t, GT.t) < 0.25, "huber: translation within 0.25");
    }
    // (d) compose associativity: (A∘B)(x) == A(B(x)).
    {
        Sim3 A; A.s = 0.8; axis_angle_R(0.1,0.2,0.9, 0.6, A.R); A.t[0]=1; A.t[1]=2; A.t[2]=-1;
        Sim3 B; B.s = 1.5; axis_angle_R(0.7,-0.3,0.2, -1.1, B.R); B.t[0]=-2; B.t[1]=0.5; B.t[2]=3;
        Sim3 C = sim3_compose(A, B);
        double maxd = 0;
        for (int i = 0; i < 50; ++i) {
            double x[3] = { sym()*4, sym()*4, sym()*4 };
            double bx[3], abx[3], cx[3];
            sim3_apply(B, x, bx); sim3_apply(A, bx, abx); sim3_apply(C, x, cx);
            maxd = std::max(maxd, tdiff(abx, cx));
        }
        check(maxd < 1e-9, "compose: (A.B)(x) == A(B(x))");
    }
    // (e) degenerate: too few points -> false.
    {
        Sim3 GT; int M = 10; std::vector<double> src, tgt; make(M, GT, src, tgt);
        std::vector<double> w(M, 1.0);
        Sim3 out; double rms = 0;
        bool ok = umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, out, rms);
        check(!ok, "degenerate: M<min_pts returns false");
    }

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
