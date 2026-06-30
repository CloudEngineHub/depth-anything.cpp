// Host-only gate for the streaming SEAM stitch: two overlapping windows whose
// local frames differ by a known Sim3 "drift", with DISJOINT per-pixel validity
// masks. Mirrors the correspondence+compose logic in src/stream.cpp (match by
// pixel identity, intersect both-valid, weighted-Umeyama, compose cumulative G)
// and asserts the recovered global transform realigns the overlap. No GGUF.
#include "sim3.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace da;

static unsigned long long g_seed = 0x9e3779b97f4a7c15ULL;
static double urand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)(g_seed >> 11) / 9007199254740992.0;
}
static double sym() { return urand() * 2.0 - 1.0; }

static void axis_angle_R(double ax, double ay, double az, double ang, double R[9]) {
    double n = std::sqrt(ax*ax + ay*ay + az*az); ax/=n; ay/=n; az/=n;
    double c = std::cos(ang), s = std::sin(ang), C = 1 - c;
    R[0]=c+ax*ax*C;    R[1]=ax*ay*C-az*s; R[2]=ax*az*C+ay*s;
    R[3]=ay*ax*C+az*s; R[4]=c+ay*ay*C;    R[5]=ay*az*C-ax*s;
    R[6]=az*ax*C-ay*s; R[7]=az*ay*C+ax*s; R[8]=c+az*az*C;
}
static Sim3 sim3_inv(const Sim3& T) {
    Sim3 I; I.s = 1.0 / T.s;
    I.R[0]=T.R[0]; I.R[1]=T.R[3]; I.R[2]=T.R[6];
    I.R[3]=T.R[1]; I.R[4]=T.R[4]; I.R[5]=T.R[7];
    I.R[6]=T.R[2]; I.R[7]=T.R[5]; I.R[8]=T.R[8];
    double rt[3] = { I.R[0]*T.t[0]+I.R[1]*T.t[1]+I.R[2]*T.t[2],
                     I.R[3]*T.t[0]+I.R[4]*T.t[1]+I.R[5]*T.t[2],
                     I.R[6]*T.t[0]+I.R[7]*T.t[1]+I.R[8]*T.t[2] };
    I.t[0]=-I.s*rt[0]; I.t[1]=-I.s*rt[1]; I.t[2]=-I.s*rt[2];
    return I;
}

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); fails++; }
    else      std::fprintf(stderr, "ok:   %s\n", msg);
}

// Run one seam: prev-window local->global = Gp, cur-window local->global = Sc.
// noise = per-coordinate jitter added to local points. Returns max global residual.
static double run_seam(const Sim3& Gp, const Sim3& Sc, double noise, double& srel_s) {
    const Sim3 Gp_inv = sim3_inv(Gp), Sc_inv = sim3_inv(Sc);
    const int Ngrid = 3000;
    std::vector<double> src, tgt, w;            // cur-local -> prev-local
    std::vector<double> keep_cur, keep_glob;    // both-valid cur-local & true global
    for (int i = 0; i < Ngrid; ++i) {
        double Pg[3] = { sym()*5, sym()*5, sym()*5 };
        double pl[3], cl[3];
        sim3_apply(Gp_inv, Pg, pl);             // prev-local
        sim3_apply(Sc_inv, Pg, cl);             // cur-local
        for (int k = 0; k < 3; ++k) { pl[k] += sym()*noise; cl[k] += sym()*noise; }
        bool pv = urand() < 0.7, cv = urand() < 0.7;   // independent disjoint masks
        if (!(pv && cv)) continue;
        double cw = 0.5 + 0.5*urand(), pw = 0.5 + 0.5*urand();
        src.push_back(cl[0]); src.push_back(cl[1]); src.push_back(cl[2]);
        tgt.push_back(pl[0]); tgt.push_back(pl[1]); tgt.push_back(pl[2]);
        w.push_back(cw < pw ? cw : pw);
        keep_cur.push_back(cl[0]); keep_cur.push_back(cl[1]); keep_cur.push_back(cl[2]);
        keep_glob.push_back(Pg[0]); keep_glob.push_back(Pg[1]); keep_glob.push_back(Pg[2]);
    }
    int M = (int)w.size();
    Sim3 S_rel; double rms = 0;
    bool ok = umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, S_rel, rms);
    if (!ok) { srel_s = 0; return 1e30; }
    srel_s = S_rel.s;
    Sim3 G_cur = sim3_compose(Gp, S_rel);       // cur-local -> global
    double maxd = 0;
    for (size_t i = 0; i < keep_cur.size()/3; ++i) {
        double cl[3] = { keep_cur[3*i+0], keep_cur[3*i+1], keep_cur[3*i+2] };
        double g[3]; sim3_apply(G_cur, cl, g);
        double dx = g[0]-keep_glob[3*i+0], dy = g[1]-keep_glob[3*i+1], dz = g[2]-keep_glob[3*i+2];
        double d = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d > maxd) maxd = d;
    }
    std::fprintf(stderr, "  seam: M=%d s_rel=%.4f rms=%.3g max_resid=%.3g\n", M, S_rel.s, rms, maxd);
    return maxd;
}

int main() {
    Sim3 Gp; Gp.s = 1.2; axis_angle_R(0.2,-0.4,0.6, 0.5, Gp.R); Gp.t[0]=1; Gp.t[1]=-2; Gp.t[2]=0.5;
    Sim3 Sc; Sc.s = 0.9; axis_angle_R(-0.5,0.3,0.7, -0.8, Sc.R); Sc.t[0]=-1.5; Sc.t[1]=0.4; Sc.t[2]=2.0;

    // Exact: disjoint masks, no noise -> overlap realigns to machine precision.
    double srel_s = 0;
    double m0 = run_seam(Gp, Sc, 0.0, srel_s);
    check(m0 < 1e-5, "seam exact: overlap realigns (max resid < 1e-5)");
    // S_rel scale should equal Sc.s / Gp.s (cur-local -> prev-local).
    check(std::fabs(srel_s - Sc.s/Gp.s) < 1e-4, "seam exact: relative scale = Sc.s/Gp.s");

    // Noisy: small jitter on local points -> still tightly aligned.
    double m1 = run_seam(Gp, Sc, 0.01, srel_s);
    check(m1 < 0.2, "seam noisy: overlap realigns within 0.2");

    std::fprintf(stderr, "%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
