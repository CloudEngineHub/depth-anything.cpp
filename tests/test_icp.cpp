// Analytic unit test for point-to-plane ICP (src/icp.cpp, task B) on SURFACE
// inputs. Proves the properties the method is KNOWN to have:
//   1. curvature-rich overlap (corner + sphere): recovers a KNOWN rigid Sim3 to
//      <0.1 deg / sub-mm, full rank 6, and the point-to-plane residual decreases
//      monotonically to ~0;
//   2. single flat wall: the tangential slide + in-plane spin are UNOBSERVABLE
//      (aperture problem) -> rank 3, and ICP corrects ONLY the normal-direction
//      error while leaving the tangential offset at the prior (it does NOT drift
//      or hallucinate a fix);
//   3. convergence basin: recovers from progressively larger initial rotation;
//   4. no-op when already aligned;
//   5. estimate_normals agrees with the analytic surface normals.
#include "icp.hpp"
#include "sim3.hpp"
#include "geom_metrics.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace da;

static unsigned long long g_seed = 0xa3f9c7bb51ed2701ULL;
static double urand(){ g_seed^=g_seed<<13; g_seed^=g_seed>>7; g_seed^=g_seed<<17;
                       return (double)(g_seed>>11)/9007199254740992.0; }
static double sym(){ return urand()*2.0-1.0; }

static int fails = 0;
static void check(bool ok, const char* msg){
    if(!ok){ std::fprintf(stderr,"FAIL: %s\n",msg); fails++; } else std::fprintf(stderr,"ok:   %s\n",msg);
}

// axis-angle (unit axis, radians) -> row-major R via Rodrigues.
static void axis_angle_R(const double k[3], double th, double R[9]){
    double c=std::cos(th), s=std::sin(th), v=1-c, kx=k[0],ky=k[1],kz=k[2];
    R[0]=c+kx*kx*v;    R[1]=kx*ky*v-kz*s; R[2]=kx*kz*v+ky*s;
    R[3]=ky*kx*v+kz*s; R[4]=c+ky*ky*v;    R[5]=ky*kz*v-kx*s;
    R[6]=kz*kx*v-ky*s; R[7]=kz*ky*v+kx*s; R[8]=c+kz*kz*v;
}
static double angle_of_R(const double R[9]){
    double tr=R[0]+R[4]+R[8];
    double c=(tr-1.0)/2.0; c=std::max(-1.0,std::min(1.0,c));
    return std::acos(c);
}
static void apply_cloud(const Sim3& T, const std::vector<double>& in, std::vector<double>& out){
    out.resize(in.size());
    const int n=(int)(in.size()/3);
    for(int i=0;i<n;++i) sim3_apply(T,&in[3*i],&out[3*i]);
}

// ---- surface generators (dense, so normals + NN are well-posed) ----------
static void add_plane(std::vector<double>& v, double ox,double oy,double oz,
                      double ax,double ay,double az, double bx,double by,double bz,
                      double la,double lb,double step){
    for(double a=-la;a<=la+1e-9;a+=step) for(double b=-lb;b<=lb+1e-9;b+=step){
        v.push_back(ox+ax*a+bx*b); v.push_back(oy+ay*a+by*b); v.push_back(oz+az*a+bz*b);
    }
}
static void add_sphere(std::vector<double>& v,double cx,double cy,double cz,double r,int n){
    for(int i=0;i<n;++i){ double dx=sym(),dy=sym(),dz=sym();
        double nn=std::sqrt(dx*dx+dy*dy+dz*dz)+1e-12; dx/=nn;dy/=nn;dz/=nn;
        v.push_back(cx+dx*r); v.push_back(cy+dy*r); v.push_back(cz+dz*r); }
}
// corner (two orthogonal planes) + a sphere => normals span R^3 (full rank).
static std::vector<double> corner_scene(){
    std::vector<double> v;
    add_plane(v, 0.5,0,0,  1,0,0, 0,1,0, 0.5,0.5, 0.02);   // floor z=0
    add_plane(v, 0,0,0.5,  0,0,1, 0,1,0, 0.5,0.5, 0.02);   // wall  x=0
    add_sphere(v, 0.5,0.0,0.7, 0.22, 2500);                // curved patch
    return v;
}

int main(){
    IcpParams P; P.max_corr_dist=0.10; P.normal_radius=0.06; P.huber_delta=0.05;
    P.max_iter=40; P.rank_tol=1e-3;

    // ================= 1. full-rank recovery of a known rigid Sim3 =================
    {
        std::vector<double> tgt = corner_scene();
        std::vector<double> nrm; estimate_normals(tgt, P.normal_radius, nullptr, nrm);

        // Known perturbation G (rigid): 2 deg about a tilted axis + 4 cm translation.
        double ax[3]={0.3,0.8,0.5}; double an=std::sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]);
        ax[0]/=an;ax[1]/=an;ax[2]/=an;
        Sim3 G; axis_angle_R(ax, 2.0*M_PI/180.0, G.R); G.t[0]=0.04; G.t[1]=-0.03; G.t[2]=0.02;
        std::vector<double> src; apply_cloud(G, tgt, src);       // src = G(tgt) => T should be G^-1

        IcpResult r = icp_point_to_plane(src, tgt, nrm, Sim3(), P);

        // T ∘ G should be identity (T recovers G^-1).
        Sim3 comp = sim3_compose(r.T, G);
        double ang = angle_of_R(comp.R) * 180.0/M_PI;
        double tnorm = std::sqrt(comp.t[0]*comp.t[0]+comp.t[1]*comp.t[1]+comp.t[2]*comp.t[2]);
        std::fprintf(stderr,"[full] rank=%d iters=%d rms %.5f->%.5f  resid_angle=%.4fdeg resid_t=%.5fm\n",
                     r.rank, r.iters, r.rms_before, r.rms_after, ang, tnorm);
        check(r.ok, "full-rank: ICP ran");
        check(r.rank==6, "full-rank: normal system is full rank (6)");
        check(ang < 0.1, "full-rank: recovers known rotation to < 0.1 deg");
        check(tnorm < 1e-3, "full-rank: recovers known translation to < 1 mm");
        check(r.rms_after < r.rms_before, "full-rank: residual decreased");
        check(r.rms_after < 1e-3, "full-rank: converged to ~0 point-to-plane residual");
        bool mono=true; for(size_t i=1;i<r.rms_iters.size();++i)
            if(r.rms_iters[i] > r.rms_iters[i-1]+1e-9) mono=false;
        check(mono, "full-rank: point-to-plane residual is monotonically non-increasing");
    }

    // ================= 2. flat wall: aperture problem, fall back to prior =========
    {
        std::vector<double> tgt;
        add_plane(tgt, 0,0,0, 1,0,0, 0,1,0, 0.6,0.6, 0.015);   // single plane z=0, normal +z
        std::vector<double> nrm; estimate_normals(tgt, 0.06, nullptr, nrm);

        // Perturb with a NORMAL part (z translate + small tilt) AND a TANGENTIAL
        // slide (x). Only the normal part is observable.
        double kx[3]={1,0,0};                                   // tilt about x
        Sim3 G; axis_angle_R(kx, 1.0*M_PI/180.0, G.R);
        G.t[0]=0.05;   // tangential (unobservable)
        G.t[2]=0.03;   // along normal (observable)
        std::vector<double> src; apply_cloud(G, tgt, src);

        IcpResult r = icp_point_to_plane(src, tgt, nrm, Sim3(), P);

        // Point-to-plane residual must collapse (normal error fixed)...
        // ...but the x tangential offset must SURVIVE (not hallucinated away).
        std::vector<double> moved; apply_cloud(r.T, src, moved);
        int n=(int)(moved.size()/3);
        double mean_z=0, mean_x=0; for(int i=0;i<n;++i){ mean_x+=moved[3*i]; mean_z+=moved[3*i+2]; }
        mean_x/=n; mean_z/=n;
        // tangential translation the solver applied (should be ~0 => stayed at prior)
        double applied_tx = r.T.t[0];
        std::fprintf(stderr,"[wall] rank=%d iters=%d rms %.5f->%.5f  mean_z=%.5f mean_x=%.5f applied_tx=%.5f\n",
                     r.rank, r.iters, r.rms_before, r.rms_after, mean_z, mean_x, applied_tx);
        check(r.rank==3, "wall: exactly 3 observable DOF (rank 3 = aperture problem detected)");
        check(r.rms_after < 2e-3, "wall: normal-direction error corrected (residual ~0)");
        check(std::fabs(mean_z) < 2e-3, "wall: transformed source lands ON the plane (z~0)");
        check(std::fabs(mean_x - 0.05) < 5e-3, "wall: tangential slide SURVIVES (not hallucinated)");
        check(std::fabs(applied_tx) < 5e-3, "wall: solver applied ~no tangential correction (fell back to prior)");
        check(std::isfinite(r.rms_after) && r.rms_after<=r.rms_before, "wall: did NOT drift/diverge");
    }

    // ================= 3. convergence basin (rotation) ============================
    {
        std::vector<double> tgt = corner_scene();
        std::vector<double> nrm; estimate_normals(tgt, P.normal_radius, nullptr, nrm);
        double kax[3]={0.2,0.3,0.93}; double kn=std::sqrt(kax[0]*kax[0]+kax[1]*kax[1]+kax[2]*kax[2]);
        kax[0]/=kn;kax[1]/=kn;kax[2]/=kn;
        int max_ok_deg=0;
        for(int deg=2; deg<=30; deg+=2){
            Sim3 G; axis_angle_R(kax, deg*M_PI/180.0, G.R); G.t[0]=0.02; G.t[1]=0.01;
            std::vector<double> src; apply_cloud(G, tgt, src);
            IcpResult r = icp_point_to_plane(src, tgt, nrm, Sim3(), P);
            Sim3 comp = sim3_compose(r.T, G);
            double ang = angle_of_R(comp.R)*180.0/M_PI;
            if(ang < 0.5) max_ok_deg = deg; else break;
        }
        std::fprintf(stderr,"[basin] recovers initial rotation error up to ~%d deg\n", max_ok_deg);
        check(max_ok_deg >= 14, "basin: converges from >=14 deg initial rotation error");
    }

    // ================= 4. no-op when already aligned ==============================
    {
        std::vector<double> tgt = corner_scene();
        std::vector<double> nrm; estimate_normals(tgt, P.normal_radius, nullptr, nrm);
        IcpResult r = icp_point_to_plane(tgt, tgt, nrm, Sim3(), P);
        double tnorm=std::sqrt(r.T.t[0]*r.T.t[0]+r.T.t[1]*r.T.t[1]+r.T.t[2]*r.T.t[2]);
        double ang=angle_of_R(r.T.R)*180.0/M_PI;
        std::fprintf(stderr,"[noop] ang=%.5fdeg t=%.6fm rms=%.6f\n", ang, tnorm, r.rms_after);
        check(ang<1e-2 && tnorm<1e-4, "no-op: already-aligned input stays put");
    }

    // ================= 5. estimate_normals vs analytic normals ====================
    {
        std::vector<double> pl;
        add_plane(pl, 0,0,0, 1,0,0, 0,1,0, 0.5,0.5, 0.02);   // z=0 -> normal (0,0,1)
        std::vector<double> nrm; estimate_normals(pl, 0.06, nullptr, nrm);
        int n=(int)(pl.size()/3), good=0, tot=0;
        for(int i=0;i<n;++i){ double nz=std::fabs(nrm[3*i+2]);
            if(nrm[3*i]==0&&nrm[3*i+1]==0&&nrm[3*i+2]==0) continue; ++tot; if(nz>0.99) ++good; }
        check(tot>0 && (double)good/tot > 0.95, "normals: plane normals align with (0,0,1)");

        std::vector<double> sp; add_sphere(sp, 0,0,0, 1.0, 3000);
        std::vector<double> sn; estimate_normals(sp, 0.12, nullptr, sn);
        int sgood=0, stot=0;
        for(int i=0;i<(int)(sp.size()/3);++i){
            if(sn[3*i]==0&&sn[3*i+1]==0&&sn[3*i+2]==0) continue; ++stot;
            double dotr=std::fabs(sn[3*i]*sp[3*i]+sn[3*i+1]*sp[3*i+1]+sn[3*i+2]*sp[3*i+2]); // |n·r̂|, r=1
            if(dotr>0.98) ++sgood; }
        check(stot>0 && (double)sgood/stot > 0.90, "normals: sphere normals are ~radial");
    }

    std::fprintf(stderr, "%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
    return fails ? 1 : 0;
}
