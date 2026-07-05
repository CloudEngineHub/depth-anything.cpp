// Analytic unit test for Sim3 pose-graph optimization (src/posegraph.cpp, task C).
// Proves the properties the optimizer is KNOWN to have:
//   0. residual algebra: r(Gi,Gj, Gi^{-1}∘Gj) == 0 (zero at a consistent edge);
//   1. exact recovery: consistent measurements + a loop closure + a badly
//      perturbed init  ->  converges back to the known ground-truth poses (cost
//      ~0, per-node error ~0), with node 0 holding the gauge;
//   2. loop-closure drift reduction: NOISY sequential seams + one exact loop
//      closure -> ATE drops sharply vs the open-loop chained init;
//   3. scale drift: recovers known per-node Sim3 SCALE (the DOF SE3 graphs lack);
//   4. no-op: a consistent sequential-only chain is left untouched.
#include "posegraph.hpp"
#include "sim3.hpp"
#include "geom_metrics.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace da;

static unsigned long long gs = 0xda3c105e9e3779b9ULL;
static double urand(){ gs^=gs<<13; gs^=gs>>7; gs^=gs<<17; return (double)(gs>>11)/9007199254740992.0; }
static double sym(){ return urand()*2.0-1.0; }

static int fails=0;
static void check(bool ok,const char* m){ if(!ok){std::fprintf(stderr,"FAIL: %s\n",m);fails++;} else std::fprintf(stderr,"ok:   %s\n",m); }

static void axis_angle_R(const double k[3], double th, double R[9]){
    double n=std::sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2])+1e-30; double kx=k[0]/n,ky=k[1]/n,kz=k[2]/n;
    double c=std::cos(th),s=std::sin(th),v=1-c;
    R[0]=c+kx*kx*v; R[1]=kx*ky*v-kz*s; R[2]=kx*kz*v+ky*s;
    R[3]=ky*kx*v+kz*s; R[4]=c+ky*ky*v; R[5]=ky*kz*v-kx*s;
    R[6]=kz*kx*v-ky*s; R[7]=kz*ky*v+kx*s; R[8]=c+kz*kz*v;
}
static double rot_angle(const double A[9], const double B[9]){   // angle of A^T B
    double R[9]; for(int i=0;i<3;++i)for(int j=0;j<3;++j){double s=0;for(int k=0;k<3;++k)s+=A[k*3+i]*B[k*3+j];R[i*3+j]=s;}
    double c=(R[0]+R[4]+R[8]-1)*0.5; c=std::max(-1.0,std::min(1.0,c)); return std::acos(c);
}
static Sim3 rand_perturb(double tsc,double rsc,double ssc){
    Sim3 d; double k[3]={sym(),sym(),sym()}; axis_angle_R(k, rsc*sym(), d.R);
    d.t[0]=tsc*sym(); d.t[1]=tsc*sym(); d.t[2]=tsc*sym(); d.s=std::exp(ssc*sym()); return d;
}
// GT loop: nodes look outward from a circle (closed => node 0 area revisited at the end).
static std::vector<Sim3> gt_loop(int N, double radius, double scale_each=1.0){
    std::vector<Sim3> g(N); const double TWO_PI=6.283185307179586;
    for(int i=0;i<N;++i){ double a=TWO_PI*i/N; double k[3]={0,0,1};
        axis_angle_R(k, a, g[i].R);
        g[i].t[0]=radius*std::cos(a); g[i].t[1]=radius*std::sin(a); g[i].t[2]=0.2*std::sin(2*a);
        g[i].s=std::pow(scale_each, i); }
    return g;
}
static std::vector<std::array<double,3>> positions(const std::vector<Sim3>& g){
    std::vector<std::array<double,3>> p(g.size());
    for(size_t i=0;i<g.size();++i) p[i]={g[i].t[0],g[i].t[1],g[i].t[2]}; return p;
}
static double max_pose_err(const std::vector<Sim3>& a, const std::vector<Sim3>& b){
    double e=0; for(size_t i=0;i<a.size();++i){
        double dt=std::sqrt((a[i].t[0]-b[i].t[0])*(a[i].t[0]-b[i].t[0])+(a[i].t[1]-b[i].t[1])*(a[i].t[1]-b[i].t[1])+(a[i].t[2]-b[i].t[2])*(a[i].t[2]-b[i].t[2]));
        double dr=rot_angle(a[i].R,b[i].R); double ds=std::fabs(std::log(a[i].s/b[i].s));
        e=std::max(e, dt); e=std::max(e, dr); e=std::max(e, ds); }
    return e;
}
static PoseEdge seq_edge(const std::vector<Sim3>& g,int i,int j,const Sim3& noise){
    PoseEdge e; e.i=i; e.j=j; e.z=sim3_compose(sim3_compose(sim3_inverse(g[i]),g[j]), noise); return e;
}

int main(){
    // ---- 0. residual is zero at a consistent edge --------------------------
    {
        Sim3 Gi=rand_perturb(1.0,0.5,0.2), Gj=rand_perturb(1.0,0.5,0.2);
        Sim3 z=sim3_compose(sim3_inverse(Gi),Gj);
        double r[7]; pose_edge_residual(Gi,Gj,z,r);
        double nr=0; for(int k=0;k<7;++k) nr+=r[k]*r[k];
        check(std::sqrt(nr)<1e-9, "residual is ~0 at a consistent edge");
    }

    // ---- 1. exact recovery from a perturbed init (SE3, s=1) ----------------
    {
        const int N=12; auto G=gt_loop(N, 2.5);
        PoseGraph pg; pg.nodes=G;                       // start AT gt, then wreck the init
        Sim3 id;
        for(int i=0;i<N;++i) pg.edges.push_back(seq_edge(G,i,(i+1)%N,id)); // ring incl loop 11->0
        // Perturb every non-anchor node badly.
        for(int i=1;i<N;++i) pg.nodes[i]=sim3_compose(G[i], rand_perturb(0.6,0.4,0.15));
        double err0=max_pose_err(pg.nodes,G);
        PoseGraphResult r=optimize_pose_graph(pg);
        double err1=max_pose_err(pg.nodes,G);
        std::fprintf(stderr,"[exact] free=%d iters=%d cost %.3e->%.3e  pose_err %.4f->%.2e\n",
                     r.free_dofs,r.iters,r.cost0,r.cost,err0,err1);
        check(r.ok,"exact: optimizer ran");
        check(r.cost < 1e-12, "exact: cost driven to ~0");
        check(err1 < 1e-5, "exact: recovers ground-truth poses (<1e-5) from a bad init");
        check(max_pose_err({pg.nodes[0]},{G[0]})<1e-12, "exact: node 0 held the gauge (unchanged)");
    }

    // ---- 2. loop-closure reduces drift (SYSTEMATIC seam drift + strong loop) --
    // Real ghosting is driven by systematic accumulating error (monocular scale /
    // small rotation bias per seam), not iid noise; and a loop closure from dense
    // overlap ICP is a high-confidence constraint. Model both: each sequential
    // measurement carries a consistent small bias (that piles up into a large
    // open-loop end gap), and one accurate loop closure ties the ends.
    {
        const int N=16; auto G=gt_loop(N, 3.0);
        PoseGraph pg; pg.nodes.resize(N); pg.nodes[0]=G[0];
        Sim3 bias; { double kz[3]={0,0,1}; axis_angle_R(kz, 0.010, bias.R);   // +0.57deg/seam
                     bias.t[0]=0.010; bias.s=std::exp(0.006); }               // +1%/... scale creep
        for(int i=0;i<N-1;++i){
            Sim3 nz=sim3_compose(bias, rand_perturb(0.004,0.004,0.001));      // systematic + small iid
            pg.edges.push_back(seq_edge(G,i,i+1,nz));
            pg.nodes[i+1]=sim3_compose(pg.nodes[i], pg.edges.back().z);       // chain => accumulates drift
        }
        double ate_open=eval::ate_rmse(positions(pg.nodes),positions(G));
        PoseEdge lc; lc.i=N-1; lc.j=0; lc.z=sim3_compose(sim3_inverse(G[N-1]),G[0]);
        lc.loop=true; lc.info=25.0;                                           // dense-overlap ICP: precise
        pg.edges.push_back(lc);
        PoseGraphResult r=optimize_pose_graph(pg);
        double ate_closed=eval::ate_rmse(positions(pg.nodes),positions(G));
        std::fprintf(stderr,"[loop] iters=%d cost %.3e->%.3e  ATE %.4f->%.4f m  (%.0f%% cut)\n",
                     r.iters,r.cost0,r.cost,ate_open,ate_closed,100*(1-ate_closed/ate_open));
        check(ate_open>0 && ate_closed>=0, "loop: ATE computable");
        check(ate_closed < 0.4*ate_open, "loop: loop closure cuts drift ATE by >60%");
    }

    // ---- 3. recovers known SCALE drift (the Sim3-only DOF) -----------------
    {
        const int N=10; auto G=gt_loop(N, 2.0, /*scale_each=*/1.05);  // scale ramps 1.05^k
        PoseGraph pg; pg.nodes=G; Sim3 id;
        for(int i=0;i<N;++i) pg.edges.push_back(seq_edge(G,i,(i+1)%N,id));
        for(int i=1;i<N;++i){ pg.nodes[i]=G[i]; pg.nodes[i].s*=std::exp(0.3*sym()); } // wreck scales
        double serr0=0; for(int i=0;i<N;++i) serr0=std::max(serr0,std::fabs(std::log(pg.nodes[i].s/G[i].s)));
        optimize_pose_graph(pg);
        double serr1=0; for(int i=0;i<N;++i) serr1=std::max(serr1,std::fabs(std::log(pg.nodes[i].s/G[i].s)));
        std::fprintf(stderr,"[scale] max |log(s/s*)| %.4f->%.2e\n",serr0,serr1);
        check(serr1<1e-5, "scale: recovers known per-node scale drift");
    }

    // ---- 4. no-op on a consistent sequential-only chain --------------------
    {
        const int N=8; auto G=gt_loop(N, 2.0); PoseGraph pg; pg.nodes=G; Sim3 id;
        for(int i=0;i<N-1;++i) pg.edges.push_back(seq_edge(G,i,i+1,id));   // open chain, no loop
        std::vector<Sim3> before=pg.nodes;
        PoseGraphResult r=optimize_pose_graph(pg);
        check(r.cost0<1e-12 && max_pose_err(pg.nodes,before)<1e-9, "no-op: consistent chain left untouched");
    }

    std::fprintf(stderr,"%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
    return fails?1:0;
}
