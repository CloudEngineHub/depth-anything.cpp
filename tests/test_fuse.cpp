// Analytic unit test for surface fusion (src/fuse.cpp, task A). Uses SURFACE
// inputs (sheets / a sphere), not random volumes. Asserts the properties a
// normal-aware fusion is KNOWN to have: radius smoothing collapses a doubled
// sheet and thins a fuzzy one REGARDLESS of voxel-boundary alignment (the case
// that defeats naive voxel downsampling); the voxel stage is idempotent and
// reduces count; and neither erodes a curved surface (Chamfer to GT preserved).
#include "fuse.hpp"
#include "geom_metrics.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

using namespace da;

static unsigned long long g_seed = 0x51ed2701a3f9c7bbULL;
static double urand(){ g_seed^=g_seed<<13; g_seed^=g_seed>>7; g_seed^=g_seed<<17;
                       return (double)(g_seed>>11)/9007199254740992.0; }
static double sym(){ return urand()*2.0-1.0; }

static int fails = 0;
static void check(bool ok, const char* msg){
    if(!ok){ std::fprintf(stderr,"FAIL: %s\n",msg); fails++; } else std::fprintf(stderr,"ok:   %s\n",msg);
}
static std::vector<double> to_d(const std::vector<float>& v, int n){
    std::vector<double> o(3*n); for (int i=0;i<3*n;++i) o[i]=v[i]; return o;
}

int main() {
    // ---- 1. ADVERSARIAL doubled sheet (straddles z=0 boundary) collapses ----
    // Sheets at z = +/- delta/2, symmetric about a voxel boundary — the exact
    // case naive voxel downsampling fails. Radius smoothing (radius > delta)
    // merges them by metric distance, not grid cell.
    {
        const int NX = 4000; const double delta = 0.05;
        std::vector<float> xyz; std::vector<uint8_t> rgb; std::vector<float> rad;
        for (int i=0;i<NX;++i){
            double x=sym()*1.0, y=sym()*1.0;
            xyz.push_back(x); xyz.push_back(y); xyz.push_back(+delta/2);
            xyz.push_back(x); xyz.push_back(y); xyz.push_back(-delta/2);
        }
        const int Nin = (int)(xyz.size()/3);
        const double p0[3]={0,0,0}, nz[3]={0,0,1};
        double th_before = da::eval::thickness_to_plane(to_d(xyz,Nin).data(), Nin, p0, nz);
        FuseParams fp; fp.radius = 0.10f; fp.voxel = 0.f;   // smoothing only
        int Nout = fuse_voxel(xyz, rgb, rad, fp);
        double th_after = da::eval::thickness_to_plane(to_d(xyz,Nout).data(), Nout, p0, nz);
        check(th_before > 0.9*(delta/2), "double sheet starts thick (~delta/2)");
        check(th_after < 0.15*th_before, "radius smoothing collapses the (boundary-straddling) double");
        double maxz=0; for(int i=0;i<Nout;++i) maxz=std::max(maxz,std::fabs((double)xyz[3*i+2]));
        check(maxz < 0.01, "collapsed sheet lies on the true plane (z ~ 0)");
    }

    // ---- 2. fuzzy sheet centred on a boundary is thinned (denoise) ----
    {
        const int N=30000; const double sigma=0.03;
        std::vector<float> xyz; std::vector<uint8_t> rgb; std::vector<float> rad;
        for (int i=0;i<N;++i){
            xyz.push_back((float)(sym()*2)); xyz.push_back((float)(sym()*2));
            xyz.push_back((float)((sym()+sym()+sym())/std::sqrt(3.0)*sigma));
        }
        const int Nin=(int)(xyz.size()/3);
        const double p0[3]={0,0,0}, nz[3]={0,0,1};
        double th_before = da::eval::thickness_to_plane(to_d(xyz,Nin).data(), Nin, p0, nz);
        FuseParams fp; fp.radius=0.06f; fp.voxel=0.f;
        int Nout = fuse_voxel(xyz, rgb, rad, fp);
        double th_after = da::eval::thickness_to_plane(to_d(xyz,Nout).data(), Nout, p0, nz);
        check(th_after < 0.5*th_before, "radius smoothing thins a fuzzy sheet (denoise)");
    }

    // ---- 3. voxel stage (radius=0) reduces count and is idempotent ----
    {
        std::vector<float> xyz; std::vector<uint8_t> rgb; std::vector<float> rad;
        for (int i=0;i<3000;++i){ xyz.push_back(sym()*2); xyz.push_back(sym()*2); xyz.push_back(sym()*0.01); }
        const int Nin=(int)(xyz.size()/3);
        FuseParams fp; fp.radius=0.f; fp.voxel=0.05f;
        std::vector<float> a=xyz, ra; std::vector<uint8_t> rb;
        int n1 = fuse_voxel(a, rb, ra, fp);
        std::vector<float> b=a; std::vector<uint8_t> rb2; std::vector<float> ra2;
        int n2 = fuse_voxel(b, rb2, ra2, fp);
        bool same = (n1==n2);
        for (int i=0;same && i<3*n1;++i) if (std::fabs(a[i]-b[i])>1e-6f) same=false;
        check(n1 < Nin, "voxel stage reduces count");
        check(same, "voxel stage is idempotent (re-fusing changes nothing)");
    }

    // ---- 4. no erosion of a curved surface (sphere): Chamfer preserved ----
    {
        const int Ns=20000; std::vector<float> xyz; std::vector<uint8_t> rgb; std::vector<float> rad;
        std::vector<double> gt;
        for (int i=0;i<Ns;++i){
            double dx=sym(),dy=sym(),dz=sym(),n=std::sqrt(dx*dx+dy*dy+dz*dz)+1e-12; dx/=n;dy/=n;dz/=n;
            double r = 1.0 + sym()*0.01;
            xyz.push_back((float)(dx*r)); xyz.push_back((float)(dy*r)); xyz.push_back((float)(dz*r));
        }
        for (int i=0;i<4000;++i){ double dx=sym(),dy=sym(),dz=sym(),n=std::sqrt(dx*dx+dy*dy+dz*dz)+1e-12;
            gt.push_back(dx/n); gt.push_back(dy/n); gt.push_back(dz/n); }
        const int Nin=(int)(xyz.size()/3);
        auto sub=[](const std::vector<float>& v,int s){ std::vector<double> o;
            for(size_t i=0;i+2<v.size();i+=3*s){o.push_back(v[i]);o.push_back(v[i+1]);o.push_back(v[i+2]);} return o; };
        std::vector<double> in_s=sub(xyz,5);
        double ch_before=da::eval::chamfer(in_s.data(),(int)(in_s.size()/3),gt.data(),(int)(gt.size()/3)).a_to_b;
        FuseParams fp; fp.radius=0.03f; fp.voxel=0.03f;
        int Nout=fuse_voxel(xyz,rgb,rad,fp);
        std::vector<double> out_s=sub(xyz,1);
        double ch_after=da::eval::chamfer(out_s.data(),(int)(out_s.size()/3),gt.data(),(int)(gt.size()/3)).a_to_b;
        check(Nout < Nin, "fusion reduces sphere point count");
        check(ch_after <= ch_before + 0.005, "fusion does NOT erode the curved surface (Chamfer preserved)");
    }

    std::fprintf(stderr, "%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
    return fails ? 1 : 0;
}
