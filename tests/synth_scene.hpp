#pragma once
// Surface-concentrated synthetic ground truth for the A/B/C alignment validation.
//
// Real fused clouds are 2-manifolds (points on walls/floor/objects), NOT volume
// samples — and ghosting is a surface-doubling artifact. So we DON'T sprinkle
// random points in a box: we build an analytic room (finite planes + a sphere),
// raycast a pinhole camera against it, and emit per-view depth + normals + conf.
// Back-projecting that depth yields a surface-concentrated cloud with real
// view-dependent density, occlusion, and along-ray (=> along-normal) noise —
// exactly the structure the DA3 pipeline produces.
//
// Conventions match src/stream.cpp: OpenCV camera (x-right, y-down, z-forward);
// intr = K row-major; ext = world->cam [R|t] 3x4 row-major; depth = camera-frame
// Z. back_project_world() reproduces the pipeline's exact math so these views can
// drive stream.cpp verbatim (Task #4) and so tests share one convention.
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace da { namespace synth {

// ---- tiny deterministic RNG (per-call seed; no global state) ------------
struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    double u01() { s ^= s<<13; s ^= s>>7; s ^= s<<17; return (double)(s>>11)/9007199254740992.0; }
    double sym() { return u01()*2.0 - 1.0; }
    // ~N(0,1) via central limit of 6 uniforms (plenty for sensor-noise modeling).
    double gauss() { double t=0; for (int i=0;i<6;++i) t+=u01(); return (t-3.0)/std::sqrt(0.5); }
};

// ---- vector helpers ------------------------------------------------------
struct V3 { double x=0,y=0,z=0; };
inline V3 operator+(V3 a, V3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline V3 operator-(V3 a, V3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline V3 operator*(V3 a, double s){ return {a.x*s,a.y*s,a.z*s}; }
inline double dot(V3 a, V3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
inline V3 cross(V3 a, V3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
inline double norm(V3 a){ return std::sqrt(dot(a,a)); }
inline V3 normalized(V3 a){ double n=norm(a); return n>0 ? a*(1.0/n) : a; }

// ---- scene primitives ----------------------------------------------------
// Finite rectangular plane: center c, unit normal n, in-plane axes u,v with
// half-extents hu,hv. A hit counts only within the rectangle (=> bounded walls).
struct Plane { V3 c, n, u, v; double hu, hv; };
struct Sphere { V3 c; double r; };

struct Hit { bool ok=false; double t=1e300; V3 nrm; int prim=-1; };

struct Scene {
    std::vector<Plane> planes;
    std::vector<Sphere> spheres;

    // Nearest positive-t intersection of ray O + t*D (D need not be unit).
    Hit raycast(V3 O, V3 D) const {
        Hit h;
        for (size_t i=0;i<planes.size();++i) {
            const Plane& p = planes[i];
            double dn = dot(p.n, D);
            if (std::fabs(dn) < 1e-12) continue;
            double t = dot(p.n, p.c - O) / dn;
            if (t <= 1e-9 || t >= h.t) continue;
            V3 P = O + D*t, d = P - p.c;
            if (std::fabs(dot(d,p.u)) > p.hu || std::fabs(dot(d,p.v)) > p.hv) continue;
            h.ok=true; h.t=t; h.prim=(int)i;
            h.nrm = (dn < 0) ? p.n : p.n*-1.0;   // face the camera
        }
        for (size_t i=0;i<spheres.size();++i) {
            const Sphere& s = spheres[i];
            V3 oc = O - s.c;
            double a=dot(D,D), b=2*dot(oc,D), c=dot(oc,oc)-s.r*s.r;
            double disc=b*b-4*a*c; if (disc<0) continue;
            double sq=std::sqrt(disc); double t=(-b-sq)/(2*a);
            if (t<=1e-9) t=(-b+sq)/(2*a);
            if (t<=1e-9 || t>=h.t) continue;
            V3 P=O+D*t; h.ok=true; h.t=t; h.prim=1000+(int)i;
            h.nrm = normalized(P - s.c);
        }
        return h;
    }
};

// Distance from a world point to the nearest scene surface (planes clamped to
// their finite rectangle; spheres by radial offset). Used by tests to prove a
// back-projected cloud actually lies ON the surfaces (thickness ~ 0), not in a
// volume. Returns the min over all primitives.
inline double surface_distance(const Scene& scene, V3 P) {
    double best = 1e300;
    for (const Plane& p : scene.planes) {
        V3 d = P - p.c;
        double du = dot(d,p.u), dv = dot(d,p.v), dn = dot(d,p.n);
        double cu = std::max(-p.hu, std::min(p.hu, du));   // clamp onto rectangle
        double cv = std::max(-p.hv, std::min(p.hv, dv));
        double ex = du-cu, ey = dv-cv;                     // in-plane overshoot
        double dist = std::sqrt(dn*dn + ex*ex + ey*ey);
        if (dist < best) best = dist;
    }
    for (const Sphere& s : scene.spheres) {
        double dist = std::fabs(norm(P - s.c) - s.r);
        if (dist < best) best = dist;
    }
    return best;
}

// A closed-ish room: floor + 4 walls (all facing inward) + a sphere object.
inline Scene make_room(double half=4.0, double height=3.0) {
    Scene s;
    // floor at z=0 (normal +z), extent 2*half square
    s.planes.push_back({{0,0,0},{0,0,1},{1,0,0},{0,1,0}, half, half});
    // +x wall (normal -x), -x wall (normal +x)
    s.planes.push_back({{ half,0,height/2},{-1,0,0},{0,1,0},{0,0,1}, half, height/2});
    s.planes.push_back({{-half,0,height/2},{ 1,0,0},{0,1,0},{0,0,1}, half, height/2});
    // +y wall (normal -y), -y wall (normal +y)
    s.planes.push_back({{0, half,height/2},{0,-1,0},{1,0,0},{0,0,1}, half, height/2});
    s.planes.push_back({{0,-half,height/2},{0, 1,0},{1,0,0},{0,0,1}, half, height/2});
    // sphere object sitting on the floor near center
    s.spheres.push_back({{0.8,-0.5,0.8}, 0.8});
    return s;
}

// ---- camera --------------------------------------------------------------
// Pose stored as world->cam rotation Rwc (row-major) and center C (world). ext
// (world->cam [R|t]) is Rwc | -Rwc*C.
struct Camera {
    double fx, fy, cx, cy; int W, H;
    double Rwc[9];   // world->cam
    V3 C;            // camera center in world
};

// Look-from eye toward target with world-up (OpenCV: z forward, y down).
inline Camera look_at(V3 eye, V3 target, V3 up, double fx, double fy, int W, int H) {
    V3 fwd = normalized(target - eye);          // +z_cam
    V3 right = normalized(cross(fwd, up));      // +x_cam  (up is world-up)
    V3 down = cross(fwd, right);                // +y_cam  (y points down)
    Camera c; c.fx=fx; c.fy=fy; c.cx=(W-1)/2.0; c.cy=(H-1)/2.0; c.W=W; c.H=H; c.C=eye;
    // rows of Rwc are the camera axes expressed in world coords
    c.Rwc[0]=right.x; c.Rwc[1]=right.y; c.Rwc[2]=right.z;
    c.Rwc[3]=down.x;  c.Rwc[4]=down.y;  c.Rwc[5]=down.z;
    c.Rwc[6]=fwd.x;   c.Rwc[7]=fwd.y;   c.Rwc[8]=fwd.z;
    return c;
}

inline std::array<float,9> intr_of(const Camera& c) {
    return { (float)c.fx,0,(float)c.cx, 0,(float)c.fy,(float)c.cy, 0,0,1 };
}
inline std::array<float,12> ext_of(const Camera& c) {
    // t = -Rwc * C
    double tx = -(c.Rwc[0]*c.C.x + c.Rwc[1]*c.C.y + c.Rwc[2]*c.C.z);
    double ty = -(c.Rwc[3]*c.C.x + c.Rwc[4]*c.C.y + c.Rwc[5]*c.C.z);
    double tz = -(c.Rwc[6]*c.C.x + c.Rwc[7]*c.C.y + c.Rwc[8]*c.C.z);
    return { (float)c.Rwc[0],(float)c.Rwc[1],(float)c.Rwc[2],(float)tx,
             (float)c.Rwc[3],(float)c.Rwc[4],(float)c.Rwc[5],(float)ty,
             (float)c.Rwc[6],(float)c.Rwc[7],(float)c.Rwc[8],(float)tz };
}

// ---- render --------------------------------------------------------------
// Per-view raycast. depth = camera-frame Z (0 on miss). conf = |cos incidence|
// (grazing => low, like real depth confidence). depth_noise is a fraction of
// depth added along the ray => realistic along-normal surface noise.
struct View {
    int H=0, W=0;
    std::vector<float> depth, conf;       // H*W
    std::vector<uint8_t> valid;           // H*W
    std::array<float,9>  intr{};
    std::array<float,12> ext{};
};

inline View render(const Scene& scene, const Camera& cam,
                   double depth_noise = 0.0, unsigned long long seed = 1) {
    Rng rng(seed);
    View v; v.H=cam.H; v.W=cam.W; v.intr=intr_of(cam); v.ext=ext_of(cam);
    const size_t P=(size_t)cam.H*cam.W;
    v.depth.assign(P,0.f); v.conf.assign(P,0.f); v.valid.assign(P,0);
    // rows of Rwc are cam axes in world; cam->world is the transpose.
    auto cam2world_dir = [&](V3 dc)->V3 {
        return { cam.Rwc[0]*dc.x + cam.Rwc[3]*dc.y + cam.Rwc[6]*dc.z,
                 cam.Rwc[1]*dc.x + cam.Rwc[4]*dc.y + cam.Rwc[7]*dc.z,
                 cam.Rwc[2]*dc.x + cam.Rwc[5]*dc.y + cam.Rwc[8]*dc.z };
    };
    for (int y=0;y<cam.H;++y) for (int x=0;x<cam.W;++x) {
        size_t idx=(size_t)y*cam.W+x;
        V3 dc = { (x-cam.cx)/cam.fx, (y-cam.cy)/cam.fy, 1.0 };  // dir, cam frame
        V3 Dw = cam2world_dir(dc);                              // world dir; cam-Z of O+t*Dw is t
        Hit h = scene.raycast(cam.C, Dw);
        if (!h.ok) continue;
        double d = h.t;                                        // camera-frame Z
        if (depth_noise > 0) d += rng.gauss() * depth_noise * d;
        v.depth[idx]=(float)d; v.valid[idx]=1;
        double c = std::fabs(dot(normalized(Dw), h.nrm));      // incidence
        v.conf[idx]=(float)std::max(0.05, c);
    }
    return v;
}

// Back-project a View to WORLD points using the pipeline's exact math (Kinv*(u,v,1)*d
// then c2w = inv(ext)). Fills xyz (3N) for valid pixels only. Mirrors src/stream.cpp.
inline void back_project_world(const View& v, std::vector<double>& xyz) {
    xyz.clear();
    const double fx=v.intr[0], fy=v.intr[4], cx=v.intr[2], cy=v.intr[5];
    // c2w: since ext=[Rwc|t], world = Rwc^T (x_cam - t) ... but x_cam = d*(dx,dy,1).
    // Equivalent: P_world = C + d * cam2world_dir((u-cx)/fx,(v-cy)/fy,1).
    // Recover C and Rwc^T from ext.
    const double R[9]={v.ext[0],v.ext[1],v.ext[2], v.ext[4],v.ext[5],v.ext[6], v.ext[8],v.ext[9],v.ext[10]};
    const double t[3]={v.ext[3],v.ext[7],v.ext[11]};
    // C = -Rwc^T t
    double C[3]={ -(R[0]*t[0]+R[3]*t[1]+R[6]*t[2]),
                  -(R[1]*t[0]+R[4]*t[1]+R[7]*t[2]),
                  -(R[2]*t[0]+R[5]*t[1]+R[8]*t[2]) };
    for (int y=0;y<v.H;++y) for (int x=0;x<v.W;++x) {
        size_t idx=(size_t)y*v.W+x; if(!v.valid[idx]) continue;
        double d=v.depth[idx];
        double dcx=(x-cx)/fx, dcy=(y-cy)/fy, dcz=1.0;
        // world dir = Rwc^T * dc
        double wx=R[0]*dcx+R[3]*dcy+R[6]*dcz;
        double wy=R[1]*dcx+R[4]*dcy+R[7]*dcz;
        double wz=R[2]*dcx+R[5]*dcy+R[8]*dcz;
        xyz.push_back(C[0]+d*wx); xyz.push_back(C[1]+d*wy); xyz.push_back(C[2]+d*wz);
    }
}

// Dense GT surface samples (world), for Chamfer / F-score references.
inline void sample_surface(const Scene& scene, int per_plane, int per_sphere,
                           std::vector<double>& xyz, unsigned long long seed=7) {
    Rng rng(seed); xyz.clear();
    for (const Plane& p : scene.planes)
        for (int i=0;i<per_plane;++i) {
            V3 q = p.c + p.u*(rng.sym()*p.hu) + p.v*(rng.sym()*p.hv);
            xyz.push_back(q.x); xyz.push_back(q.y); xyz.push_back(q.z);
        }
    for (const Sphere& s : scene.spheres)
        for (int i=0;i<per_sphere;++i) {
            V3 d=normalized({rng.sym(),rng.sym(),rng.sym()}); V3 q=s.c+d*s.r;
            xyz.push_back(q.x); xyz.push_back(q.y); xyz.push_back(q.z);
        }
}

// Camera trajectory that orbits inside the room and RETURNS to the start
// (closed loop => the same walls are seen early and late; drift shows as double
// walls). Cameras look outward from a circle of the given radius.
inline std::vector<Camera> loop_trajectory(int n, double radius, double z,
                                            double fx, double fy, int W, int H) {
    std::vector<Camera> cams; cams.reserve(n);
    const double TWO_PI = 6.283185307179586;
    for (int i=0;i<n;++i) {
        double a = TWO_PI * (double)i / (double)n;   // i=0 and i=n coincide => loop
        V3 eye = { radius*std::cos(a), radius*std::sin(a), z };
        V3 tgt = { 2.0*radius*std::cos(a), 2.0*radius*std::sin(a), z*0.8 }; // look outward
        cams.push_back(look_at(eye, tgt, V3{0,0,1}, fx, fy, W, H));
    }
    return cams;
}

}} // namespace da::synth
