// Standalone harness for scripts/oracle_posegraph.py (NOT a ctest): reads a Sim3
// pose graph, runs src/posegraph.cpp optimization, writes the optimized nodes, so
// the Python oracle can diff against gtsam's Pose3 pose-graph (BetweenFactorPose3
// + PriorFactorPose3 + LevenbergMarquardt) on the SE3 (scale=1) case.
//
// All little-endian. Node/edge transforms serialized as 13 f64 = [s, R0..R8, t0..t2].
//   in:  int32 N; N*(13 f64);  int32 M; M*(int32 i, int32 j, f64 info, 13 f64 z);
//        int32 F; F*int32 fixed_idx
//   out: int32 N; N*(13 f64);  f64 cost0; f64 cost; int32 iters
//   posegraph_dump <in> <out>
#include "posegraph.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

static da::Sim3 read_T(FILE* f) {
    double b[13]; std::fread(b, sizeof(double), 13, f);
    da::Sim3 T; T.s=b[0]; for(int k=0;k<9;++k) T.R[k]=b[1+k]; T.t[0]=b[10]; T.t[1]=b[11]; T.t[2]=b[12];
    return T;
}
static void write_T(FILE* f, const da::Sim3& T) {
    double b[13]; b[0]=T.s; for(int k=0;k<9;++k) b[1+k]=T.R[k]; b[10]=T.t[0]; b[11]=T.t[1]; b[12]=T.t[2];
    std::fwrite(b, sizeof(double), 13, f);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: posegraph_dump <in> <out>\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    da::PoseGraph g;
    int32_t N=0; std::fread(&N, sizeof(int32_t), 1, f);
    g.nodes.resize(N); for (int i=0;i<N;++i) g.nodes[i]=read_T(f);
    int32_t M=0; std::fread(&M, sizeof(int32_t), 1, f);
    g.edges.resize(M);
    for (int e=0;e<M;++e) {
        int32_t i,j; double info;
        std::fread(&i,sizeof(int32_t),1,f); std::fread(&j,sizeof(int32_t),1,f); std::fread(&info,sizeof(double),1,f);
        g.edges[e].i=i; g.edges[e].j=j; g.edges[e].info=info; g.edges[e].z=read_T(f);
    }
    int32_t F=0; std::fread(&F, sizeof(int32_t), 1, f);
    g.fixed.assign(N, 0);
    for (int k=0;k<F;++k) { int32_t idx; std::fread(&idx,sizeof(int32_t),1,f); if(idx>=0&&idx<N) g.fixed[idx]=1; }
    std::fclose(f);

    da::PoseGraphResult r = da::optimize_pose_graph(g);

    FILE* o = std::fopen(argv[2], "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    std::fwrite(&N, sizeof(int32_t), 1, o);
    for (int i=0;i<N;++i) write_T(o, g.nodes[i]);
    double c0=r.cost0, c=r.cost; int32_t it=r.iters;
    std::fwrite(&c0, sizeof(double), 1, o); std::fwrite(&c, sizeof(double), 1, o); std::fwrite(&it, sizeof(int32_t), 1, o);
    std::fclose(o);
    return 0;
}
