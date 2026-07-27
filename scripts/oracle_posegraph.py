#!/usr/bin/env python3
# Differential oracle for task C (src/posegraph.cpp). Cross-checks our Sim3 pose-
# graph optimizer against gtsam's Pose3 pose-graph (BetweenFactorPose3 +
# PriorFactorPose3 + LevenbergMarquardt) on the SE3 (scale=1) case, which is what
# gtsam 4.2's Python wrapper can express (it has no BetweenFactorSimilarity3).
#   1. consistent ring + loop + perturbed init -> both recover the KNOWN poses,
#      and our result agrees with gtsam node-for-node;
#   2. systematic drift + strong loop closure -> both cut ATE the same way.
# The Sim3 SCALE dimension (which gtsam Python can't graph-optimize) is validated
# analytically in tests/test_posegraph.cpp.
#   scripts/oracle.sh python scripts/oracle_posegraph.py <posegraph_dump-binary>
import os, struct, subprocess, sys, tempfile
import numpy as np
import gtsam
from scipy.spatial.transform import Rotation

DUMP = sys.argv[1] if len(sys.argv) > 1 else "./build/tests/posegraph_dump"
np.random.seed(7)

fails = 0
def check(ok, msg):
    global fails
    print(("ok:   " if ok else "FAIL: ") + msg)
    if not ok: fails += 1

# ---- SE3 helpers (4x4 homogeneous) ---------------------------------------
def se3(R, t):
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t; return M
def inv(M):
    R = M[:3,:3]; t = M[:3,3]; o = np.eye(4); o[:3,:3] = R.T; o[:3,3] = -R.T @ t; return o
def rand_se3(tsc, rsc):
    return se3(Rotation.from_rotvec(rsc*np.random.randn(3)).as_matrix(), tsc*np.random.randn(3))
def pose_diff(A, B):
    dt = np.linalg.norm(A[:3,3] - B[:3,3])
    Rr = A[:3,:3].T @ B[:3,:3]
    ang = np.degrees(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
    return dt, ang
def gt_loop(N, radius):
    P = []
    for i in range(N):
        a = 2*np.pi*i/N
        R = Rotation.from_rotvec(a*np.array([0,0,1.0])).as_matrix()
        P.append(se3(R, [radius*np.cos(a), radius*np.sin(a), 0.2*np.sin(2*a)]))
    return P

# ---- pose-graph (de)serialization for our dump ---------------------------
def pack_T(M):          # SE3 as Sim3 with s=1: 13 f64 [s, R0..8, t0..2]
    R = M[:3,:3].reshape(-1); t = M[:3,3]
    return struct.pack("<13d", 1.0, *R, *t)
def unpack_T(buf):
    b = struct.unpack("<13d", buf); R = np.array(b[1:10]).reshape(3,3)
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = b[10:13]; return M
def run_ours(nodes, edges, fixed, td):
    ip, op = os.path.join(td,"in.bin"), os.path.join(td,"out.bin")
    with open(ip,"wb") as f:
        f.write(struct.pack("<i", len(nodes)))
        for M in nodes: f.write(pack_T(M))
        f.write(struct.pack("<i", len(edges)))
        for (i,j,info,Z) in edges:
            f.write(struct.pack("<ii", i, j)); f.write(struct.pack("<d", info)); f.write(pack_T(Z))
        f.write(struct.pack("<i", len(fixed)))
        for idx in fixed: f.write(struct.pack("<i", idx))
    subprocess.run([DUMP, ip, op], check=True)
    with open(op,"rb") as f:
        N = struct.unpack("<i", f.read(4))[0]
        out = [unpack_T(f.read(13*8)) for _ in range(N)]
        c0, c = struct.unpack("<dd", f.read(16)); it = struct.unpack("<i", f.read(4))[0]
    return out, c0, c, it

# ---- gtsam Pose3 pose-graph ----------------------------------------------
def to_pose3(M):
    return gtsam.Pose3(gtsam.Rot3(M[:3,:3].copy()), M[:3,3].copy())
def run_gtsam(nodes, edges, fixed):
    graph = gtsam.NonlinearFactorGraph()
    tight = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    for idx in fixed: graph.add(gtsam.PriorFactorPose3(int(idx), to_pose3(nodes[idx]), tight))
    for (i,j,info,Z) in edges:
        nm = gtsam.noiseModel.Isotropic.Sigma(6, 1.0/np.sqrt(info))
        graph.add(gtsam.BetweenFactorPose3(int(i), int(j), to_pose3(Z), nm))
    init = gtsam.Values()
    for i,M in enumerate(nodes): init.insert(int(i), to_pose3(M))
    res = gtsam.LevenbergMarquardtOptimizer(graph, init).optimize()
    return [res.atPose3(int(i)).matrix() for i in range(len(nodes))]

def rel(A, B): return inv(A) @ B

# ================= 1. consistent ring + loop, perturbed init ================
N = 12; GT = gt_loop(N, 2.5)
edges = [(i, (i+1) % N, 1.0, rel(GT[i], GT[(i+1) % N])) for i in range(N)]   # ring (incl loop)
init = [GT[0]] + [GT[i] @ rand_se3(0.3, 0.25) for i in range(1, N)]
fixed = [0]
with tempfile.TemporaryDirectory() as td:
    ours, c0, c, it = run_ours(init, edges, fixed, td)
gt_out = run_gtsam(init, edges, fixed)
d_ours = max(pose_diff(ours[i], GT[i]) for i in range(N))
d_gt   = max(pose_diff(gt_out[i], GT[i]) for i in range(N))
d_pair = max(pose_diff(ours[i], gt_out[i]) for i in range(N))
print(f"[exact] ours vs GT : {d_ours[0]*1e3:.4f} mm  {d_ours[1]:.5f} deg   (cost {c0:.2e}->{c:.2e}, it={it})")
print(f"[exact] gtsam vs GT: {d_gt[0]*1e3:.4f} mm  {d_gt[1]:.5f} deg")
print(f"[exact] ours vs gts: {d_pair[0]*1e3:.4f} mm  {d_pair[1]:.5f} deg")
check(d_ours[0] < 1e-4 and d_ours[1] < 1e-3, "ours recovers known poses (<0.1mm, <1e-3deg)")
check(d_gt[0]   < 1e-4 and d_gt[1]   < 1e-3, "gtsam recovers known poses (<0.1mm, <1e-3deg)")
check(d_pair[0] < 1e-4 and d_pair[1] < 1e-3, "ours agrees with gtsam node-for-node")

# ================= 2. systematic drift + strong loop closure ================
N = 16; GT = gt_loop(N, 3.0)
bias = se3(Rotation.from_rotvec(0.010*np.array([0,0,1.0])).as_matrix(), [0.010, 0, 0])
edges = []; nodes = [GT[0].copy()]
for i in range(N-1):
    Z = rel(GT[i], GT[i+1]) @ bias @ rand_se3(0.004, 0.004)   # systematic + small iid
    edges.append((i, i+1, 1.0, Z))
    nodes.append(nodes[i] @ Z)                                 # chain => accumulates drift
edges.append((N-1, 0, 25.0, rel(GT[N-1], GT[0])))             # accurate loop closure
fixed = [0]
with tempfile.TemporaryDirectory() as td:
    ours2, _, _, _ = run_ours(nodes, edges, fixed, td)
gt2 = run_gtsam(nodes, edges, fixed)
def ate(P): return np.sqrt(np.mean([np.linalg.norm(P[i][:3,3]-GT[i][:3,3])**2 for i in range(N)]))
a_open, a_ours, a_gt = ate(nodes), ate(ours2), ate(gt2)
d_pair2 = max(pose_diff(ours2[i], gt2[i]) for i in range(N))
# On INCONSISTENT data the two optima differ by the parametrization of the
# translation error (we use E.t directly; gtsam's Pose3 Logmap folds in the SE3
# left-Jacobian), so we cross-check EQUIVALENT BEHAVIOUR — both cut the drift to
# the same degree and agree to a small fraction of the corrected drift — not a
# bit-identical trajectory.
print(f"[drift] ATE open={a_open*1e3:.2f}mm  ours={a_ours*1e3:.2f}mm  gtsam={a_gt*1e3:.2f}mm")
print(f"[drift] ours vs gtsam: {d_pair2[0]*1e3:.3f} mm ({100*d_pair2[0]/a_open:.1f}% of drift)  {d_pair2[1]:.4f} deg")
check(a_ours < 0.3*a_open, "ours: loop closure cuts systematic-drift ATE by >70%")
check(a_gt   < 0.3*a_open, "gtsam: loop closure cuts systematic-drift ATE by >70%")
check(abs(a_ours-a_gt) < 0.2*max(a_ours,a_gt), "ours & gtsam fix the drift to the same degree (ATE within 20%)")
check(d_pair2[0] < 0.05*a_open and d_pair2[1] < 0.1, "ours agrees with gtsam to <5% of the corrected drift, <0.1deg")

print("ORACLE PASS" if fails == 0 else f"ORACLE FAIL ({fails})")
sys.exit(1 if fails else 0)
