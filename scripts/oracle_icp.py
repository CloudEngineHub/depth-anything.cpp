#!/usr/bin/env python3
# Differential oracle for task B (src/icp.cpp). On SURFACE-concentrated clouds it
# diffs our point-to-plane ICP against open3d.registration_icp (also point-to-
# plane) AND against the known ground-truth transform:
#   * curvature-rich corner+sphere: both recover a KNOWN rigid transform, and our
#     result agrees with open3d's to <0.1 deg / sub-mm;
#   * single flat wall: both drive the point-to-plane residual to ~0 (the
#     observable normal error), confirming the fit is correct where it is
#     constrained (our rank-truncation of the unobservable tangential slide is
#     asserted separately in tests/test_icp.cpp).
# Run via:  scripts/oracle.sh python scripts/oracle_icp.py <icp_dump-binary>
import os, struct, subprocess, sys, tempfile
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

DUMP = sys.argv[1] if len(sys.argv) > 1 else "./build/tests/icp_dump"
MAXC, NRAD = 0.10, 0.06

fails = 0
def check(ok, msg):
    global fails
    print(("ok:   " if ok else "FAIL: ") + msg)
    if not ok: fails += 1

def corner_scene(seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, (3000, 2)); floor = np.c_[xy[:,0]+0.5, xy[:,1], np.zeros(3000)]
    yz = rng.uniform(-0.5, 0.5, (3000, 2)); wall  = np.c_[np.zeros(3000), yz[:,0], yz[:,1]+0.5]
    d = rng.normal(size=(2500,3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    sph = np.c_[d[:,0]*0.22+0.5, d[:,1]*0.22, d[:,2]*0.22+0.7]
    return np.vstack([floor, wall, sph]).astype(np.float32)

def wall_scene(seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.6, 0.6, (8000, 2))
    return np.c_[xy[:,0], xy[:,1], np.zeros(8000)].astype(np.float32)

def write_cloud(path, P):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(P))); f.write(P.astype(np.float32).tobytes())

def run_ours(src, tgt, td):
    sp, tp, op = (os.path.join(td, n) for n in ("s.bin", "t.bin", "o.bin"))
    write_cloud(sp, src); write_cloud(tp, tgt)
    subprocess.run([DUMP, sp, tp, op, str(MAXC), str(NRAD)], check=True)
    with open(op, "rb") as f:
        hdr = struct.unpack("<13d", f.read(13*8))
        rms = struct.unpack("<2d", f.read(2*8))
        meta = struct.unpack("<2i", f.read(2*4))
    T = np.eye(4); T[:3,:3] = np.array(hdr[1:10]).reshape(3,3); T[:3,3] = hdr[10:13]
    return T, hdr[0], rms, meta   # (4x4, scale, (rms_before,rms_after), (rank,iters))

def run_o3d(src, tgt):
    ps = o3d.geometry.PointCloud(); ps.points = o3d.utility.Vector3dVector(src.astype(np.float64))
    pt = o3d.geometry.PointCloud(); pt.points = o3d.utility.Vector3dVector(tgt.astype(np.float64))
    pt.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=NRAD))
    reg = o3d.pipelines.registration.registration_icp(
        ps, pt, MAXC, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    return np.asarray(reg.transformation), reg

def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))

def apply4(T, P):
    return (P @ T[:3,:3].T) + T[:3,3]

# ================= 1. curvature-rich corner+sphere ==========================
tgt = corner_scene()
Rg = Rotation.from_rotvec(np.radians(2.0) * np.array([0.3,0.8,0.5]) / np.linalg.norm([0.3,0.8,0.5]))
G = np.eye(4); G[:3,:3] = Rg.as_matrix(); G[:3,3] = [0.04,-0.03,0.02]
src = apply4(G, tgt.astype(np.float64)).astype(np.float32)     # src = G(tgt) => T ~ G^-1
Ginv = np.linalg.inv(G)

with tempfile.TemporaryDirectory() as td:
    T_ours, s, rms, meta = run_ours(src, tgt, td)
T_o3d, reg = run_o3d(src, tgt)

ang_ours_gt = rot_angle_deg(T_ours[:3,:3], Ginv[:3,:3]); t_ours_gt = np.linalg.norm(T_ours[:3,3]-Ginv[:3,3])
ang_o3d_gt  = rot_angle_deg(T_o3d[:3,:3],  Ginv[:3,:3]); t_o3d_gt  = np.linalg.norm(T_o3d[:3,3]-Ginv[:3,3])
ang_pair    = rot_angle_deg(T_ours[:3,:3], T_o3d[:3,:3]); t_pair   = np.linalg.norm(T_ours[:3,3]-T_o3d[:3,3])
print(f"[corner] ours vs GT : {ang_ours_gt:.4f} deg  {t_ours_gt*1e3:.3f} mm   (rank={meta[0]} iters={meta[1]} s={s:.4f})")
print(f"[corner] o3d  vs GT : {ang_o3d_gt:.4f} deg  {t_o3d_gt*1e3:.3f} mm   (fitness={reg.fitness:.3f} rmse={reg.inlier_rmse*1e3:.3f}mm)")
print(f"[corner] ours vs o3d: {ang_pair:.4f} deg  {t_pair*1e3:.3f} mm")
check(ang_ours_gt < 0.1 and t_ours_gt < 1e-3, "ours recovers known transform (<0.1deg, <1mm)")
check(ang_o3d_gt  < 0.1 and t_o3d_gt  < 1e-3, "open3d recovers known transform (<0.1deg, <1mm)")
check(ang_pair    < 0.1 and t_pair    < 1e-3, "ours agrees with open3d (<0.1deg, <1mm)")
check(rms[1] < 1e-3, "ours point-to-plane residual converged to ~0")

# ================= 2. flat wall: both drive plane residual to ~0 ============
wt = wall_scene()
Rw = Rotation.from_rotvec(np.radians(1.0) * np.array([1.0,0,0]))
Gw = np.eye(4); Gw[:3,:3] = Rw.as_matrix(); Gw[:3,3] = [0.05, 0.0, 0.03]  # tangential x + normal z + tilt
ws = apply4(Gw, wt.astype(np.float64)).astype(np.float32)

with tempfile.TemporaryDirectory() as td:
    Tw_ours, _, rms_w, meta_w = run_ours(ws, wt, td)
Tw_o3d, reg_w = run_o3d(ws, wt)

# residual to the true plane z=0 after each transform (the OBSERVABLE error).
def plane_resid_z(T, P):
    return float(np.sqrt(np.mean(apply4(T, P.astype(np.float64))[:,2]**2)))
rz_ours, rz_o3d = plane_resid_z(Tw_ours, ws), plane_resid_z(Tw_o3d, ws)
print(f"[wall] plane resid  ours={rz_ours*1e3:.3f}mm  o3d={rz_o3d*1e3:.3f}mm  (ours rank={meta_w[0]}, o3d rmse={reg_w.inlier_rmse*1e3:.3f}mm)")
check(rz_ours < 2e-3, "ours corrects the observable normal error (plane residual ~0)")
check(rz_o3d  < 2e-3, "open3d corrects the observable normal error (plane residual ~0)")
check(meta_w[0] == 3, "ours flags the wall as rank-3 (aperture problem)")

print("ORACLE PASS" if fails == 0 else f"ORACLE FAIL ({fails})")
sys.exit(1 if fails else 0)
