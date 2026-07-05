#!/usr/bin/env python3
# Differential oracle for task A (src/fuse.cpp). Diffs our C++ fusion against
# reference implementations on a SURFACE-concentrated cloud (planes + sphere):
#   * voxel stage  vs open3d.voxel_down_sample
#   * radius stage vs an independent scipy cKDTree neighbourhood-centroid
# Run via:  scripts/oracle.sh python scripts/oracle_fuse.py <fuse_dump-binary>
import os, struct, subprocess, sys, tempfile
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

DUMP = sys.argv[1] if len(sys.argv) > 1 else "./build/tests/fuse_dump"

def surface_cloud(seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    # floor z=0, +x wall x=2, -y wall y=-2 (each a thin noisy sheet)
    xy = rng.uniform(-2, 2, (20000, 2)); parts.append(np.c_[xy, rng.normal(0, 0.004, 20000)])
    yz = rng.uniform(-2, 2, (20000, 2)); parts.append(np.c_[2 + rng.normal(0, 0.004, 20000), yz])
    xz = rng.uniform(-2, 2, (16000, 2)); parts.append(np.c_[xz[:,0], -2 + rng.normal(0, 0.004, 16000), xz[:,1]])
    d = rng.normal(size=(14000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)   # sphere r=1
    parts.append(d)
    return np.vstack(parts).astype(np.float32)

def run_dump(P, voxel, radius=0.0):
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(ip, "wb") as f:
            f.write(struct.pack("<i", len(P))); f.write(P.tobytes())
        subprocess.run([DUMP, ip, op, str(voxel), str(radius)], check=True)
        with open(op, "rb") as f:
            M = struct.unpack("<i", f.read(4))[0]
            return np.frombuffer(f.read(), np.float32).reshape(M, 3).astype(np.float64)

def chamfer(a, b):
    d1, _ = cKDTree(b).query(a); d2, _ = cKDTree(a).query(b)
    return 0.5 * (d1.mean() + d2.mean())

fails = 0
def check(ok, msg):
    global fails
    print(("ok:   " if ok else "FAIL: ") + msg)
    if not ok: fails += 1

P = surface_cloud()

# ---- 1. voxel stage: EXACT vs numpy floor-partition + surface-equiv to open3d ----
# The unambiguous reference is the floor(p/voxel) partition centroid — our fuse's
# exact defined semantics. np.unique(axis=0) sorts keys lexicographically, the
# same order our fuse emits, so we can compare point-by-point.
vox = 0.05
ours = run_dump(P, vox, 0.0)
keys = np.floor(P.astype(np.float64) / vox).astype(np.int64)
uk, inv = np.unique(keys, axis=0, return_inverse=True)
M = len(uk); sums = np.zeros((M, 3)); cnt = np.zeros(M)
np.add.at(sums, inv, P.astype(np.float64)); np.add.at(cnt, inv, 1.0)
ref_np = sums / cnt[:, None]
same_n = (len(ours) == M)
maxerr = np.max(np.linalg.norm(ours - ref_np, axis=1)) if same_n else float("inf")
print(f"[voxel] ours={len(ours)}  numpy-ref={M}  max|ours-ref|={maxerr*1e6:.2f}um")
check(same_n, "voxel count == numpy floor-partition count (exact)")
check(maxerr < 1e-4, "voxel centroids == numpy reference (exact)")

# open3d uses its own (coarser) internal voxel convention, so its COUNT differs;
# what matters is that our output describes the SAME surface as the reference lib.
pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
ref_o3d = np.asarray(pc.voxel_down_sample(vox).points)
ch = chamfer(ours, ref_o3d)
print(f"[voxel] open3d={len(ref_o3d)}  chamfer(ours,open3d)={ch*1000:.3f}mm  (voxel={vox*1000:.0f}mm)")
check(ch < 0.5 * vox, "voxel output is the same surface as open3d (chamfer < voxel/2)")

# ---- 2. radius stage vs scipy cKDTree neighbourhood centroid ----
# Use a smaller cloud (radius smoothing is O(N*k)); compare point-by-point since
# radius-only fusion (voxel=0) preserves input order.
Ps = P[::6].copy()
r = 0.05
ours_s = run_dump(Ps, 0.0, r)                     # voxel=0 => order preserved, N out
tree = cKDTree(Ps)
ref_s = np.empty_like(Ps, dtype=np.float64)
nbrs = tree.query_ball_point(Ps, r)
for i, nb in enumerate(nbrs):
    ref_s[i] = Ps[nb].mean(axis=0)
maxerr = np.max(np.linalg.norm(ours_s - ref_s, axis=1))
print(f"[radius] N={len(Ps)}  max |ours - scipy_centroid| = {maxerr*1e6:.2f} um")
check(len(ours_s) == len(Ps), "radius stage preserves point count/order")
check(maxerr < 1e-4, "radius smoothing matches scipy neighbourhood centroid")

print(("ORACLE PASS" if fails == 0 else f"ORACLE FAIL ({fails})"))
sys.exit(1 if fails else 0)
