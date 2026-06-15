# M5 — 3D Gaussian reconstruction (DA3-GIANT): GaussianAdapter parity + e2e

This is the M5-T4 deliverable: the **GaussianAdapter** (host geometry, no learned
weights) that turns the GSDPT raw gaussian channels + giant depth + camera pose
into **world-space 3D Gaussians**, gated against the reference, plus a `.ply`
export and a `da3 reconstruct` CLI.

## What it does

`src/gs_adapter.{hpp,cpp}` is a pure-host port of
`depth_anything_3.model.gs_adapter.GaussianAdapter.forward`
(sh_degree=2 → d_sh=9, pred_offset_depth, pred_offset_xy, no pred_color,
scale_min=1e-5, scale_max=30.0). For each of the N=H·W pixels it produces:

- **means** — unproject `(depth + offset_depth)` along the world ray:
  `cam2world = affine_inverse(extrinsics)`, `intr_normed = K` with row0/=W,
  row1/=H; pixel `xy = ((w+0.5)/W, (h+0.5)/H) + offset_xy·(1/W,1/H)`;
  `dir = R_c2w · normalize(inv(intr_normed)·(x,y,1))`;
  `mean = cam_center + dir·depth`. (`get_world_rays` + `sample_image_grid`.)
- **scales** — `(scale_min + (scale_max-scale_min)·sigmoid(s))·depth·multiplier`
  where `multiplier = 0.1·(1/fx + 1/fy)` from `get_scale_multiplier`.
- **rotations** — camera quaternion (xyzw, normalized) rotated to world via
  `cam_quat_xyzw_to_world_quat_wxyz` (quat→mat, `R_c2w·R_cam`, mat→quat). Note
  the reference's *historical quirk*: it feeds the wxyz-reordered quaternion
  positionally into `quat_to_mat` (which expects xyzw) and returns `mat_to_quat`'s
  raw xyzw output **labelled** "wxyz" with **no reorder** — we reproduce this
  verbatim so the stored buffer matches.
- **harmonics** — SH coefficients masked
  (`sh_mask[deg²:(deg+1)²] = 0.1·0.25^deg`, DC=1) then rotated to world by
  `rotate_sh` (see below).
- **opacities** — `map_pdf_to_opacity(gs_conf)` with `global_step=0` →
  `exponent=1` → identity (`opacity == gs_conf`).

### rotate_sh — fully ported (e3nn Wigner-D)

The hard part. `rotate_sh` rotates the per-color 9-vector of SH coefficients
(degrees 0,1,2) by the cam2world rotation using **e3nn**'s real-SH Wigner-D
convention. The C++ port reproduces the exact e3nn path (verified against e3nn
itself to ~1e-5, float32):

1. Permute axes yzx→xyz: `permR = Pᵀ · R_c2w · P`, `P=[[0,0,1],[1,0,0],[0,1,0]]`.
2. `matrix_to_angles(permR)` → `(α,β,γ)` (e3nn `Ry(α)Rx(β)Ry(γ)` Euler
   decomposition: `β=acos(x_y)`, `α=atan2(x_x,x_z)` with `x=permR·ŷ`, then
   `γ=atan2(R2[0,2],R2[0,0])`).
3. Per degree l, `wigner_D(l, α, −β, γ) = z_rot(α)·exp(−β·X0ˣ)·z_rot(γ)` where
   `z_rot` is the closed-form real-SH z-rotation and `X0ˣ` is the hardcoded so3
   x-generator (exact entries 0,±1,±√3,±2); `exp` via scaling-and-squaring.
   Degree 0 is identity; degree 1 uses the 3×3 D, degree 2 the 5×5 D.

So **all five attributes incl. harmonics are exact** — no approximation.

## Measured result — gate (`tests/test_gs_adapter.cpp`)

Drives `GsAdapter::build` from the dumped giant `raw_gs` / `gs_conf` / `depth_g`
/ `extrinsics_g` / `intrinsics_g`, comparing each attribute to the dumped
reference (`gs_means`/`gs_scales`/`gs_rotations`/`gs_harmonics`/`gs_opacities`)
at 2e-3 atol+rtol:

```
[gs_means]     n=150528  max|d|=4.578e-05  mean|d|=3.197e-07  -> OK
[gs_scales]    n=150528  max|d|=7.153e-07  mean|d|=5.703e-09  -> OK
[gs_rotations] n=200704  max|d|=2.384e-07  mean|d|=2.124e-08  -> OK
[gs_harmonics] n=1354752 max|d|=1.670e-05  mean|d|=6.072e-07  -> OK
[gs_opacities] n=50176   max|d|=2.980e-08  mean|d|=1.027e-08  -> OK
```

All well under tolerance. The largest absolute error is on `gs_means` (~4.6e-05
on coordinates up to ±180 — relative ~2.5e-7, pure f32/f64 rounding) and on
`gs_harmonics` (~1.7e-5 from the Wigner-D matrix_exp), both float-precision noise.

## .ply export + `da3 reconstruct`

`src/ply_export.{hpp,cpp}` writes a **standard 3DGS (INRIA gaussian-splatting)
binary-little-endian** `.ply`, vertex props in order:
`x y z`, `f_dc_0..2` (SH DC = harmonics[:,:,0]), `opacity` (stored as **logit** /
inverse-sigmoid — viewers apply sigmoid), `scale_0..2` (stored as **log** —
viewers apply exp), `rot_0..3` (quaternion wxyz). This is the convention used by
common splat viewers / the original INRIA renderer.

`Engine::reconstruct(image)` runs the giant pipeline once: backbone →
`DptHead::depth` + `CamPose::pose` + `GsHead::raw_gaussians` →
`GsAdapter::build` → `Gaussians`. CLI:

```sh
da3-cli reconstruct --model models/depth-anything-giant-f32.gguf \
        --input img.png --ply out.ply [--pose out.json]
```

## e2e (full C++ pipeline vs original)

`scripts/e2e_verify_gs.py` feeds an identical 224×224 uint8 image to both the
C++ `da3 reconstruct` CLI (whole giant chain) and the reference
`net(x, infer_gs=True)`, parses the C++ `.ply`, and compares the world-space
gaussian attributes (means / scales / f_dc / opacity / sign-aware rotations).
This exercises the accumulated pipeline error end-to-end (each giant CPU forward
is slow — minutes — as expected). Measured on a structured 224×224 image
(`viol` = elements outside 2e-3 atol+rtol):

```
means     max|d|=4.364e-03  mean|d|=6.006e-05  viol=0/150528
scales    max|d|=1.345e-03  mean|d|=2.127e-06  viol=0/150528
f_dc      max|d|=2.923e-04  mean|d|=2.658e-06  viol=0/150528
opacity   max|d|=8.678e-05  mean|d|=6.910e-07  viol=0/50176
rot       max|d|(sign-aware)=3.034e-05  mean=2.922e-06
```

Zero violations on every attribute through the **whole** C++ giant chain vs the
original PyTorch — the largest error is on `means` (~4.4e-3 on coordinates up to
±180, within rtol; accumulated f32 backbone/depth noise). The adapter itself is
gated exactly by `test_gs_adapter` above; the upstream backbone / depth+pose /
GSDPT raw_gs are gated by `test_backbone_giant` / `test_giant_depth_pose` /
`test_gs_head`.

## Reproduce

```sh
. .venv/bin/activate
cmake -B build -DDA_BUILD_TESTS=ON -DDA_BUILD_CLI=ON && cmake --build build -j
(cd build && ctest -R "test_gs_adapter|test_gs_head" --output-on-failure)
python scripts/e2e_verify_gs.py          # giant; slow (minutes)
```

## Status

GaussianAdapter parity **DONE** on all five attributes (<2e-3), **including
`rotate_sh` fully ported** via the e3nn Wigner-D convention (degrees 0/1/2). No
documented approximation remains.
