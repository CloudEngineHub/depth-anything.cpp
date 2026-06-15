# M4 — End-to-end multi-view verification (C++ vs original PyTorch)

This is the M4 milestone deliverable: proof that the C++/ggml **multi-view** path
matches the **original** Depth Anything 3 PyTorch model end-to-end, on two
structured (non-fixture-trivial) views, through the full pipeline
(preprocess each view → one DINOv2 backbone pass with cross-view global attention →
per-view DptHead depth + per-view CamPose pose), driven by a single multi-view
backbone forward shared across all views.

## What was verified

Two structured **224×224 uint8** views (the `fixed_input_multiview(S=2)` fixture:
each view a different gradient + block pattern so cross-view attention carries real
signal) are fed to both sides:

- **Reference:** the original model's default forward `net(x_mv)` with `x_mv`
  stacked as `[1, 2, 3, 224, 224]`, returning per-view `depth` (`[1,2,224,224]`),
  `extrinsics` (`[1,2,3,4]`, w2c) and `intrinsics` (`[1,2,3,3]`).
- **C++:** `da3-cli depth --input a.png --input b.png --out-prefix dumps/e2e_mv`,
  which loads both images, runs `Engine::depth_pose_multi` (one
  `DinoBackbone::forward_mv` over both views, then per-view `DptHead::depth` and
  `CamPose::pose` on the layer-11 camera token), and writes
  `dumps/e2e_mv_view{i}.pfm` / `.png` / `.json` per view.

Per-view depth (PFM) and pose (JSON) are compared element-wise against the
reference view-slices.

## Measured result

```
e2e mv: view0 depth max|d|=8.345e-07 mean|d|=1.166e-07 corr=1.000000 / pose ext max|d|=1.676e-08 intr max|d|=6.104e-05
e2e mv: view1 depth max|d|=7.749e-07 mean|d|=1.221e-07 corr=1.000000 / pose ext max|d|=6.668e-07 intr max|d|=1.526e-04
E2E_MV PASS
```

- Depth `max|d| ≈ 8e-07` both views (PASS `< 5e-3`), `corr = 1.000000`
  (PASS `> 0.999`).
- Pose extrinsics `max|d| ≤ 7e-07` (PASS `< 1e-2`) — f32 rounding noise on the
  3×4 world-to-camera matrices.
- Pose intrinsics `max|d| ≤ 1.5e-04` on focal magnitudes ~330–350 px
  (relative ~5e-7) — f32 accumulation noise.

The fixture gate (`tests/test_engine_mv.cpp`) builds two `Image`s from the dumped
`raw_mv_0`/`raw_mv_1`, runs `Engine::depth_pose_multi`, and compares each view's
`depth` / `extrinsics` / `intrinsics` against the dumped `depth_mv` /
`extrinsics_mv` / `intrinsics_mv` view-slices (5e-3 atol+rtol):

```
[mv_depth_v0] n=50176 max|d|=8.345e-07 mean|d|=1.166e-07 -> OK
[mv_ext_v0]   n=12    max|d|=1.676e-08 -> OK
[mv_intr_v0]  n=9     max|d|=6.104e-05 -> OK
[mv_depth_v1] n=50176 max|d|=7.749e-07 mean|d|=1.221e-07 -> OK
[mv_ext_v1]   n=12    max|d|=6.668e-07 -> OK
[mv_intr_v1]  n=9     max|d|=1.526e-04 -> OK
```

Accumulated error through the full multi-view backbone (incl. cross-view global
attention) is tiny — well under tolerance and on par with the single-view path: the
C++ port reproduces the original model's per-view depth and camera pose up to
float32 accumulation order.

## Reproduce

```sh
. .venv/bin/activate
cmake -B build -DDA_BUILD_TESTS=ON -DDA_BUILD_CLI=ON && cmake --build build -j
(cd build && ctest --output-on-failure)   # incl. test_engine_mv
python scripts/e2e_verify_mv.py
```

Intermediate artifacts under `dumps/` (gitignored): `dumps/e2e_mv_input{0,1}.png`,
`dumps/e2e_mv_view{0,1}.pfm` / `.png` / `.json`.

## C-API

`da_capi_depth_pose_multi(ctx, image_paths, n_images, &h, &w, &n, out_ext, out_intr)`
returns a malloc'd `float[n*H*W]` view-major depth buffer (free via
`da_capi_free_floats`) and fills `out_ext[i*12]` / `out_intr[i*9]` per view.

## Scope

Same 224×224 parity-verified path as M2/M3, extended to S=2 views sharing one
backbone forward, matching the original model's single multi-view pass structure.
