# M3 — End-to-end camera-pose verification (C++ vs original PyTorch)

This is the M3 milestone deliverable: proof that the C++/ggml **camera-pose** path
matches the **original** Depth Anything 3 PyTorch model end-to-end, on a real
(non-fixture) image, through the full pipeline
(preprocess → DINOv2 backbone → layer-11 camera token → CamPose / cam_dec →
extrinsics + intrinsics), reusing a single backbone pass that also drives depth.

## What was verified

On an identical **224×224 uint8** image fed to both sides (the same structured
fallback or `--image` photo used by the M2 depth e2e):

- **Depth (M2, still green):** `da3-cli depth --pfm` vs
  `net.backbone.pretrained.get_intermediate_layers(...)` → `net.head(...)`.
- **Pose (M3):** the same uint8 image is run through `da3-cli depth --pose
  dumps/e2e_pose.json`, which runs the backbone once and emits
  `Engine::depth_pose` (depth via DptHead + pose via CamPose on the layer-11 camera
  token). The reference is the original model's **default forward** `net(img)`
  (`use_ray_pose=False` → cam_dec path) returning `extrinsics` (3×4 w2c) and
  `intrinsics` (3×3). Both pose matrices are compared element-wise.

## Measured result

```
e2e depth: shape=(224, 224) max|d|=9.537e-07 mean|d|=7.868e-08 median_rel=6.247e-08 corr=1.000000
ref range [0.8667,0.9881] cpp range [0.8667,0.9881]
e2e pose: ext max|d|=4.534e-08 intr max|d|=2.594e-04
E2E PASS
```

- Depth `max|d| = 9.5e-07` (PASS `< 5e-3`), `corr = 1.000000` (PASS `> 0.999`).
- Pose extrinsics `max|d| = 4.5e-08` (PASS `< 1e-2`) — f32 rounding noise on the
  3×4 world-to-camera matrix.
- Pose intrinsics `max|d| = 2.6e-04` on focal magnitudes ~244–347 px
  (relative ~1e-6) — also f32 accumulation noise.

The fixture gate (`tests/test_engine_pose.cpp`) feeds the dumped `raw_image`
through `Engine::depth_pose` and compares against the dumped `extrinsics` /
`intrinsics` (5e-3 atol+rtol, accumulated through the backbone):

```
[engine_extrinsics] n=12 max|d|=7.823e-08 mean|d|=2.889e-08 -> OK
[engine_intrinsics] n=9  max|d|=1.526e-04 mean|d|=1.865e-05 -> OK
```

Accumulated error through the full backbone is tiny (well under tolerance): the C++
port reproduces the original model's camera pose up to float32 accumulation order.

## Reproduce

```sh
. .venv/bin/activate
cmake -B build -DDA_BUILD_TESTS=ON -DDA_BUILD_CLI=ON && cmake --build build -j
(cd build && ctest --output-on-failure)   # incl. test_engine_pose
python scripts/e2e_verify.py              # structured fallback image (depth AND pose)
python scripts/e2e_verify.py --image path.jpg
```

Intermediate artifacts under `dumps/` (gitignored): `dumps/e2e_input.png`,
`dumps/e2e_cpp.pfm`, `dumps/e2e_pose.json`.

## Scope

Same 224×224 parity-verified path as M2 (the native-resolution upper-bound-to-504
resize policy is still a later milestone). The pose path reuses one backbone forward
for both depth and pose, matching the original model's single-pass structure.
