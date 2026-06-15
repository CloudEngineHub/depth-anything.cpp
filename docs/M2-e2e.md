# M2 — End-to-end depth verification (C++ vs original PyTorch)

This is the M2 milestone deliverable: proof that the C++/ggml depth path matches the
**original** Depth Anything 3 PyTorch model end-to-end, on a real (non-fixture) image,
through the full pipeline (preprocess → DINOv2 backbone → DualDPT head → depth).

## What was verified

The C++ `da3-cli depth` output is compared against the original DA3 network
(`net.backbone.pretrained.get_intermediate_layers(...)` → `net.head(...)`) on an
identical **224×224 uint8** image fed to both sides:

- A real image (`--image`) is loaded/resized to 224×224, or a deterministic
  *structured* fallback (smooth gradients + blocks, so the depth has real
  structure — not noise) is used.
- **Reference (PyTorch):** the uint8 image is normalized with ImageNet mean/std,
  run through `get_intermediate_layers(n=[5,7,9,11], ref_view_strategy="saddle_balanced")`,
  then `net.head(list(outs), 224, 224, patch_start_idx=0)` → depth (224,224).
- **C++:** the same uint8 image is written as a lossless PNG and fed to
  `da3-cli depth --pfm`. PNG is lossless so stb reads byte-identical uint8; the C++
  preprocess resizes 224→224 (a no-op) then applies the same ImageNet normalize.
- The two depth maps are compared (max/mean abs diff, median relative diff,
  correlation, and value ranges).

This is the same path the parity dumps exercise, but with a real/structured image
instead of fixed noise.

## Measured result

```
e2e depth: shape=(224, 224) max|d|=9.537e-07 mean|d|=7.868e-08 median_rel=6.247e-08 corr=1.000000
ref range [0.8667,0.9881] cpp range [0.8667,0.9881]
E2E PASS
```

- `max|d| = 9.5e-07` (PASS threshold `< 5e-3`) — essentially f32 rounding noise.
- `corr = 1.000000` (PASS threshold `> 0.999`).
- Reference and C++ depth ranges are identical to 4 decimals.

The C++ port reproduces the original model's depth bit-for-bit up to float32
accumulation order.

## Reproduce

```sh
. .venv/bin/activate
cmake -B build -DDA_BUILD_CLI=ON && cmake --build build -j --target da3-cli
python scripts/e2e_verify.py                 # structured fallback image
python scripts/e2e_verify.py --image path.jpg # a real photo (resized to 224x224)
```

Intermediate artifacts are written under `dumps/` (gitignored):
`dumps/e2e_input.png` (the shared 224×224 input) and `dumps/e2e_cpp.pfm` (C++ depth).

## Limitation / scope

To keep this e2e on the **parity-verified path**, the input is fixed at **224×224**.
The original model's `inference()` uses a different native-resolution resize policy
(upper-bound to 504), which the C++ preprocess does **not** yet implement (resize is
floor-to-patch-multiple + ImageNet normalize; exact-gated in M1). Giving both sides
the same 224² uint8 input therefore yields a fair numeric comparison of the
implemented pipeline. Full-resolution e2e (matching the upper-bound-to-504 resize) is
a later milestone (M3+).
