# M7 — Quantization Accuracy

Verification that quantized GGUF models preserve depth and pose relative to the
f32 reference. Numbers below are measured by `tests/test_quantize_accuracy.cpp`,
which quantizes the f32 model on the fly, runs the full `depth_pose` pipeline on
the dumped `raw_image` (224×224×3), and compares the predicted depth against the
dumped f32 `head_depth` (224×224) and the predicted extrinsics against the dumped
`extrinsics` (3×4).

Depth values are in the ~0.82–1.0 range, so absolute and relative error are
effectively the same magnitude. `corr` is the Pearson correlation of the
flattened depth map vs the f32 reference. `ext max|d|` is the max absolute
difference over the 12 extrinsic (3×4 pose) entries.

## Results (DA3-BASE)

| model | size   | depth max\|d\| vs f32 | depth corr | ext max\|d\| |
|-------|--------|-----------------------|------------|--------------|
| f32   | 393.0 MB | 0           (exact)   | 1.000000   | 0            (exact) |
| q8_0  | 141.8 MB | 1.90e-03              | 0.999979   | 2.04e-03     |
| q4_k  | 99.1 MB  | 1.90e-02              | 0.998579   | 1.79e-02     |

- **q8_0** is near-lossless: depth max\|d\| ≈ 1.9e-3 and correlation ≈ 0.99998.
- **q4_k** has small, bounded degradation: depth max\|d\| ≈ 1.9e-2 (≈2% on
  depths near 1.0) and correlation > 0.998, well above the 0.99 floor.

### Why depth degradation stays small

Only the backbone attention/MLP matmuls and the camera-decoder (`cam_dec`)
matmuls are quantized. The DPT depth-head convolutions stay f32, so the final
depth regression path is not quantized and small per-layer matmul errors do not
exp-amplify into large depth errors. This is why even q4_k keeps depth
correlation above 0.998.

## Test assertions

`test_quantize_accuracy` enforces:

- q8_0 depth max\|d\| < 2e-2  (measured 1.90e-3)
- q4_k depth correlation > 0.99 (measured 0.998579)

The test SKIPs (exit 77) when `DA_TEST_GGUF` / `DA_TEST_BASELINE` are unset.

## Reproduce

Run the accuracy test (envs are wired by `tests/CMakeLists.txt`):

```sh
cmake --build build -j
cd build && ctest -R test_quantize_accuracy --output-on-failure
```

To produce a quantized model with the CLI:

```sh
# q8_0 (near-lossless)
da3-cli quantize models/depth-anything-base-f32.gguf models/depth-anything-base-q8_0.gguf q8_0

# q4_k (smallest, bounded degradation)
da3-cli quantize models/depth-anything-base-f32.gguf models/depth-anything-base-q4_k.gguf q4_k
```

Supported types: `f16 | q8_0 | q6_k | q5_k | q4_k`.
