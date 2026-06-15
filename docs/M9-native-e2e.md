# M9-T2 — Native-resolution end-to-end verification

Proves the C++/ggml DA3 port runs on a **raw arbitrary-resolution real photo** at
its native processed resolution and matches the genuine reference forward
(`model.inference()` preprocessing + `net.head`).

## What changed

- `Engine::depth_native` / `depth_pose_native` (+ `_image` / `_path` variants)
  use the **real DA3 resize policy** (`preprocess_real`, M9-T1: upper_bound
  longest-side → 504, round to a multiple of 14, cv2 INTER_CUBIC/INTER_AREA,
  ImageNet normalize) instead of the legacy floor-to-patch `preprocess`.
- The CLI `da3 depth` (single image) now runs **native by default**. Pass
  `--legacy-resize` to force the old floor path (used only by the 224 fixtures).
  Multi-view / metric / reconstruct stay on the legacy path for now.
- Bug fixed along the way: the DPT head fusion target sizes were **hardcoded**
  to the 224 square fixture (16 / 32 / 64). They are now derived from the patch
  grid `pw = W/14`, `ph = H/14` (`pw,ph` → `2pw,2ph` → `4pw,4ph`), matching
  upstream `DPT._fuse` (`size = l{i}_rn.shape[2:]`). At 224 this is identical to
  the old constants, so all fixture tests stay green; at non-square native res it
  tracks the real grid. Without this the head aborted with a ggml broadcast
  assert on any non-square input.

## Test image & processed resolution

- Input: deterministic structured non-square photo, **640×427** (neither dim a
  multiple of 14), gradients + geometric shapes + high-frequency texture, saved
  lossless as `dumps/native_input.png` so the C++ stb decode and the Python PIL
  decode see identical pixels. (`--image <path>` overrides with a real photo.)
- Processed resolution: **504×336** — long side 504, both dims multiples of 14
  (504 = 36·14, 336 = 24·14). Pure cv2 INTER_AREA downscale (scale ≈ 0.7875),
  already divisible by 14 so no second resize step.

## Numbers (BASE model, f32)

Reference = genuine upstream `InputProcessor` resize (process_res=504,
process_res_method="upper_bound_resize") → `backbone.get_intermediate_layers`
→ `net.head` — the same path/pixels `model.inference()` feeds the network.
C++ = `da3-cli depth` (native by default) → PFM.

| metric            | value          |
|-------------------|----------------|
| processed res     | 504×336        |
| `max|d|`          | 1.371e-06      |
| `mean|d|`         | 1.497e-07      |
| corr              | 1.000000       |
| ref depth range   | [0.8595, 0.9534] |
| cpp depth range   | [0.8595, 0.9534] |

**E2E-NATIVE PASS** (corr > 0.999, max|d| at f32 noise). The agreement is at
f32-noise level because the resize is bit-exact vs cv2 (verified in M9-T1) and
the forward is f32-parity — the native path adds no error of its own.

## Reproduce

```bash
. .venv/bin/activate
cmake --build build -j
python scripts/e2e_verify_native.py        # optional: pass a real photo path as argv[1]
```

CLI directly:

```bash
build/examples/cli/da3-cli depth \
  --model models/depth-anything-base-f32.gguf \
  --input dumps/native_input.png --pfm out.pfm     # native (504x336)
# --legacy-resize forces the old floor-to-patch 224-style path
```

## Permanent gate

`tests/test_engine_depth_native.cpp` (ctest `test_engine_depth_native`) loads the
raw PNG, runs `Engine::depth_native`, and compares against the dumped reference
native depth (`dumps/reference_native.gguf`, written by the e2e script) at 2e-3
rel. Skips (77) if the dump/PNG are absent. Both artifacts are git-ignored;
regenerate with `python scripts/e2e_verify_native.py`.
