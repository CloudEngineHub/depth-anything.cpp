# M6 — Nested metric depth (DA3NESTED-GIANT-LARGE): alignment parity + e2e

This is the M6-T3 deliverable (the FINAL milestone of the DA3 ggml port): the
**nested metric alignment** host math, an `Engine::depth_metric` that runs both
branches end-to-end, a CLI entry, and an e2e check against the original
`NestedDepthAnything3Net.forward`.

## Architecture

`NestedDepthAnything3Net` (`da3.py`) wraps two `DepthAnything3Net` branches:

- **anyview** = GIANT (`models/depth-anything-nested-anyview.gguf`): SwiGLU
  backbone + DualDPT depth head (`depth` + `depth_conf`) + CamPose
  (`extrinsics` + `intrinsics`). Runs via the existing engine giant path.
- **metric** = ViT-L (`models/depth-anything-nested-metric.gguf`): plain backbone
  (cat_token false) + `DptHead::depth_sky` (`depth_metric_raw` + `sky`, output_dim
  1, no confidence).

Both branches consume the **same** preprocessed 224×224 input `x`.

## The alignment (`src/nested.{hpp,cpp}` — pure host)

Ports `NestedDepthAnything3Net.forward` + `utils/alignment.py` exactly:

1. **apply_metric_scaling** — `focal = (Kxx + Kyy)/2` from the *anyview*
   intrinsics; `metric_depth = depth_metric_raw · (focal / 300)`.
2. **\_apply_depth_alignment**:
   - `non_sky = sky < 0.3`.
   - `median_conf = quantile(depth_conf[non_sky], 0.5)` (torch linear-interp
     quantile; 224²=50176 < 100000 so NO sampling — full deterministic quantile).
   - `align_mask = (depth_conf ≥ median_conf) & non_sky & (metric_depth > 1e-2)
     & (depth > 1e-3)`.
   - `scale_factor = least_squares_scale_scalar(metric_depth[m], depth[m])`
     `= Σ(a·b)/Σ(b·b)` with `a = metric_depth`, `b = anyview depth` (maps
     anyview → metric).
   - `depth *= scale_factor`; `extrinsics[:3,3] *= scale_factor` (translation
     column, flat indices 3/7/11).
3. **\_handle_sky_regions** — `non_sky_max = min(quantile(depth[non_sky], 0.99),
   200)`; sky pixels set to `non_sky_max`. (For the noise fixture sky is all
   < 0.3, so this is identity — but it is implemented.)

The metric branch additionally applies its OWN sky-fill inside `da3_metric(x)`
(`DepthAnything3Net._process_mono_sky_estimation`: sky → `quantile(non_sky, 0.99)`
uncapped, guarded by >10 non-sky AND >10 sky pixels). This is reproduced in
`NestedAligner::process_mono_sky` and applied to the metric depth before
alignment (a no-op on the no-sky fixture).

`torch.quantile` linear interpolation is matched exactly: sort ascending,
`pos = q·(n−1)`, interpolate `v[floor] + frac·(v[ceil]−v[floor])`.

## Gates

- **`tests/test_nested_align.cpp`** (fast, isolated) — loads the dumped anyview +
  metric branch tensors from `dumps/reference_nested.gguf`, runs `NestedAligner`,
  compares `depth_final` / `scale_factor` / `extrinsics_final` at 2e-3 (rel).
  The host math is bit-exact (the dump inputs are the exact reference branch
  outputs):

  ```
  [depth_final]      n=50176 max|d|=0.000e+00 mean|d|=0.000e+00 -> OK
  [scale_factor]     n=1     max|d|=0.000e+00 mean|d|=0.000e+00 (got=2.37049 ref=2.37049) -> OK
  [extrinsics_final] n=12    max|d|=0.000e+00 mean|d|=0.000e+00 -> OK
  ```

- **`tests/test_engine_metric.cpp`** (slow, ~14 s) — feeds the dumped `raw_image`
  through the FULL `Engine::depth_metric` (both ggml backbones + heads + cam pose
  + NestedAligner) and compares vs the dump at 5e-3:

  ```
  [depth_final]      n=50176 max|d|=1.860e-05 mean|d|=1.775e-05 -> OK
  [scale_factor]     n=1     max|d|=1.812e-05 (got=2.37050 ref=2.37049) -> OK
  [extrinsics_final] n=12    max|d|=6.496e-08 -> OK
  ```

## CLI

```
da3-cli depth --model <anyview.gguf> --metric-model <metric.gguf> --input img.png --pfm out.pfm [--png out.png] [--pose out.json]
```

`--metric-model` switches the `depth` subcommand into the nested metric pipeline
(`Engine::load_nested` + `Engine::depth_metric`), emitting metric-scale depth +
the scale_factor and (translation-scaled) extrinsics.

## e2e

`scripts/e2e_verify_nested.py` runs the real `NestedDepthAnything3Net.forward`
(`net(x)`, giant + large backbones on CPU — minutes) on a 224×224 image and the
C++ CLI nested pipeline on the identical uint8 input, comparing metric-scale
depth. PASS criterion: `max|d| < 5e-3` and `corr > 0.999`. On a 224×224 structured
image:

```
metric depth 224x224 min=1.5436 max=1.6442 scale_factor=1.666550
e2e nested depth: shape=(224, 224) max|d|=2.086e-05 mean|d|=1.941e-05 median_rel=1.218e-05 corr=1.000000
ref range [1.5436,1.6442] cpp range [1.5436,1.6442]  ref scale=1.666530
E2E_NESTED PASS
```
