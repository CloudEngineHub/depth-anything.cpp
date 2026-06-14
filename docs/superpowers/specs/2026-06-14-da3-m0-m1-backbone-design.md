# Depth Anything 3 → C++/ggml port — M0 + M1 design

Date: 2026-06-14
Status: approved scope, ready for implementation plan
Anchor checkpoint: **DA3-BASE** (ViT-Base, `depth-anything/DA3-BASE`, Apache 2.0)

## 1. Context

We are porting ByteDance's Depth Anything 3 (DA3) to a standalone C++/ggml
implementation with self-contained GGUF weights and no Python at inference, then
exposing it as a LocalAI backend. The reference implementation lives at
`/tmp/da3-src` (github.com/bytedance-seed/depth-anything-3); the architecture is
a modified DINOv2 backbone → DualDPT depth head → optional camera decoder →
optional 3D-Gaussian head, with a nested metric branch for metric checkpoints.

The full project is decomposed into parity-gated milestones M0–M8 (scaffolding,
backbone, depth head, pose, multi-view cross-attention, 3D Gaussians, nested
metric, quantization, LocalAI backend). **This document specifies only M0 and
M1** — the buildable foundation. M2+ get their own spec → plan → build cycles.

Templates we follow for structure and conventions: `~/_git/locate-anything.cpp`
and `~/_git/parakeet.cpp` (file layout, flat C-API, parity-gate tests, GGUF
converter, CMake/ggml static-link).

### Global design principles (apply to all milestones)

- **Parity-first.** The primary test is numerical equality to dumped reference
  tensors, per component, within tolerance — not end-to-end plausibility.
- **Metadata-driven.** Every dimension, hyperparameter, preprocessing constant,
  and architectural flag lives inside the GGUF as KV pairs. The loader reads
  them. Nothing architectural is hardcoded; no external config files shipped.
- **TaskMode selector.** The engine exposes a task scope
  (`DEPTH` / `DEPTH_POSE` / `MULTIVIEW` / `RECONSTRUCT` / `NESTED_METRIC`) so a
  request can ask for *less* and skip heads/branches it does not need. M0/M1 only
  establish the enum and the `DEPTH`-path plumbing; higher modes are wired in
  later milestones.

## 2. Scope of M0 + M1

**In scope**
- M0: repo scaffolding, ggml submodule + CMake, GGUF key registry, the DA3-BASE
  GGUF converter, the C++ model loader + backend, the PyTorch reference-dump
  script, and an `info` CLI subcommand.
- M1: the DINOv2 backbone forward for the **monocular (single image, N=1)** case,
  gated to per-layer numerical parity against the reference dump.

**Explicitly out of scope (later milestones)**
- DualDPT head / depth output (M2) — M1 stops at backbone features.
- Camera pose, ray path, sky head (M3).
- Multi-view cross-view attention, reference-view selection, `cam_enc` token
  injection from provided poses (M4). M1 handles N=1 only.
- 3D Gaussians (M5), nested metric branch (M6), quantization (M7),
  LocalAI backend (M8).

## 3. Repository skeleton (created in M0)

```
depth-anything.cpp/
├── CMakeLists.txt                 # ggml static + PIC; DA_GGML_{CUDA,METAL,VULKAN}; DA_SHARED; DA_BUILD_TESTS/CLI
├── .gitmodules                    # third_party/ggml pinned to a commit SHA
├── third_party/{ggml, stb}/
├── include/
│   ├── da_capi.h                  # flat C ABI (load/free/info/last_error/abi_version)
│   └── da_gguf_keys.h             # AUTO-GENERATED from scripts/gguf_keys.py
├── scripts/
│   ├── gguf_keys.py               # ARCH, KV key names, tensor rename table (single source of truth)
│   ├── gen_gguf_keys_header.py    # gguf_keys.py → include/da_gguf_keys.h
│   ├── convert_da3_to_gguf.py     # HF safetensors → one self-contained GGUF
│   ├── download_model.py          # fetch depth-anything/DA3-BASE from HF
│   ├── dump_reference.py          # PyTorch forward-hook dumps → dumps/reference.gguf
│   ├── da3_reference.py           # build reference model + fixed input fixture
│   └── requirements.txt
├── src/
│   ├── common.hpp ggml_extend.hpp # logging macro; helper ops (layernorm, linear, gelu, rope)
│   ├── model_loader.{cpp,hpp}     # GGUF KV + tensors → map; GPU offload
│   ├── backend.{cpp,hpp}          # device select; persistent gallocr; compute(build_fn)->floats
│   ├── image_io.{cpp,hpp}         # stb load/save RGB
│   ├── preprocess.{cpp,hpp}       # lower_bound_resize + normalize + patch grid (M1 input)
│   ├── dino_backbone.{cpp,hpp}    # patch embed, pos-emb interp, block loop, output feats
│   ├── vit_block.{cpp,hpp}        # norm1/attn/ls1 + norm2/mlp/ls2
│   ├── attention.{cpp,hpp}        # qkv, qk_norm, rope apply, sdpa, proj
│   ├── rope2d.{cpp,hpp}           # 2D rotary embedding + position grid (pos / pos_nodiff)
│   ├── engine.{cpp,hpp}           # orchestration; TaskMode enum; M1 = backbone-only path
│   ├── da_capi.cpp
│   └── cli.{cpp,hpp}              # arg parse; `info` subcommand (M0)
├── examples/cli/main.cpp
├── tests/
│   ├── parity.hpp                 # load_baseline(gguf,name)->floats+shape; compare(atol,rtol)
│   ├── CMakeLists.txt             # da_add_test() macro; sets DA_TEST_GGUF / DA_TEST_BASELINE
│   ├── test_model_loader.cpp      # M0
│   ├── test_preprocess.cpp        # M1
│   ├── test_rope2d.cpp            # M1
│   ├── test_backbone.cpp          # M1 (the main gate)
│   └── test_capi.cpp              # M0
├── models/                        # depth-anything-base-f32.gguf (generated, gitignored)
├── dumps/                         # reference.gguf (generated, gitignored)
└── docs/
```

## 4. M0 — Scaffolding, converter, loader

### 4.1 GGUF metadata (KV) — written by converter, read by loader

`ARCH = "depthanything3"`. KV keys (all read by the loader; values come from the
HF `config.json`, never hardcoded in C++):

- General: `da3.arch`, `da3.checkpoint_name`, `da3.patch_size` (14),
  `da3.image_size`, `da3.task_caps` (bitmask of heads present in this GGUF).
- Backbone: `vit.embed_dim` (768), `vit.depth` (12), `vit.num_heads` (12),
  `vit.mlp_ratio` (4), `vit.num_register_tokens` (0), `vit.init_values`
  (layerscale), `vit.alt_start` (4), `vit.rope_start` (4), `vit.qknorm_start`
  (4), `vit.rope_freq` (100), `vit.cat_token` (true), `vit.interpolate_offset`
  (0.1), `vit.interpolate_antialias` (false), `vit.out_layers` ([5,7,9,11]),
  `vit.pos_embed_grid` (M, where pos_embed has M·M+1 rows).
- Preprocess: `img.mean`, `img.std` (DINOv2 normalization), and the resize
  policy parameters (`img.resize_mode="lower_bound"`, multiple-of-patch
  constraint). Exact constants are read from `preprocessor_config.json` during
  converter implementation; the converter asserts they exist.

`da_gguf_keys.h` is generated from `gguf_keys.py` so the C++ constants and the
Python writer can never drift.

### 4.2 Tensor naming (rename table in `gguf_keys.py`)

HF backbone names → GGUF names (prefix `vit.`). Confirm exact HF prefixes against
the downloaded checkpoint during implementation; expected mapping:

```
patch_embed.proj.{weight,bias}      → vit.patch_embed.{weight,bias}
cls_token                           → vit.cls_token
camera_token                        → vit.camera_token        # present (alt_start=4)
pos_embed                           → vit.pos_embed
blocks.{i}.norm1.{weight,bias}      → vit.blk.{i}.norm1.{w,b}
blocks.{i}.attn.qkv.{weight,bias}   → vit.blk.{i}.attn_qkv.{w,b}
blocks.{i}.attn.q_norm.{weight,bias}→ vit.blk.{i}.attn_qnorm.{w,b}   # qknorm_start=4 ⇒ present from blk 4
blocks.{i}.attn.k_norm.{weight,bias}→ vit.blk.{i}.attn_knorm.{w,b}
blocks.{i}.attn.proj.{weight,bias}  → vit.blk.{i}.attn_proj.{w,b}
blocks.{i}.ls1.gamma                → vit.blk.{i}.ls1
blocks.{i}.norm2.{weight,bias}      → vit.blk.{i}.norm2.{w,b}
blocks.{i}.mlp.fc1.{weight,bias}    → vit.blk.{i}.mlp_fc1.{w,b}
blocks.{i}.mlp.fc2.{weight,bias}    → vit.blk.{i}.mlp_fc2.{w,b}
blocks.{i}.ls2.gamma                → vit.blk.{i}.ls2
norm.{weight,bias}                  → vit.norm.{w,b}
```

The converter loads safetensors, casts bf16→f32, renames, writes via
`gguf.GGUFWriter`, and **errors on any unmapped tensor** that belongs to the
backbone. Non-backbone tensors (head, cam, gs) are written too (so the single
GGUF is complete for later milestones) but are not required by M0/M1 tests; the
rename table for those is filled in their milestones — for M0 we pass them
through with a documented `head.*`/`cam.*`/`gs.*` prefix or defer, decided at
implementation time without blocking M0's backbone gate.

### 4.3 `model_loader` and `backend`

Mirror locate-anything.cpp:
- `ModelLoader::load(path)` → `gguf_init_from_file` (alloc host tensors), read all
  KV into a `Config` struct, build `name → ggml_tensor*` map. Returns false if
  required backbone KV/tensors are missing.
- `ModelLoader::offload_weights(Backend&)` → CPU no-op (zero-copy host tensors);
  GPU mirrors tensors into a device buffer and repoints the map.
- `Backend` selects device (`DA_DEVICE` env override), owns a **persistent**
  `ggml_gallocr` reused across `compute()` calls, and exposes
  `compute(build_fn, out_floats)` plus `add_graph_input*` helpers.

### 4.4 `dump_reference.py` (the gold parity source)

Loads DA3-BASE via the reference repo, registers forward hooks, runs one fixed
RGB fixture image at one fixed resolution (a small multiple of 14, e.g.
`H=W=14·k` chosen so the run is fast on CPU), and dumps to `dumps/reference.gguf`:

- `input_image` (the exact normalized tensor fed to the net, shape `[1,1,3,H,W]`),
- `pos_embed_added` (tokens after cls + interpolated pos-embed, pre-block),
- `blk_{i}_out` for a few diagnostic layers (e.g. 0, 3, 4, 5),
- `feat_{L}` for each `L in out_layers=[5,7,9,11]` — the post-processed backbone
  features exactly as `get_intermediate_layers` returns them (width `2·768`,
  with the half-norm applied and token 0 / registers stripped),
- `cam_token_{L}` (the `out_x[:,:,0]` camera token per output layer),
- a `manifest.json` with H, W, k, tolerances.

These `feat_{L}` tensors are M1's parity targets and the **input contract for
M2's depth head**.

### 4.5 M0 parity gate

- `convert_da3_to_gguf.py` produces `models/depth-anything-base-f32.gguf`;
  re-reading it yields every expected KV and tensor (round-trip test).
- `test_model_loader.cpp`: loads the GGUF, asserts config values
  (embed_dim=768, depth=12, alt_start=4, …) and that all backbone tensors
  resolve.
- `da3-cli info --model …` prints the config and "loaded ok".
- `dump_reference.py` runs and writes `dumps/reference.gguf` + `manifest.json`.

## 5. M1 — DINOv2 backbone forward (monocular, N=1)

Reference: `/tmp/da3-src/src/depth_anything_3/model/dinov2/vision_transformer.py`
(`DinoVisionTransformer._get_intermediate_layers_not_chunked` and
`get_intermediate_layers`) plus `model/dinov2/layers/*` (Block, Attention,
PatchEmbed, RoPE2D, LayerScale) and `model/utils/attention.py`.

For N=1 (S=1) the following reference branches are inactive and **must not** be
implemented in M1 (they are M4): reference-view selection
(`x.shape[1] >= THRESH_FOR_REF_SELECTION` is false) and any view reordering.
However, the parts below **are active even at S=1** and are the parity-critical
subtleties:

1. **Patch embed.** `Conv2d(3→768, k=14, s=14)` over the resized image →
   `[768, H/14, W/14]` → flatten to `[N_patch, 768]`.
2. **CLS token + pos-embed.** Prepend `cls_token`; add
   `interpolate_pos_encoding`: **bicubic** interpolation of the `M·M` patch
   pos-embed grid to `(H/14, W/14)` using `scale_factor = (w0+0.1)/M,
   (h0+0.1)/M` (the DINOv2 "interpolate_offset" kludge), class pos-embed added to
   token 0. If input grid already equals the native grid, pos-embed is used
   as-is (no interpolation). Bicubic-with-scale-factor must match
   `torch.nn.functional.interpolate(mode="bicubic", antialias=false)`
   numerically — this is a known parity-risk op and gets its own sub-gate.
3. **RoPE positions.** Build two position sets via `PositionGetter`:
   - `pos` (local): real 2D grid positions, `+1`, with a zero "special" row
     prepended for token 0.
   - `pos_nodiff` (global): all-ones positions, with the same zero special row.
4. **Block loop** (i = 0..11), each `Block`:
   `x = x + ls1·attn(norm1(x), pos)` then `x = x + ls2·mlp(norm2(x))`,
   LayerNorm eps=1e-6, GELU MLP (ratio 4), LayerScale (`ls1`,`ls2` = gamma
   vectors). Attention: `qkv` linear → split heads (head_dim=64) → optional
   `q_norm`/`k_norm` (LayerNorm over head_dim, **active for i ≥ qknorm_start=4**)
   → apply RoPE for **i ≥ rope_start=4** → scaled-dot-product attention → `proj`.
   Attention scale = 1/sqrt(head_dim).
5. **Local vs global per layer (active at S=1).** For `i ≥ alt_start=4`:
   - **odd i** → "global" attention, RoPE uses `pos_nodiff` (`g_pos`);
   - **even i** (and all `i < 4`) → "local" attention, RoPE uses `pos` (`l_pos`),
     and `local_x` is updated to this layer's output.
   At S=1 the global/local token reshapes are identity, but the **RoPE position
   set differs**, so the two paths are not interchangeable — M1 must select
   `pos` vs `pos_nodiff` per layer exactly.
6. **Camera token injection at i == alt_start (4).** With no user pose,
   `cam_token = camera_token[:, :1]` (the reference slot; `S-1=0` source slots),
   and `x[:, :, 0] = cam_token` — i.e. **the position-0 token (formerly cls) is
   overwritten by the learned camera token at layer 4.** Must be reproduced.
7. **Output features at out_layers [5,7,9,11].** `out_x = cat([local_x, x],
   dim=-1)` (width `2·768=1536`). In `get_intermediate_layers`, because width ==
   `2·embed_dim`, the post-norm is applied **only to the second half**:
   `feat = cat([out_x[..., :768], norm(out_x[..., 768:])])`; then token 0 (and
   the 0 register tokens) are stripped → final per-layer feature
   `[N_patch, 1536]`. The camera token is `out_x[:, :, 0]` captured **before**
   stripping. These are exactly `feat_{L}` / `cam_token_{L}` in the dump.

### 5.1 ggml mapping notes

- Use `ggml_conv_2d` (or im2col+matmul) for patch embed.
- Bicubic pos-embed interpolation: precompute on host in the loader for the
  target resolution (the resolution is fixed per run), upload as a graph input —
  avoids implementing bicubic in ggml and removes a parity risk. (On-device
  bicubic is a possible M7 optimization, not now.)
- RoPE 2D: implement to match `RotaryPositionEmbedding2D` (frequency=100); verify
  in isolation (`test_rope2d`).
- q_norm/k_norm are LayerNorm over the per-head dim (64) with weight+bias.

### 5.2 M1 parity gates (per component, in order)

1. `test_preprocess`: resized+normalized input equals `input_image` from the
   dump (atol ~1e-4). Validates resize policy + normalization.
2. `test_rope2d`: rotary application on a fixture matches a tiny reference dump
   (bit-close).
3. `test_backbone` (main gate): run the full N=1 backbone on `input_image`;
   assert `pos_embed_added` and each `feat_{L}` (L∈{5,7,9,11}) and each
   `cam_token_{L}` match the dump within tolerance (target: max|Δ| ≤ ~1e-3 at
   f32, mean|Δ| ≤ ~1e-4). Diagnostic `blk_{i}_out` comparisons help localize the
   first diverging layer when it fails.

Passing `test_backbone` is the M1 completion criterion. At that point the engine
exposes a `backbone-features` debug path (not yet depth); the user-visible depth
CLI arrives in M2.

## 6. Components and boundaries (isolation check)

| Unit | Does what | Depends on |
|------|-----------|-----------|
| `gguf_keys.py` | single source of KV/tensor names + ARCH | — |
| `convert_da3_to_gguf.py` | HF → self-contained GGUF | gguf_keys, safetensors |
| `dump_reference.py` | gold per-component tensors | da3-src reference model |
| `ModelLoader` | GGUF → Config + tensor map | gguf, da_gguf_keys.h |
| `Backend` | device + gallocr + compute() | ggml |
| `preprocess` | image → normalized patch tensor | image_io |
| `rope2d` | position grids + rotary apply | ggml |
| `attention`/`vit_block` | one transformer block | rope2d, ggml_extend |
| `dino_backbone` | full N=1 forward → feats | loader, backend, block, rope2d |
| `engine` | TaskMode + orchestration (M1: backbone only) | all above |
| `da_capi`/`cli` | C ABI + `info` | engine, loader |

Each unit has a matching `test_*.cpp` gated on dumped references.

## 7. Testing strategy

- ctest, one test per component, each gated on `DA_TEST_GGUF`
  (models/...-f32.gguf) and `DA_TEST_BASELINE` (dumps/reference.gguf); skip
  (return 77) when env not set so the suite runs without the large artifacts.
- `parity.hpp` provides `load_baseline()` + `compare(label, atol, rtol)` printing
  `max|d|`/`mean|d|`/worst-index, copied/adapted from locate-anything.cpp.
- No mocking of ggml; tests run real graphs on CPU.

## 8. Risks / parity hot-spots

- **Bicubic pos-embed interpolation** — mitigated by precomputing on host to
  match PyTorch exactly.
- **RoPE2D formula + the pos/pos_nodiff split** — isolated gate + per-layer
  diagnostics.
- **Camera-token overwrite at layer 4 and the half-norm cat at output layers** —
  easy to miss; explicitly dumped and asserted.
- **Conv2d patch embed numerics** (im2col vs PyTorch conv) — verified via
  `pos_embed_added` gate.

## 9. Deliverables at end of M1

- Buildable `depth-anything.cpp` (CPU; CUDA/Metal/Vulkan toggles compile).
- `convert_da3_to_gguf.py` producing a self-contained DA3-BASE GGUF.
- `da3-cli info` working.
- Green ctest: loader, preprocess, rope2d, backbone parity, capi.
- Committed + pushed at the green gate, ready for M2 (DualDPT depth head), whose
  input contract is the validated `feat_{L}` tensors.
