# DA3 ggml port — M2 (DualDPT depth head + end-to-end depth) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Implement the DualDPT **main depth path** in C++/ggml so the port produces a depth map + confidence numerically equal to the reference, and wire an end-to-end `da3 depth <image>` CLI plus an e2e parity check vs the original model. (Aux head = ray/sky is M3; only the main depth + depth_conf are in M2.)

**Architecture (traced from `/tmp/da3-src/src/depth_anything_3/model/{dualdpt,dpt}.py` + `model/utils/head_utils.py`):**
Input = the 4 validated backbone feature tensors `feat_{5,7,9,11}` (each `[256 patches × 1536]`) from `Engine`/`DinoBackbone`. For DA3-BASE the head config is `dim_in=1536, output_dim=2, features=128, out_channels=[96,192,384,768], pos_embed=true, down_ratio=1, patch_size=14`. Forward (main path):
1. For stage s in 0..3 (intermediate_layer_idx fixed (0,1,2,3) → uses feat_5,7,9,11 in order): take patches → `norm` (LayerNorm 1536) → permute to `[1536, ph=16, pw=16]` → `projects[s]` (Conv2d 1×1, 1536→out_channels[s]) → `+ UV_pos_embed*0.1` → `resize_layers[s]`: s0 ConvTranspose2d k4 s4 (→64×64), s1 ConvTranspose2d k2 s2 (→32×32), s2 Identity (16×16), s3 Conv2d k3 s2 p1 (→8×8).
2. `_fuse`: `layer{1..4}_rn` (Conv2d 3×3 pad1, **no bias**, out_channels[i]→128) → l1_rn(64²),l2_rn(32²),l3_rn(16²),l4_rn(8²). Then top-down: `out=refinenet4(l4_rn, size=16²)`; `out=refinenet3(out,l3_rn,size=32²)`; `out=refinenet2(out,l2_rn,size=64²)`; `out=refinenet1(out,l1_rn)` (scale_factor 2 → 128²). Then `out=output_conv1(out)` (Conv2d 3×3 pad1, 128→64).
   - `FeatureFusionBlock.forward(*xs,size)`: `y=xs[0]`; if has_residual and 2 inputs: `y = y + resConfUnit1(xs[1])`; `y=resConfUnit2(y)`; bilinear-interpolate(align_corners=True) to `size` (or scale_factor 2 if none); `y=out_conv(y)` (Conv2d 1×1 128→128). refinenet4 has **no** resConfUnit1 (has_residual=False).
   - `ResidualConvUnit.forward(x)`: `out=relu(x); out=conv1(out); out=relu(out); out=conv2(out); return out+x` (convs 3×3 pad1, with bias).
3. `fused = custom_interpolate(out, (224,224), bilinear, align_corners=True)`; `fused += UV_pos_embed*0.1` (channels=64).
4. `main_logits = output_conv2(fused)`: Conv2d 3×3 pad1 (64→32) → ReLU → Conv2d 1×1 (32→2). → `[2,224,224]`.
5. depth = `exp(logits[ch0])`, depth_conf = `exp(logits[ch1]) + 1`. Both `[224,224]`.

UV pos-embed (`_add_pos_embed`, ratio 0.1): `create_uv_grid(pw,ph,aspect=W/H)` → `position_grid_to_embed(grid, C, omega_0=100)` (sincos) → `*0.1`, broadcast-add over batch. Deterministic per (pw,ph,C) → **precompute on host**, add as graph input. `custom_interpolate`/FFB upsample = `F.interpolate(bilinear, align_corners=True)` → `ggml_interpolate(..., GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS)`.

**Tensor names (HF `model.head.*`, prefix stripped → GGUF `head.*`):**
```
norm.{weight,bias} -> head.norm.{w,b}
projects.{0..3}.{weight,bias} -> head.proj.{i}.{w,b}
resize_layers.0.{weight,bias} -> head.resize.0.{w,b}   (ConvTranspose k4s4)
resize_layers.1.{weight,bias} -> head.resize.1.{w,b}   (ConvTranspose k2s2)
resize_layers.3.{weight,bias} -> head.resize.3.{w,b}   (Conv k3s2p1)   [.2 is Identity, no weights]
scratch.layer{1..4}_rn.weight -> head.scratch.layer{i}_rn.weight   (no bias)
scratch.refinenet{1..4}.resConfUnit1.conv1.{weight,bias} -> head.scratch.rn{i}.rc1.c1.{w,b}   (rn4 has none)
scratch.refinenet{1..4}.resConfUnit1.conv2.{weight,bias} -> head.scratch.rn{i}.rc1.c2.{w,b}
scratch.refinenet{1..4}.resConfUnit2.conv1.{weight,bias} -> head.scratch.rn{i}.rc2.c1.{w,b}
scratch.refinenet{1..4}.resConfUnit2.conv2.{weight,bias} -> head.scratch.rn{i}.rc2.c2.{w,b}
scratch.refinenet{1..4}.out_conv.{weight,bias} -> head.scratch.rn{i}.out.{w,b}
scratch.output_conv1.{weight,bias} -> head.scratch.out1.{w,b}
scratch.output_conv2.0.{weight,bias} -> head.scratch.out2a.{w,b}   (Conv 3x3 64->32)
scratch.output_conv2.2.{weight,bias} -> head.scratch.out2b.{w,b}   (Conv 1x1 32->2)
```
(The aux-head tensors `refinenet*_aux`, `output_conv1_aux*`, `output_conv2_aux*` exist in the checkpoint; M2's rename table SKIPS them — they're added in M3. The converter must still error only on *unmapped main-path/head* tensors it's told to expect, so M2 changes the converter to: write backbone + main-head tensors, and explicitly skip a known aux-prefix set.)

**Parity gates:** every component vs PyTorch dumps. New dumps (from a hook on the real head): `head_depth` `[224,224]`, `head_depth_conf` `[224,224]`, plus intermediate `head_stage{0..3}` (post-resize), `head_fused` (post output_conv1, pre-interpolate), `uv_embed_64` (the 224² 64-ch UV embed) for isolation. Tolerance 2e-3 rel/abs at f32.

**Tech stack:** same as M0/M1. New ggml ops: `ggml_conv_2d` (stride/pad), `ggml_conv_transpose_2d_p0`, `ggml_interpolate`, `ggml_exp`, `ggml_relu`.

---

## Task list

### Task 1: Extend converter + dump for the head
**Files:** `scripts/gguf_keys.py` (add `rename_head`), `scripts/convert_da3_to_gguf.py` (write head tensors; skip aux), `scripts/dump_reference.py` (dump head depth/conf + intermediates), `scripts/test_convert_roundtrip.py` (assert head tensors).

- [ ] **Step 1:** Add `rename_head(name)` to `scripts/gguf_keys.py` implementing the table above (return None for aux/unknown). Add a `HEAD_AUX_PREFIXES` set or regex (`refinenet\d+_aux`, `output_conv1_aux`, `output_conv2_aux`) recognized as "intentionally skipped".
- [ ] **Step 2:** In `convert_da3_to_gguf.py`, after the backbone loop add a head loop over `net.head.named_parameters()`: strip nothing (names are already `norm.*`, `projects.*`, `scratch.*`), map via `rename_head`; write mapped tensors; collect skipped; **fail only if a skipped name is NOT matched by the aux-skip set** (so a genuinely-unmapped main tensor still errors, but aux tensors are silently skipped). Print `head_tensors=N skipped_aux=M`. Write head KV: `head.features`(128), `head.out_channels`([96,192,384,768]), `head.output_dim`(2), `head.patch_size`(14), `head.pos_embed`(true), `head.down_ratio`(1), `head.activation`("exp"), `head.conf_activation`("expp1") — add these keys to `K.KV`.
- [ ] **Step 3:** Regenerate the GGUF; in `test_convert_roundtrip.py` assert presence of `head.norm.weight`, `head.proj.0.weight`, `head.scratch.rn4.rc2.c1.weight`, `head.scratch.rn1.rc1.c1.weight`, `head.scratch.out1.weight`, `head.scratch.out2b.weight`, and that the total backbone+head tensor count matches a recomputed expected value (don't hardcode 207; compute backbone+head).
- [ ] **Step 4:** In `dump_reference.py`, after the backbone dump, run the full `net.forward(x, ...)` (or call `net.head(feats, H, W, patch_start_idx=0)` directly with the feats from the backbone) under no_grad, and capture: `head_depth`, `head_depth_conf` (the head's `depth`/`depth_conf` outputs, squeezed to [224,224]), and intermediates via hooks: `head_stage{0..3}` (output of each `resize_layers[s]`), `head_fused` (output of `output_conv1` inside `_fuse`), `uv_embed_64` (compute `position_grid_to_embed(create_uv_grid(224,224,1.0),64)` directly). Confirm `head_depth` shape (224,224), finite, positive (exp). Dump them into `dumps/reference.gguf` (flattened f32) and add shapes to manifest.
- [ ] **Step 5:** Run converter + dump; verify; commit scripts (not artifacts).

Gate: roundtrip test passes with head tensors; dump has head_depth/conf + intermediates; head_depth positive & finite.

### Task 2: Host UV positional embedding
**Files:** `src/uv_posembed.{hpp,cpp}`, `tests/test_uv_posembed.cpp`.
- [ ] Implement `std::vector<float> uv_pos_embed(int pw, int ph, int C, float aspect, float omega0=100, float ratio=0.1)` returning a `[C, ph, pw]` host buffer (ggml-order: channel-major? choose layout that matches how it's added to the conv feature map `[pw,ph,C]` in ggml — i.e. element (c,y,x); document and match). Reproduce `create_uv_grid` + `position_grid_to_embed` + `make_sincos_pos_embed` exactly (note `meshgrid(..., indexing="xy")` and the emb = cat([sin,cos]) per axis, then cat([emb_x,emb_y])).
- [ ] Gate `test_uv_posembed` against the dumped `uv_embed_64` (224×224×64) at 1e-5. (The dump stored it pre-`*ratio`; multiply or match accordingly — be consistent.)
- [ ] Commit.

Gate: UV embed matches reference <1e-5.

### Task 3: DPT building blocks (conv/convT/interpolate/RCU/FFB)
**Files:** `src/dpt_blocks.{hpp,cpp}`, `tests/test_dpt_blocks.cpp`.
- [ ] Helpers in ggml: `conv2d(ctx, w, b, x, stride, pad)` (wrap `ggml_conv_2d` + bias add over channel dim), `conv_transpose2d_p0(ctx, w, b, x, stride)`, `interp_bilinear_ac(ctx, x, out_h, out_w)` (`ggml_interpolate` BILINEAR|ALIGN_CORNERS), `residual_conv_unit(ctx, x, c1w,c1b,c2w,c2b)` (relu→conv1→relu→conv2→+x), `feature_fusion(ctx, top, lateral_or_null, rc1*, rc2*, outconv*, out_h,out_w)`.
- [ ] Gate at least the FeatureFusionBlock or a single conv against a tiny dumped fixture (add a `ffb_in`/`ffb_out` fixture to dump_reference.py using one real refinenet on random input), OR defer numeric check to Task 4's full-head gate if a standalone fixture is too fiddly — but PREFER a standalone conv2d gate (dump one `projects[0]` output `head_proj0` and compare) to isolate conv-layout bugs early.
- [ ] Commit.

Gate: conv/FFB block matches reference fixture (or, if deferred, Task 4 isolates).

### Task 4: Full DualDPT main forward + depth parity gate
**Files:** `src/dpt_head.{hpp,cpp}`, `tests/test_dpt_head.cpp`.
- [ ] `DptHead(ModelLoader&, Backend&)` with `bool depth(const std::vector<std::vector<float>>& feats, int H, int W, std::vector<float>& depth_out, std::vector<float>& conf_out)`. Build the full main-path graph per the architecture above, precomputing all UV embeds (stage channels 96/192/384/768 at 16², and 64 at 224²) on host. Apply `exp`/`expp1` activations on host or via ggml_exp.
- [ ] Gate `test_dpt_head` against `head_depth` + `head_depth_conf` at 2e-3, using the dumped `feat_{5,7,9,11}` as input (load them from the dump as the head input, so the head is tested in isolation from the backbone). Use the `head_stage*`/`head_fused` dumps for layer-isolation debugging if it fails.
- [ ] Commit.

Gate: depth + depth_conf match reference <2e-3 (THE M2 component gate).

### Task 5: Engine depth path + CLI + export
**Files:** `src/engine.{hpp,cpp}` (add `depth()` running backbone→head from a raw image), `src/depth_export.{hpp,cpp}` (PNG colormap + PFM + raw npz-like .bin), `examples/cli/main.cpp` (+`depth` subcommand), `include/da_capi.h`/`src/da_capi.cpp` (+`da_capi_depth_path`), `tests/test_engine_depth.cpp`.
- [ ] `Engine::depth(image_path, depth_out, conf_out, H, W)`: load image → preprocess → backbone_features → DptHead.depth. CLI `da3 depth --model … --input img --output depth.pfm --png depth.png`.
- [ ] Gate `test_engine_depth` end-to-end: feed the dumped `raw_image`, run the full Engine depth, compare to `head_depth` (should match since same input path) <2e-3.
- [ ] Commit.

Gate: full image→depth pipeline matches reference depth <2e-3.

### Task 6: End-to-end verification vs original (the goal's e2e check)
**Files:** `scripts/e2e_verify.py`, `docs/M2-e2e.md`.
- [ ] `scripts/e2e_verify.py <image>`: runs the REAL DA3 model's depth on an arbitrary image (via `scripts.da3_reference` + the head), runs the C++ CLI `da3 depth` on the same image, and compares the two depth maps (max/mean abs rel diff, correlation). Use a real test image (download one, or a checkerboard/gradient). Report PASS if median rel error < 1% on a real unseen image (looser than the fixture gate because real-image preprocessing/resize parity for non-square images is an M2-noted gap — use a SQUARE image to stay on the verified path, or implement the real resize first).
- [ ] Document results in `docs/M2-e2e.md` with the numbers.
- [ ] Commit.

Gate: C++ depth vs original PyTorch depth agree on a real image within the documented tolerance — the end-to-end verification milestone for the depth path.

---

## Notes / risks
- **conv weight layout:** torch Conv2d weight `(OC,IC,KH,KW)` loads as ggml ne `[KW,KH,IC,OC]` — exactly `ggml_conv_2d`'s expected kernel layout (verified in M1 for patch_embed). ConvTranspose2d weight is `(IC,OC,KH,KW)` in torch — confirm the ggml `conv_transpose_2d_p0` kernel layout and transpose if needed (isolate via `head_stage0` which is the ConvTranspose output).
- **interpolate align_corners:** must use the ALIGN_CORNERS flag; a mismatch shows as a smooth but shifted depth — isolate via `head_fused` (pre final interpolate) vs `head_depth`.
- **square fixture:** 224² keeps us on the M1-verified preprocess/pos path; real non-square resize remains the M2-noted deferred item (Task 6 uses a square image).
- **bias add after conv:** ggml conv output is `[W,H,OC,N]`; bias `[OC]` must broadcast over W,H — reshape bias to `[1,1,OC,1]` for the add.
