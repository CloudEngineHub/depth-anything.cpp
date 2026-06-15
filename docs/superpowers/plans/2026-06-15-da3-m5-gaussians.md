# DA3 ggml port — M5 (3D Gaussians, DA3-GIANT) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Port the DA3-GIANT checkpoint (3D-Gaussian-capable) and produce 3D Gaussians matching the reference, with ply export and e2e verification. The GIANT also unlocks giant depth+pose (its head=DualDPT, cam_dec=CameraDec are the same modules, metadata-driven) and is the prerequisite for M6 (nested).

**Anchor:** `models/DA3-GIANT/` (downloaded, 5.4GB). Config: `vitg` backbone — 40 layers, embed 1536, num_heads 24, head_dim 64, **SwiGLU FFN** (the only new backbone op), alt_start/rope_start/qknorm_start=13, out_layers [19,27,33,39], cat_token. Head DualDPT dim_in 3072, features 256, out_channels [256,512,1024,1024]. cam_dec dim_in 3072. **gs_head GSDPT** dim_in 3072, output_dim 38, features 256. **gs_adapter** sh_degree 2, pred_offset_depth+xy, no pred_color — **no learned weights** (pure geometry). NOTE: giant CPU inference is slow (~40 layers); dumps + C++ runs take minutes — that's expected.

**SwiGLU FFN** (`SwiGLUFFNFused`): `w12 = Linear(1536→8192)`, `w3 = Linear(4096→1536)`. forward: `x12=w12(x); x1,x2=x12.chunk(2,-1); hidden=silu(x1)*x2; return w3(hidden)`. (hidden=4096.) Tensor names `blocks.N.mlp.w12.{weight,bias}`, `blocks.N.mlp.w3.{weight,bias}`.

**GSDPT** (`gsdpt.py`, subclass of DPT — the single-head DPT, NOT DualDPT): same stages/projects/resize/scratch/fuse/output_conv1 as DPT, PLUS `images_merger = Sequential(Conv2d(3,m/4,3,1,1), act, Conv2d(m/4,m/2,3,1,1), act, Conv2d(m/2,m,3,1,1))` where m=features//2=128 (merger_out_dim = features//2 since feature_only=False). forward: after `output_conv1` and the 224-interpolate, `fused = fused + images_merger(images)` (images = the input RGB, in [0,1], resized to 224), then `+pos_embed`, then `output_conv2` → 38 channels → `activate_head_gs` → raw_gs (37) + raw_gs_conf (1). out_dim 38, conf_dim=1, activation/conf_activation per config.

**gs_adapter** (host geometry, no weights): see gs_adapter.py forward — unproject (depth+xy offset) to world rays via cam2world+intrinsics, gs_means = origins+directions*depth; scales = sigmoid→[min,max]*depth*multiplier; quaternion xyzw normalized → world (cam_quat_xyzw_to_world_quat_wxyz); SH (sh_degree 2, d_sh=9) masked + rotate_sh to world; opacities = map_pdf_to_opacity(raw_gs_conf). Outputs Gaussians(means, scales, rotations_wxyz, harmonics, opacities).

---

## Task list

### M5-T1: GIANT converter + dump (SwiGLU + gs_head)
- [ ] gguf_keys.py: add SwiGLU renames `blocks.N.mlp.w12→vit.blk.N.mlp_w12`, `mlp.w3→vit.blk.N.mlp_w3`; add `vit.ffn_type` KV ("mlp"|"swiglu"); add gs_head renames (`gs_head.images_merger.{0,2,4}→gs.merger.{i}`, `gs_head.projects/resize/scratch.*→gs.*` mirroring the head table) + gs KV (gs.output_dim=38, gs.sh_degree=2, gs.features=256, gs.out_channels, scale_min/max, pred flags). Converter: detect ffn_type from module (SwiGLU has mlp.w12); write giant to `models/depth-anything-giant-f32.gguf`. Quantize allowlist (M7) extended to mlp_w12/w3 later.
- [ ] dump (`scripts/dump_giant.py`, separate): giant backbone `feat_g_{19,27,33,39}` [256,3072] + `cam_g_{L}`; `depth_g`,`extrinsics_g`,`intrinsics_g` (giant depth+pose); `raw_gs` (37,224,224) + `gs_conf`; and the final `net(x, infer_gs=True)` gaussians (means/scales/rotations/harmonics/opacities) — flattened. Use the fixed 224 fixture. Verify shapes.
Gate: giant GGUF roundtrips; backbone/gs dumps produced.

### M5-T2: SwiGLU block + GIANT backbone + depth+pose parity
- [ ] Add SwiGLU path to `vit_block` (config `ffn_type`: if swiglu, `w3(silu(x1)*x2)` from `w12`; `da::silu` via `ggml_silu`). Loader reads `ffn_type`, `mlp_w12/w3` names when swiglu.
- [ ] Gate giant backbone `feat_g_{L}` (the existing forward, metadata-driven for 40 layers/1536/alt_start13) at 2e-3. Then verify giant depth+pose via existing DptHead/CamPose (metadata dims) vs `depth_g`/`extrinsics_g`/`intrinsics_g` — a free win confirming the head/cam generalize.
Gate: giant backbone + depth + pose <2e-3 (slow CPU OK).

### M5-T3: GSDPT head → raw_gs parity
- [ ] `src/gs_head.{cpp,hpp}`: reuse dpt_blocks; implement the DPT main path with `images_merger` add + output_dim 38 + `activate_head_gs` (split 37 raw + 1 conf; activations). Gate raw_gs + gs_conf vs dump at 2e-3.
Gate: raw_gs/gs_conf <2e-3.

### M5-T4: gs_adapter (host geometry) + ply export + e2e
- [ ] `src/gs_adapter.{cpp,hpp}`: host math — world-ray unproject, scales, quat xyzw→world wxyz, SH mask+rotate, opacities. Gate gaussians (means/scales/rotations/harmonics/opacities) vs dump at 2e-3 (read geometry helpers get_world_rays/sample_image_grid/rotate_sh/cam_quat_xyzw_to_world_quat_wxyz/map_pdf_to_opacity from /tmp/da3-src utils).
- [ ] ply export (`src/ply_export`), `Engine::reconstruct(image)` → gaussians + `da3 reconstruct --ply out.ply`. e2e vs `net(x, infer_gs=True)`. docs/M5-e2e.md.
Gate: gaussians match reference; ply written; e2e vs original.

---

## Notes
- gs_adapter needs geometry utils from `/tmp/da3-src/src/depth_anything_3/utils/geometry.py` (get_world_rays, sample_image_grid, affine_inverse) + `model/utils/transform.py` (cam_quat_xyzw_to_world_quat_wxyz) + `utils/sh_helpers.py` (rotate_sh) + `utils/geometry.py` (map_pdf_to_opacity). Read them exactly.
- The giant is the M6 prerequisite (M6 = nested giant+metric branch + alignment).
- Quantize allowlist (M7) should add `vit.blk.*.mlp_w12/w3.weight` for the giant — update in M5-T2 or a follow-up.
- Giant dumps are slow; keep the fixed 224 square fixture.
