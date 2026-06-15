# DA3 ggml port — M4 (multi-view cross-attention) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Generalize the DINOv2 backbone to S>1 views with cross-view attention so multi-image depth+pose matches the reference, gated on an S=2 fixture. Phase 1 = S=2 (no reference-view selection, no cam_enc). Phase 2 (follow-on) = S≥3 reference-view selection.

**Why S=2 first:** `THRESH_FOR_REF_SELECTION=3`, so for S=2 the reference-view selection branch (vision_transformer.py:314-321) is skipped, and `cam_enc` is only used when input camera poses are PROVIDED (not the case at inference). So S=2 isolates the genuinely-new mechanic — cross-view global attention + ref/src camera-token injection — with no other complications. **No new GGUF weights needed** (camera_token + all blocks already converted in M0).

**The S>1 forward (from vision_transformer.py `_get_intermediate_layers_not_chunked` + `process_attention`):**
Input x = `[B, S, 3, H, W]`. prepare_tokens → `[B, S, N=257, C=768]` (each view: cls + 256 patches; pos-embed added per view identically). Two RoPE position sets over (B,S,N): `pos` (local: real 2D per-patch +1, special row 0) and `pos_nodiff` (global: all-ones patches + special). Block loop i=0..11:
- **local layers** (`i < alt_start` OR (`i>=alt_start` and even)): `process_attention(x, "local")` = rearrange `b s n c -> (b s) n c` → per-view attention over each view's 257 tokens, rope = `pos`; rearrange back. `local_x = x`.
- **global layers** (`i>=alt_start=4` and odd): `process_attention(x, "global")` = rearrange `b s n c -> b (s n) c` → ONE attention over all S·257 tokens jointly (cross-view mixing), rope = `pos_nodiff`; rearrange back.
- **cam-token injection at i==alt_start=4** (no input poses): `ref_token = camera_token[:, :1]` (slot 0), `src_token = camera_token[:, 1:].expand(B, S-1, -1)` (slot 1, repeated for views 1..S-1); `cam_token = cat([ref_token, src_token], dim=1)` `[B,S,768]`; `x[:, :, 0] = cam_token` (token 0 of EVERY view ← its camera token: view 0 gets ref slot, views ≥1 get src slot).
- **out_layers [5,7,9,11]:** `out_x = cat([local_x, norm? no — raw cat], -1)` then in get_intermediate_layers the second-half gets `vit.norm`; `feat_L = [B,S,256,1536]` (cat[local_x, norm(x)], token0 stripped per view), `cam_token_L = out_x[:,:,0]` `[B,S,1536]` (raw). For S=2 these are 2 views each.

**Key ggml change:** the M1 backbone graph assumed S=1 (token tensor `[C, N]`). M4 carries S as a real dim. Suggested layout: tokens `[C=768, N=257, S]` (ne0=C, ne1=N, ne2=S). Local attention = run the existing single-view attention independently per S-slice (loop s, or batch over ne2). Global attention = reshape to `[C, N·S]` (one attention over all tokens), then back. The per-block `vit_block`/`attention` already work on `[C, N]`; wrap them to handle the S dim (local: per-slice; global: flattened). Reuse rope2d (pos/pos_nodiff built over N·S for global, N per-view for local — careful: global rope uses pos_nodiff which is per-(s,n); since all patches map to (1,1) it's view-independent, so the S·N global rope table = the per-view pos_nodiff tiled S times).

**Parity:** dump S=2 backbone `feat_{5,7,9,11}` `[2,256,1536]` + `cam_token_{5,7,9,11}` `[2,1536]` from the real backbone on a 2-image input, plus full `extrinsics`/`intrinsics` `[2,3,4]`/`[2,3,3]` and `depth` `[2,224,224]`. Gate the C++ S=2 backbone + head + pose per-view. Tol 2e-3.

---

## Task list (Phase 1: S=2)

### M4-T1: multi-view reference dump (S=2)
**Files:** `scripts/dump_reference.py` (add S=2 fixtures behind a flag/section), `scripts/da3_reference.py` (`fixed_input_multiview(S=2)`).
- [ ] Add `fixed_input_multiview(S)` returning `(x[1,S,3,224,224], raws[S])` deterministic (two different structured 224² images).
- [ ] Dump S=2: run `bb.get_intermediate_layers(x_mv, n=[5,7,9,11], ...)` → `feat_mv_{L}` `[2,256,1536]`, `cam_mv_{L}` `[2,1536]`; run `net(x_mv)` → `depth_mv` `[2,224,224]`, `extrinsics_mv` `[2,3,4]`, `intrinsics_mv` `[2,3,3]`, `raw_mv_0/1`. Write to a SEPARATE `dumps/reference_mv.gguf` (keep the S=1 dump clean). Add manifest.
- [ ] Verify shapes; both views' extrinsics orthonormal. Commit script.

### M4-T2: backbone S>1 generalization + S=2 backbone gate
**Files:** `src/dino_backbone.{hpp,cpp}` (S-aware forward), `tests/test_backbone_mv.cpp`.
- [ ] Generalize `DinoBackbone::forward` (or add `forward_mv(views_chw[S], H, W, feats[L][S], cam_tokens[L][S])`). Implement local (per-view) / global (cross-view, S·N tokens) attention selection per layer; cam-token ref/src injection; out-layer cat+half-norm per view. Reuse attention/vit_block/rope2d.
- [ ] Gate `test_backbone_mv` vs `feat_mv_{L}` + `cam_mv_{L}` (both views) at 2e-3, using `reference_mv.gguf` (new `DA_TEST_BASELINE_MV` env). Use S=1 path unchanged (don't regress test_backbone).
- [ ] Commit.

Gate: S=2 per-view feat/cam match reference <2e-3 (THE M4 gate). Debug via the global-attention reshape (most likely snag) and the src-token injection.

### M4-T3: multi-view engine + CLI + e2e
**Files:** `src/engine.{hpp,cpp}` (`depth_pose_multi(images[])`), CLI (`da3 depth --inputs a.png,b.png` or repeated `--input`), export (per-view depth/pose), `tests/test_engine_mv.cpp`, `scripts/e2e_verify.py` (+MV), `docs/M4-e2e.md`.
- [ ] `Engine::depth_pose_multi(std::vector<Image>)` → per-view depth + pose (backbone_mv → head per view + cam_dec per view's cam token).
- [ ] Gate end-to-end on `raw_mv_0/1` vs `depth_mv`/`extrinsics_mv`/`intrinsics_mv` (5e-3).
- [ ] e2e: C++ multi-image vs real `net(x_mv)` on two real images; document `docs/M4-e2e.md`.
- [ ] Commit.

Gate: multi-view image→(depth,pose) matches original e2e.

---

## Phase 2 (follow-on, separate plan): S≥3 reference-view selection
`select_reference_view` (4 strategies; default `saddle_balanced`), `reorder_by_reference`, `restore_original_order` at layer `alt_start-1`. Needs an S=4 fixture + the saddle-metric computation. Deferred until S=2 is solid.

## Notes / risks
- **Global attention layout:** the cross-view reshape `b s n c -> b (s n) c` means token order is view-major (view0's 257 tokens, then view1's). Match exactly when flattening `[C,N,S]`→`[C,N·S]` (S as the outer/slower dim → ne: [C, N, S] contiguous gives N fastest within a view, view slower = view-major when read as [C, N·S]). Verify.
- **rope for global:** pos_nodiff per-(s,n); since patches all map to (1,1), the global rope table is the per-view nodiff tiled S times (token 0 of each view = special (0,0)). Build it over N·S.
- **src-token expand:** for S=2, views[0] token0 ← camera_token slot0, views[1] token0 ← camera_token slot1. For S>2, views[1..] all share slot1.
- multi-view depth at native resolution still uses the M2-noted 224 square path (real resize deferred).
