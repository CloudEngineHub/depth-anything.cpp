# DA3 ggml port — M7 (quantization) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Add q8_0/q6_k/q5_k/q4_k quantization of the matmul weights, a `da3 quantize` CLI, and verify the quantized models produce depth+pose near-lossless at q8 and bounded-degradation at q4 vs f32 — keeping the depth/pose/multi-view e2e correctness from M0–M4.

**Why it works with no loader change:** the loader reads tensors via `gguf_init_from_file(no_alloc=false)` (any ggml type loads), and `da::linear` = `ggml_mul_mat`, which consumes quantized weights transparently. So only matmul weights are quantized; **conv kernels stay f32** (ggml_conv_2d/im2col needs f32/f16 kernels), as do norms/biases/pos_embed/cls/camera_token.

**Quantize allowlist (`should_quantize`):** only 2D Linear/matmul weights —
`vit.blk.{i}.attn_qkv.weight`, `vit.blk.{i}.attn_proj.weight`, `vit.blk.{i}.mlp_fc1.weight`, `vit.blk.{i}.mlp_fc2.weight`, and `cam.bb0.weight`, `cam.bb2.weight`, `cam.fc_t.weight`, `cam.fc_q.weight`, `cam.fc_fov.weight`. Everything else (head.* convs, norms, biases, pos_embed, cls_token, camera_token) stays f32. All allowlisted weights have ne[0] ∈ {768,1536,3072} (÷256 ⇒ k-quant OK).

Model is from the sibling `~/_git/locate-anything.cpp/src/quantize.{hpp,cpp}` (dequant→`ggml_quantize_chunk`→`gguf_add_tensor`→`gguf_write_to_file`).

---

## Task list

### M7-T1: quantizer + `da3 quantize` CLI
**Files:** `src/quantize.{hpp,cpp}`, `src/cli.*` + `examples/cli/main.cpp` (quantize subcommand), `tests/test_quantize.cpp`.
- [ ] Copy `~/_git/locate-anything.cpp/src/quantize.{hpp,cpp}`, rename namespace `la`→`da`, and replace `should_quantize` with the DA3 allowlist above (regex `^vit\.blk\.\d+\.(attn_qkv|attn_proj|mlp_fc1|mlp_fc2)\.weight$` OR `^cam\.(bb0|bb2|fc_t|fc_q|fc_fov)\.weight$`). Keep the f16/q8_0/q6_k/q5_k/q4_k parse + the dequant/quantize-chunk core. Add `src/quantize.cpp` to `DA_SOURCES`.
- [ ] CLI: `da3 quantize <in.gguf> <out.gguf> <type>` → calls `da::quantize_gguf`. Print result size.
- [ ] `tests/test_quantize.cpp`: quantize the f32 GGUF (env DA_TEST_GGUF) to q8_0 in a temp path, load it with `ModelLoader`, assert config reads back correct and a quantized tensor (`vit.blk.0.attn_qkv.weight`) has type Q8_0 while a conv (`vit.patch_embed.weight`) stays F32. SKIP (77) if no GGUF.
- [ ] Build + run; produce `models/depth-anything-base-q8_0.gguf` and `-q4_k.gguf`; print sizes (q8 ≈ ¼–½ of f32, q4 smaller). Commit (no gguf artifacts).

Gate: quantizer runs; quantized GGUF loads; matmul weights are q-typed, convs stay f32.

### M7-T2: accuracy verification (near-lossless q8, bounded q4)
**Files:** `tests/test_quantize_accuracy.cpp`, `scripts/quant_verify.py` (optional), `docs/M7-accuracy.md`.
- [ ] `tests/test_quantize_accuracy.cpp`: for each of {q8_0, q4_k} (quantize on the fly from DA_TEST_GGUF to temp), run `Engine::depth_pose` on the dumped `raw_image`, compare depth to the f32 `head_depth` (from DA_TEST_BASELINE) — assert q8_0 max|d| < 2e-2 (near-lossless) and q4_k correlation > 0.99 (bounded). Also compare extrinsics (pose should be robust). Report the actual numbers. SKIP if envs unset.
- [ ] `docs/M7-accuracy.md`: table of {f32,q8_0,q6_k,q5_k,q4_k} → file size, depth max|d| / corr vs f32, pose ext max|d|. Reproduce command.
- [ ] Build + run + commit.

Gate: q8_0 depth near-lossless (<2e-2), q4_k corr>0.99; documented size/accuracy table.

---

## Notes
- `da::ModelLoader::offload_weights` is a CPU no-op, so quantized weights stay mmap'd host tensors; `ggml_mul_mat` dequantizes on the fly. No GPU path here (M-future).
- Quantization does NOT touch the head convs, so depth-head precision is unchanged; degradation comes only from the backbone attention/mlp + cam_dec matmuls — expect q8 to be ~lossless for depth.
- Keep `models/*.gguf` gitignored.
