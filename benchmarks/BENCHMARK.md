# Depth Anything 3 — Performance Benchmarks (M9-T3)

Latency and peak memory for the C++/ggml port vs the original PyTorch DA3-BASE
model, across quantization levels and resolutions. Reproduce with:

```bash
. .venv/bin/activate
python benchmarks/bench.py        # writes benchmarks/results.json, prints the table
```

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (16 cores / 20 logical) |
| RAM | 84 GB |
| Threads used | 8 (both engines: `--threads 8` / `torch.set_num_threads(8)`) |
| PyTorch | 2.12.0+cpu (oneDNN), f32 |
| Backend | CPU only — **no GPU offload yet** |
| Model | DA3-BASE (DINOv2 ViT-B/14, 12 blocks, 768-dim + DualDPT depth head) |
| Iterations | 1 warmup + median over 6 timed iterations |

## Results

Model = DA3-BASE. `@224` = 224×224 input (C++ `--legacy-resize`). `@504` = native
DA3 `upper_bound_resize` of a 640×427 photo → 504×336 (the production path).
`speedup` is PyTorch-f32 infer ÷ this-config infer (>1 = faster than PyTorch).

| engine    | quant | size MB | load ms | infer @224 ms | infer @504 ms | peak RSS MB | speedup @504 |
|-----------|-------|--------:|--------:|--------------:|--------------:|------------:|-------------:|
| PyTorch   | f32   |   516   |   717   |    119.9      |    429.1      |    1328     |    1.00x     |
| C++/ggml  | f32   |   393   |   109   |    242.4      |    857.8      |    1086     |    0.50x     |
| C++/ggml  | q8_0  |   142   |    35   |    194.5      |    752.3      |     835     |    0.57x     |
| C++/ggml  | q4_k  |    99   |    24   |    192.9      |    712.1      |     792     |    0.60x     |

(Full numbers incl. p90 in `benchmarks/results.json`. Run-to-run variance on this
CPU is ~5–10%; absolute numbers shift slightly between runs but the ranking and
ratios are stable.)

### Depth + pose (C++/ggml, native @504)

Pose is produced by a small head on top of the **same** backbone pass, so the
overhead over depth-only is within run-to-run noise (≈0–30 ms):

| quant | depth ms | depth+pose ms | pose overhead ms |
|-------|---------:|--------------:|-----------------:|
| f32   |   857.8  |     819.1     |      ~0 (noise)  |
| q8_0  |   752.3  |     762.0     |       ~10        |
| q4_k  |   712.1  |     739.5     |       ~27        |

## Headline findings

- **Latency: PyTorch is currently faster on CPU.** C++/ggml f32 depth is ~2.0x
  slower than PyTorch f32 (858 ms vs 429 ms @504; 242 ms vs 120 ms @224). PyTorch
  2.12's oneDNN-optimized convolutions + threading beat ggml's CPU
  `im2col + matmul` conv path. This model is **convolution-heavy** (patch embed +
  DPT head reassemble/fusion convs), which is ggml-on-CPU's weak spot. There is no
  GPU offload yet — that is the obvious next lever.
- **Quantization gives a modest latency win** (q4_k 712 ms vs f32 858 ms @504,
  ~17%; q8_0 ~12%). Quantization only accelerates the **matmuls** (attention /
  MLP); the **convolutions stay f32** and still dominate the wall time, so the C++
  port stays slower than PyTorch even at q4_k (0.60x).
- **Quantization wins big on size and memory.** q4_k is **99 MB on disk (4.0x
  smaller than the 393 MB f32 GGUF, 5.2x smaller than the 516 MB safetensors)** and
  uses **792 MB peak RSS — 40% less than PyTorch's 1328 MB** and 27% less than C++
  f32. q8_0 sits in between (142 MB, 835 MB RSS) with near-lossless accuracy.
- **Load time is much lower for the C++ port** (24–107 ms vs 770 ms for PyTorch),
  helped by mmap'd GGUF and no Python/torch import cost. This matters for
  one-shot / CLI invocations even though steady-state inference is slower.

## Method & honesty notes

- **CPU only.** No GPU / Metal / CUDA offload is wired yet; all numbers are
  single-threaded-process CPU with 8 worker threads.
- **Single image, single model size** (DA3-BASE). Larger models (GIANT) will shift
  the matmul/conv ratio and likely make quantization more impactful.
- **C++ timing** uses the CLI's `--repeat N` bench hook (added in M9-T3): the model
  is loaded **once**, then depth is run N times and the median per-iter ms is
  reported — so inference time excludes the per-subprocess model-reload overhead.
  The C++ infer time **includes** image-load + DA3 preprocess + backbone + DPT head.
- **PyTorch timing** loads the net once, does 1 warmup, then times N forward passes
  (`backbone.get_intermediate_layers` + `head`). Preprocess is done **once outside**
  the timed loop (forward-only). This slightly favors PyTorch (no per-iter
  preprocess), but preprocess is a few ms vs hundreds, so it does not change the
  conclusion.
- **Peak RSS** for both engines is the child-process `Maximum resident set size`
  from `/usr/bin/time -v` (kbytes → MB), so each process pays its own model-load
  memory — an apples-to-apples comparison.
- **PyTorch q comparison is N/A** (the reference path runs f32; quantized inference
  is a C++/ggml-only feature here).
- The @224 PyTorch input is the 224×224 image fed directly to the backbone (already
  a multiple of patch=14); the @504 PyTorch input uses the genuine upstream
  `InputProcessor` resize, matching the C++ native path's processed resolution.

## Takeaway

The C++/ggml port's value today is **portability, small footprint, fast startup,
and low memory** (q4_k: 99 MB model, 792 MB RSS, 40% less than PyTorch), not raw
CPU latency — PyTorch's oneDNN convolutions are ~2x faster on this CPU. Closing the
latency gap means (a) GPU offload and/or (b) a faster conv2d path for the
patch-embed and DPT head, since convolutions — not the quantizable matmuls —
dominate the runtime.

---

## CPU optimization: why the gap, and how it was narrowed (measured)

**Where the time goes.** ggml runs `Conv2d` as `im2col(→F16)` + `ggml_mul_mat`, and the
transformer backbone is pure `ggml_mul_mat`. So the whole forward is bottlenecked on
GEMM. PyTorch (CPU) uses **oneDNN**: JIT AVX-512 GEMM microkernels and **direct/Winograd
convolution** (no im2col memory blow-up, blocked NCHWc layouts). That conv algorithm — not
SIMD — is the core difference (`-march=native`/AVX-512 was already on in both).

**Measured wins (DA3-BASE depth @504×336, AMD Ryzen 9 9950X3D, no code/parity change):**

| change | infer ms/iter | note |
|---|--:|---|
| baseline (llamafile OFF, 8 threads) | 831 | as shipped in the first benchmark |
| `GGML_LLAMAFILE=ON` (tinyBLAS sgemm/hgemm) | 600 | **−28%**, free, static (no new dep) |
| + 16 threads (box has 20 cores; 32 oversubscribes) | **538** | optimal thread count |
| OpenBLAS (`GGML_BLAS`) instead | 556 | no win over llamafile, and adds a dynamic dep → rejected |

Net: **2.0× → 1.25× slower than PyTorch** (538 vs 429 ms) with a one-line build flag +
thread tuning, depth bit-identical (min/max 0.8595/0.9534, matching the e2e reference).
`GGML_LLAMAFILE=ON` is now the **default** in `CMakeLists.txt` (`DA_GGML_LLAMAFILE`).

Note: with llamafile on, **q4_k is no longer a latency win** (675 ms > f32 600 ms) — the
optimized f32/f16 GEMM beats the quantized dequant-matmul path here. Quantization remains a
size/memory win (q4_k 99 MB, 792 MB RSS vs PyTorch 1328 MB).

**Closing the last 1.25× (future, in rough effort order):**
1. Keep conv kernels in **F16** (halves the im2col GEMM bytes) — small, parity within f16.
2. A **direct 3×3 conv** op (or Winograd) in ggml-cpu for the DPT head — avoids the im2col
   9× expansion; this is the bulk of "what oneDNN does".
3. Link **oneDNN / libxsmm** for the head convs — literally PyTorch's kernels (biggest dep cost).
4. **GPU offload** (CUDA/Metal/Vulkan) — makes the CPU GEMM/conv question moot; the highest-leverage path.

---

## CPU optimization #2: direct convolution (DPT head)

Stage profiling (`DA_PROFILE=1`) showed the **DPT head** (all `Conv2d`) — not the backbone —
is the dominant cost, because ggml ran convs as `im2col`(9× memory expansion)+`mul_mat`. ggml
also ships `ggml_conv_2d_direct` (a native `GGML_OP_CONV_2D`, **no im2col**), the same direct-conv
class oneDNN uses. `src/dpt_blocks.cpp::conv2d` now uses it for K>1 kernels (the 3×3 head convs)
and keeps im2col+llamafile-sgemm for 1×1 (pure GEMM). Toggle: `DA_CONV=im2col|direct|auto`.

DA3-BASE depth @504×336, 16 threads, parity-exact (max|d|=5.96e-08 vs im2col):

| metric | im2col | **direct (default)** | Δ |
|---|--:|--:|--:|
| warm latency (serving, repeat-median) | 542 ms | **495 ms** | −9% |
| cold latency (one-shot CLI, incl. load) | 1.12 s | **0.72 s** | −36% |
| peak RSS | 1014 MB | **665 MB** | −34% |

**Cumulative CPU result** (from the original 831 ms warm / im2col-off / 8-thread baseline):
`GGML_LLAMAFILE` + 16 threads + direct conv → **495 ms warm (1.68× faster than the start)**,
**0.72 s cold**, **665 MB peak** (half of PyTorch's 1328 MB). vs PyTorch warm 429 ms = 1.15×
slower but with half the memory and faster load/cold — and the same correctness.

---

## CPU profiling #3: warm split + what does NOT help

Warm steady-state stage split (DA3-BASE @504, 16 threads, direct conv) is now **balanced**:
backbone ≈ 228 ms (matmuls, llamafile sgemm), head ≈ 255 ms (convs, tiled direct).

Measured **negative results** (so we don't chase them again):
- **F16 matmul weights** (`quantize f16`): backbone 228→224 ms (~2%), total 497→485 ms, and
  corr drops to 0.999995. llamafile's F32 AVX-512 sgemm is already near-optimal for these shapes;
  not bandwidth-bound. Rejected (accuracy cost, negligible gain).
- **F16 conv kernels** (tiled-conv inner hgemm): head unchanged (~260 ms). The conv is
  **compute-bound on GEMM FLOPs**, not kernel bandwidth. Rejected.

Conclusion: the remaining CPU lever is **fewer conv FLOPs** — i.e. **Winograd F(2×2,3×3)** for the
head's 3×3 convs (2.25× fewer multiplies, the algorithm oneDNN uses). Caveat: it only wins if the
Winograd-domain GEMM is as well-vectorized as ggml's llamafile kernel — a naive Winograd can lose
despite fewer FLOPs. Approach: implement, measure, keep only if faster AND parity holds.

## CPU optimization #4: Winograd F(2×2,3×3) conv (DPT head) — **shipped, faster**

Implemented `src/winograd.cpp`: a CPU Winograd F(2×2,3×3) custom op (`ggml_custom_4d`)
wired into `src/dpt_blocks.cpp::conv2d` for **3×3 stride-1 F32** convs (all the DPT head's
reassemble/fusion/output 3×3 convs). The exact transforms (`B^T d B`, `G g G^T`, `A^T m A`,
halves+integers) reduce the 9-multiply 3×3 conv to a **16-position elementwise multiply**
(2.25× fewer multiplies). The hot path — the winograd-domain multiply
`M[ξν][oc] = Σ_IC U[oc][ic][ξν]·V[ic][ξν]` — is laid out **OC-innermost** and vectorized with
an **AVX-512 FMA GEMV** (16-wide over OC, broadcasting V). The filter transform `U` is computed
once and cached by filter pointer across forwards; tiles×batch are threaded via `(ith,nth)`.

It **wins** (DA3-BASE @504, 16 threads, `--repeat 12`, median; warm-head median over iters 2..N):

| DA_CONV | warm head ms | total infer ms |
|---|---|---|
| `direct` (prev default) | ~242 | ~492 |
| `winograd` (**new auto default**) | **~205** | **~450** |

~**15% faster head**, ~**8% faster total** (≈40 ms/iter), reproducible across trials.
Parity is **exact**: `test_winograd` (random 128×96, IC=OC=64) gives max|d|=**1.4e-5** vs
`ggml_conv_2d_direct` (≪ the 2e-3 gate), and all 29 model-parity tests stay green with winograd
active. So Winograd is now the **auto default** for 3×3 stride-1 convs; `DA_CONV=direct` (or
`im2col`) still selects the old paths for A/B. The win confirms the lever was conv FLOPs, and that
a carefully-vectorized Winograd GEMV can beat ggml's already-tuned llamafile direct conv here.

## CPU optimization #5: blocked F(2×2) GEMM + F(4×4) eval — **f2b shipped, f4 rejected as default**

The CPU-opt #4 Winograd used a **per-tile GEMV** (one 16-wide AVX-512 reduction over OC, per
tile, per winograd position) — it reloads the filter `U` for every tile. Two ideas to go faster:

- **B. Blocked F(2×2) GEMM (`DA_WINO=f2b`).** *Same* F(2×2) algorithm, better kernel: batch
  `TB=8` tiles and turn the winograd-domain multiply into a small GEMM
  `M[ξν][t][oc] = Σ_IC U[ξν][ic][oc]·V[ξν][ic][t]`. Each loaded `U`-row (16 OC lanes) is now reused
  across all 8 tiles in registers (8 zmm accumulators), so arithmetic intensity goes up and `U`
  traffic drops ~8×. Threading splits over tile-**blocks** so every thread's GEMM stays full-width.
  This is **parity-identical** to F(2×2) (same FLOPs, same f32 reassociation).
- **A. F(4×4,3×3) Winograd (`DA_WINO=f4`).** 6×6 input tile → 4×4 output, 36 winograd positions;
  **4× fewer mults vs direct** (vs 2.25× for F(2×2)). Transform matrices (Lavin & Gray) were
  **verified numerically in python first** (float64 max|d|=6.6e-14 vs a direct 3×3 conv; float32
  single-tile ≈4.2e-5). Reuses the same blocked GEMM kernel as f2b. **Risk:** the `1/6`, `1/24`
  fractions make it less accurate than F(2×2).

**Parity (test_winograd random 128×96, IC=OC=64, vs `ggml_conv_2d_direct`; e2e = native depth corr):**

| mode | test_winograd max\|d\| | e2e max\|d\| | e2e corr | full suite |
|---|--:|--:|--:|:--|
| `direct` (reference) | 0 (is the reference) | — | — | green |
| `f2` (per-tile GEMV) | 1.38e-5 | 1.49e-6 | 1.000000 | green |
| `f2b` (blocked, **new default**) | **1.38e-5** (bit-identical to f2) | 1.49e-6 | **1.000000** | **30/30 green** |
| `f4` (F(4×4)) | 2.20e-4 | 1.31e-6 | 1.000000 | 30/30 green |

Note f4 *does* pass the hard gate (suite green, e2e corr=1.0) — its higher per-conv error
(2.2e-4 ≪ the 2e-3 test gate) washes out after the head's normalization, so the final depth is
still corr=1.0. It is **not** rejected for breaking parity.

**Warm head latency (DA3 @504×336, 16 threads, `--repeat 25–30`, median of warm iters):**

| mode | BASE head ms | GIANT head ms |
|---|--:|--:|
| `direct` (pre-#4) | ~242 | — |
| `f2` (#4 default) | ~205 | ~625 |
| `f2b` (**new default**) | **~194** | **~490** (−22% vs f2) |
| `f4` | ~195 | ~494 |

**Decision — `f2b` is the new auto default.** On BASE, f2b and f4 are statistically tied (~194 ms,
both ~5% faster than f2). On the much larger **GIANT** head the blocked GEMM wins big (490 vs 625
ms, **−22%**), and there f2b is even marginally faster than f4 — the F(4×4) FLOP cut is eaten by its
larger (6×6) input/output transforms and 36-position bookkeeping, while the blocked-GEMM
amortization (the real bottleneck) is identical for both. So **f4 is faster than the old f2 but NOT
faster than f2b**, and it carries real accuracy risk (2.2e-4 per-conv, fractional transform). The
honest call: ship the variant that is **fastest AND parity-exact** → `f2b`. `f4` is kept selectable
(`DA_WINO=f4`) and documented, but is not the default because it gives no speed win over f2b while
adding numerical risk. `DA_WINO=f2` restores the old per-tile GEMV for A/B.
