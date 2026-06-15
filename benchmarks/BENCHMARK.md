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
