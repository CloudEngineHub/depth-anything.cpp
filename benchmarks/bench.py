#!/usr/bin/env python3
"""Performance benchmark suite for the Depth Anything 3 C++/ggml port (M9-T3).

Measures latency (median + p90 ms/iter) and peak RSS for:
  - C++/ggml `da3-cli depth`, across {f32, q8_0, q4_k} quantizations,
    at {224x224 (--legacy-resize), native ~504 (640x427 photo)} resolutions.
  - The original PyTorch DA3-BASE reference forward (f32 only), same images.

Timing method
-------------
C++  : the CLI has a `--repeat N` bench hook (added in M9-T3). It loads the model
       ONCE, then runs depth N times and prints
         `bench: ... load=Lms infer=Xms/iter (median over N, min=.. max=.. p90=..)`
       so inference is measured WITHOUT per-subprocess model-reload overhead.
       The C++ infer time includes image-load + DA3 preprocess + backbone + DPT head.
PyTorch: this script's `--torch-worker` mode loads the net once, does 1 warmup,
       then times N forward passes (backbone get_intermediate_layers + head).
       PyTorch preprocess is done once outside the timed loop (forward-only).

Peak RSS for BOTH is the child-process "Maximum resident set size" reported by
`/usr/bin/time -v`, so the comparison is apples-to-apples (each child loads its
own model). RSS is reported in MB (kbytes/1024).

Usage:
  python benchmarks/bench.py                 # run full suite, write results.json + print table
  python benchmarks/bench.py --torch-worker --image <png> --res {224,504} --threads 8 --repeat 6
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")
IMG_224 = os.path.join(ROOT, "dumps", "bench_input_224.png")
IMG_504 = os.path.join(ROOT, "dumps", "native_input.png")  # 640x427 -> 504x336
MODELS = {
    "f32":  os.path.join(ROOT, "models", "depth-anything-base-f32.gguf"),
    "q8_0": os.path.join(ROOT, "models", "depth-anything-base-q8_0.gguf"),
    "q4_k": os.path.join(ROOT, "models", "depth-anything-base-q4_k.gguf"),
}
THREADS = 8
REPEAT = 6
DA3_SRC = "/tmp/da3-src/src"

TIME_BIN = "/usr/bin/time"
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
CPP_BENCH_RE = re.compile(r"load=([\d.]+)ms\s+infer=([\d.]+)ms/iter.*?p90=([\d.]+)")


def mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def run_under_time(cmd):
    """Run cmd under /usr/bin/time -v; return (stdout, peak_rss_mb)."""
    full = [TIME_BIN, "-v"] + cmd
    p = subprocess.run(full, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + "\n" + p.stderr + "\n")
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    m = RSS_RE.search(p.stderr)
    rss_mb = int(m.group(1)) / 1024.0 if m else float("nan")
    return p.stdout, rss_mb


def bench_cpp(quant, image, legacy, pose=False):
    cmd = [CLI, "depth", "--model", MODELS[quant], "--input", image,
           "--threads", str(THREADS), "--repeat", str(REPEAT)]
    if legacy:
        cmd.append("--legacy-resize")
    if pose:
        cmd += ["--pose", "/tmp/bench_pose.json"]
    out, rss = run_under_time(cmd)
    m = CPP_BENCH_RE.search(out)
    if not m:
        raise RuntimeError(f"could not parse cpp bench line:\n{out}")
    load_ms, infer_ms, p90 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"load_ms": load_ms, "infer_ms": infer_ms, "p90_ms": p90, "rss_mb": rss}


def bench_torch(image, res):
    cmd = [sys.executable, os.path.abspath(__file__), "--torch-worker",
           "--image", image, "--res", str(res),
           "--threads", str(THREADS), "--repeat", str(REPEAT)]
    out, rss = run_under_time(cmd)
    m = CPP_BENCH_RE.search(out)
    if not m:
        raise RuntimeError(f"could not parse torch bench line:\n{out}")
    load_ms, infer_ms, p90 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"load_ms": load_ms, "infer_ms": infer_ms, "p90_ms": p90, "rss_mb": rss}


# --------------------------------------------------------------------------- #
# PyTorch worker (separate process so its peak RSS is measured cleanly).
# --------------------------------------------------------------------------- #
def torch_worker(image, res, threads, repeat):
    import time as _time
    import types
    import numpy as np
    import torch
    torch.set_num_threads(threads)

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

    # Install the torchvision stub BEFORE load_model (which transitively imports
    # the InputProcessor): the only installable torchvision wheel here is ABI-broken
    # vs torch 2.12, so ToTensor/Normalize are stubbed with their exact math.
    tv = types.ModuleType("torchvision")
    tvt = types.ModuleType("torchvision.transforms")

    class ToTensor:
        def __call__(self, im):
            a = np.asarray(im)
            if a.ndim == 2:
                a = a[:, :, None]
            tt = torch.from_numpy(np.ascontiguousarray(a)).float() / 255.0
            return tt.permute(2, 0, 1).contiguous()

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        def __call__(self, tt):
            return (tt - self.mean) / self.std

    class CenterCrop:
        def __init__(self, size): self.size = size
        def __call__(self, tt):
            H, W = tt.shape[-2:]; th, tw = self.size
            top = max(0, (H - th) // 2); left = max(0, (W - tw) // 2)
            return tt[..., top:top + th, left:left + tw]

    tvt.ToTensor = ToTensor; tvt.Normalize = Normalize; tvt.CenterCrop = CenterCrop
    tv.transforms = tvt
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tvt

    t_load0 = _time.perf_counter()
    from da3_reference import load_model, NORM_MEAN, NORM_STD
    _, net = load_model(os.path.join(ROOT, "models", "DA3-BASE"))
    bb = net.backbone.pretrained
    t_load1 = _time.perf_counter()
    load_ms = (t_load1 - t_load0) * 1000.0

    # Build the input tensor (1,1,3,H,W).
    if res == 224:
        # Match C++ --legacy-resize @224: feed the 224x224 image directly
        # (already a multiple of patch=14), ToTensor + ImageNet normalize.
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(image).convert("RGB"), dtype=np.uint8)
        img = arr.astype(np.float32) / 255.0
        img = (img - NORM_MEAN) / NORM_STD
        t = torch.from_numpy(img).permute(2, 0, 1)[None, None].contiguous()
    else:
        # Native ~504: genuine InputProcessor upper_bound_resize (process_res=504).
        sys.path.insert(0, DA3_SRC)
        from depth_anything_3.utils.io.input_processor import InputProcessor
        proc = InputProcessor()
        tensor, _, _ = proc([image], process_res=504, process_res_method="upper_bound_resize")
        t = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1])[0][None, None].contiguous()

    _, _, _, H, W = t.shape

    def forward():
        with torch.no_grad():
            outs, _ = bb.get_intermediate_layers(
                t, n=[5, 7, 9, 11], export_feat_layers=[], ref_view_strategy="saddle_balanced")
            ho = net.head(list(outs), H, W, patch_start_idx=0)
            return ho["depth"]

    forward()  # warmup
    ms = []
    for _ in range(repeat):
        a = _time.perf_counter()
        forward()
        ms.append((_time.perf_counter() - a) * 1000.0)
    ms.sort()
    median = ms[len(ms) // 2]
    p90 = ms[min(int(len(ms) * 0.9), len(ms) - 1)]
    print(f"bench: out={W}x{H} threads={threads} load={load_ms:.1f}ms "
          f"infer={median:.1f}ms/iter (median over {repeat}, min={ms[0]:.1f} "
          f"max={ms[-1]:.1f} p90={p90:.1f})")


# --------------------------------------------------------------------------- #
def machine_info():
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        info["logical_cpus"] = sum(1 for l in open("/proc/cpuinfo") if l.startswith("processor"))
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / (1024 * 1024), 1)
                break
    except Exception as e:
        info["error"] = str(e)
    info["threads_used"] = THREADS
    info["repeat"] = REPEAT
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-worker", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    args = ap.parse_args()

    if args.torch_worker:
        torch_worker(args.image, args.res, args.threads, args.repeat)
        return

    for p in (CLI, IMG_224, IMG_504, *MODELS.values()):
        if not os.path.exists(p):
            sys.exit(f"missing required file: {p}")

    results = {"machine": machine_info(), "configs": []}

    print(f"Running benchmark suite (threads={THREADS}, repeat={REPEAT})...\n")

    # PyTorch reference (f32 only).
    print("PyTorch f32 @224 ...", flush=True)
    pt224 = bench_torch(IMG_224, 224)
    print("PyTorch f32 @504 ...", flush=True)
    pt504 = bench_torch(IMG_504, 504)
    results["configs"].append({
        "engine": "PyTorch", "quant": "f32",
        "model_size_mb": mb(os.path.join(ROOT, "models", "DA3-BASE", "model.safetensors")),
        "load_ms": round(pt504["load_ms"], 1),
        "infer_ms_224": round(pt224["infer_ms"], 1), "p90_ms_224": round(pt224["p90_ms"], 1),
        "infer_ms_504": round(pt504["infer_ms"], 1), "p90_ms_504": round(pt504["p90_ms"], 1),
        "rss_mb": round(max(pt224["rss_mb"], pt504["rss_mb"]), 1),
    })

    # C++/ggml across quant levels.
    for quant in ("f32", "q8_0", "q4_k"):
        print(f"C++ {quant} @224 ...", flush=True)
        c224 = bench_cpp(quant, IMG_224, legacy=True)
        print(f"C++ {quant} @504 ...", flush=True)
        c504 = bench_cpp(quant, IMG_504, legacy=False)
        results["configs"].append({
            "engine": "C++/ggml", "quant": quant,
            "model_size_mb": round(mb(MODELS[quant]), 1),
            "load_ms": round(c504["load_ms"], 1),
            "infer_ms_224": round(c224["infer_ms"], 1), "p90_ms_224": round(c224["p90_ms"], 1),
            "infer_ms_504": round(c504["infer_ms"], 1), "p90_ms_504": round(c504["p90_ms"], 1),
            "rss_mb": round(max(c224["rss_mb"], c504["rss_mb"]), 1),
        })

    # Depth + pose (C++/ggml only, native @504) — shows the pose-head overhead
    # on top of the shared backbone pass.
    results["pose_configs"] = []
    for quant in ("f32", "q8_0", "q4_k"):
        print(f"C++ {quant} depth+pose @504 ...", flush=True)
        cp = bench_cpp(quant, IMG_504, legacy=False, pose=True)
        depth_only = next(c for c in results["configs"]
                          if c["engine"] == "C++/ggml" and c["quant"] == quant)["infer_ms_504"]
        results["pose_configs"].append({
            "engine": "C++/ggml", "quant": quant,
            "depth_pose_ms_504": round(cp["infer_ms"], 1),
            "depth_only_ms_504": depth_only,
            "pose_overhead_ms": round(cp["infer_ms"] - depth_only, 1),
        })

    # speedup vs PyTorch f32 (using @504 infer).
    base504 = pt504["infer_ms"]
    base224 = pt224["infer_ms"]
    for c in results["configs"]:
        c["speedup_504_vs_torch"] = round(base504 / c["infer_ms_504"], 2)
        c["speedup_224_vs_torch"] = round(base224 / c["infer_ms_224"], 2)

    out_json = os.path.join(ROOT, "benchmarks", "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}\n")

    print_table(results)


def print_table(results):
    print("| engine | quant | size MB | load ms | infer @224 | infer @504 | peak RSS MB | speedup@504 |")
    print("|--------|-------|--------:|--------:|-----------:|-----------:|------------:|------------:|")
    for c in results["configs"]:
        print(f"| {c['engine']} | {c['quant']} | {c['model_size_mb']:.0f} | "
              f"{c['load_ms']:.0f} | {c['infer_ms_224']:.1f} | {c['infer_ms_504']:.1f} | "
              f"{c['rss_mb']:.0f} | {c['speedup_504_vs_torch']:.2f}x |")
    if results.get("pose_configs"):
        print("\nDepth+pose (C++/ggml, native @504):")
        print("| quant | depth ms | depth+pose ms | pose overhead ms |")
        print("|-------|---------:|--------------:|-----------------:|")
        for c in results["pose_configs"]:
            print(f"| {c['quant']} | {c['depth_only_ms_504']:.1f} | "
                  f"{c['depth_pose_ms_504']:.1f} | {c['pose_overhead_ms']:.1f} |")


if __name__ == "__main__":
    main()
