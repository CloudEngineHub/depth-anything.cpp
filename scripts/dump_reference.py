#!/usr/bin/env python3
"""Dump gold per-component reference tensors for DA3-BASE backbone (N=1).

Produces dumps/reference.gguf (flattened f32 tensors, row-major) and
dumps/manifest.json (pre-flatten shapes + tolerances). These are the GOLD
reference the C++ backbone parity gates (Tasks 12 and 15) assert against.
"""
import os, json, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import load_model, fixed_input, FIX_H, FIX_W, PATCH

OUT = "dumps/reference.gguf"
MANIFEST = "dumps/manifest.json"
OUT_LAYERS = [5, 7, 9, 11]


def main():
    os.makedirs("dumps", exist_ok=True)
    _, net = load_model()
    x, raw = fixed_input()
    bb = net.backbone.pretrained
    cap = {}

    # Capture the tokens right after cls-prepend + interpolated-pos-embed add,
    # before block 0, by wrapping prepare_tokens_with_masks.
    orig = bb.prepare_tokens_with_masks

    def wrapped(xx, *a, **k):
        out = orig(xx, *a, **k)
        cap["pos_embed_added"] = out.detach().clone()
        return out

    bb.prepare_tokens_with_masks = wrapped
    try:
        with torch.no_grad():
            outs, _aux = bb.get_intermediate_layers(
                x, n=OUT_LAYERS, export_feat_layers=[],
                ref_view_strategy="saddle_balanced")
    finally:
        bb.prepare_tokens_with_masks = orig

    feats = [o[0] for o in outs]
    cams = [o[1] for o in outs]
    for L, f, c in zip(OUT_LAYERS, feats, cams):
        cap[f"feat_{L}"] = f.detach().contiguous().float()
        cap[f"cam_token_{L}"] = c.detach().contiguous().float()
    cap["input_image"] = x.detach().contiguous().float()
    cap["pos_embed_added"] = cap["pos_embed_added"].detach().contiguous().float()

    w = gguf.GGUFWriter(OUT, "reference")
    for k, v in cap.items():
        arr = np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32))
        w.add_tensor(k, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shapes = {k: list(v.shape) for k, v in cap.items()}
    with open(MANIFEST, "w") as f:
        json.dump({"H": FIX_H, "W": FIX_W, "patch": PATCH, "out_layers": OUT_LAYERS,
                   "shapes": shapes, "atol": 2e-3, "rtol": 2e-3}, f, indent=2)

    print("wrote", OUT)
    for k, v in cap.items():
        print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
