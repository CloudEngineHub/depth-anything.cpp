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
    cap["raw_image"] = torch.from_numpy(raw.astype(np.float32))  # (224,224,3) HWC, values 0..255
    cap["pos_embed_added"] = cap["pos_embed_added"].detach().contiguous().float()

    # --- 2D RoPE isolated parity fixture (Task 11) -----------------------------
    # Uses the REAL reference module so the C++ rope is gated against ground truth.
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D
    rope = RotaryPositionEmbedding2D(frequency=100.0)
    hd, T = 64, 4
    g = torch.Generator().manual_seed(1)
    rin = torch.randn(1, 1, T, hd, generator=g)                          # (B,heads,N,hd)
    rpos = torch.tensor([[[1, 1], [1, 2], [2, 1], [2, 2]]], dtype=torch.long)  # (1,N,2) y,x
    with torch.no_grad():
        rout = rope(rin, rpos)
    cap["rope_in"] = rin.detach().contiguous().float()
    cap["rope_out"] = rout.detach().contiguous().float()
    cap["rope_pos"] = rpos.detach().contiguous().float()

    # --- DualDPT depth head reference (Task M2) --------------------------------
    # The head consumes the raw `outs` structure (a list of (feature, cam) tuples,
    # exactly what get_intermediate_layers returns). Hook the post-resize stages
    # and the post-output_conv1 fused tensor for layer-isolation debugging.
    head = net.head
    handles = []

    def _mk_hook(key):
        def _h(_m, _inp, out):
            cap[key] = out.detach().contiguous().float()
        return _h

    for s in range(4):
        handles.append(head.resize_layers[s].register_forward_hook(_mk_hook(f"head_stage{s}")))
    handles.append(head.scratch.output_conv1.register_forward_hook(_mk_hook("head_fused")))
    try:
        with torch.no_grad():
            head_out = net.head(list(outs), FIX_H, FIX_W, patch_start_idx=0)
    finally:
        for hd in handles:
            hd.remove()

    head_depth = head_out["depth"].squeeze()        # (224,224)
    head_depth_conf = head_out["depth_conf"].squeeze()
    assert tuple(head_depth.shape) == (FIX_H, FIX_W), head_depth.shape
    assert torch.isfinite(head_depth).all() and bool((head_depth > 0).all()), "depth must be positive/finite (exp)"
    assert bool((head_depth_conf >= 1.0).all()), "depth_conf must be >= 1 (expp1)"
    cap["head_depth"] = head_depth.detach().contiguous().float()
    cap["head_depth_conf"] = head_depth_conf.detach().contiguous().float()

    # UV positional embedding (224x224x64), BEFORE the *0.1 ratio scaling.
    from depth_anything_3.model.utils.head_utils import create_uv_grid, position_grid_to_embed
    uv = create_uv_grid(FIX_W, FIX_H, aspect_ratio=1.0)
    uv_emb = position_grid_to_embed(uv, 64)   # (224,224,64)
    cap["uv_embed_64"] = uv_emb.detach().float().contiguous()

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
