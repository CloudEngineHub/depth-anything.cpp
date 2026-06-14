#!/usr/bin/env python3
"""Convert DA3-BASE to a single self-contained GGUF: config as KV, backbone weights as f32."""
import argparse, sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K
from scripts.da3_reference import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DA3-BASE")
    ap.add_argument("--output", default="models/depth-anything-base-f32.gguf")
    a = ap.parse_args()

    _, net = load_model(a.model)
    bb = net.backbone.pretrained if hasattr(net.backbone, "pretrained") else net.backbone
    embed_dim = bb.embed_dim
    depth = bb.n_blocks
    num_heads = bb.num_heads
    head_dim = embed_dim // num_heads
    mlp_hidden = bb.blocks[0].mlp.fc1.out_features
    pos_rows = bb.pos_embed.shape[1] - 1
    M = int(round(pos_rows ** 0.5))
    assert M * M == pos_rows, f"pos_embed rows {pos_rows} not a perfect square"
    # LayerScale gamma is learned per-channel and exported faithfully as the
    # ls1/ls2 tensors; this scalar KV is informational only and not used at
    # inference. Do NOT derive it from gamma.mean() — that averages a learned
    # per-channel vector and bears no relation to the original init constant.
    init_values = 0.0
    qkv_bias = bb.blocks[0].attn.qkv.bias is not None

    w = gguf.GGUFWriter(a.output, K.ARCH)
    w.add_string(K.KV["arch"], K.ARCH)
    w.add_string(K.KV["checkpoint_name"], "DA3-BASE")
    w.add_uint32(K.KV["patch_size"], 14)
    w.add_uint32(K.KV["vit.embed_dim"], int(embed_dim))
    w.add_uint32(K.KV["vit.depth"], int(depth))
    w.add_uint32(K.KV["vit.num_heads"], int(num_heads))
    w.add_uint32(K.KV["vit.head_dim"], int(head_dim))
    w.add_uint32(K.KV["vit.mlp_hidden"], int(mlp_hidden))
    w.add_uint32(K.KV["vit.num_register"], int(bb.num_register_tokens))
    w.add_float32(K.KV["vit.init_values"], init_values)
    w.add_int32(K.KV["vit.alt_start"], int(bb.alt_start))
    w.add_int32(K.KV["vit.rope_start"], int(bb.rope_start))
    w.add_int32(K.KV["vit.qknorm_start"], int(bb.qknorm_start))
    w.add_float32(K.KV["vit.rope_freq"], 100.0)
    w.add_bool(K.KV["vit.cat_token"], bool(bb.cat_token))
    w.add_bool(K.KV["vit.qkv_bias"], bool(qkv_bias))
    w.add_float32(K.KV["vit.ln_eps"], 1e-6)
    w.add_float32(K.KV["vit.interp_offset"], float(bb.interpolate_offset))
    w.add_bool(K.KV["vit.interp_antialias"], bool(bb.interpolate_antialias))
    w.add_uint32(K.KV["vit.pos_embed_grid"], int(M))
    w.add_array(K.KV["vit.out_layers"], [5, 7, 9, 11])
    w.add_array(K.KV["img.mean"], [0.485, 0.456, 0.406])
    w.add_array(K.KV["img.std"], [0.229, 0.224, 0.225])
    w.add_string(K.KV["img.resize_mode"], "upper_bound")
    w.add_uint32(K.KV["img.resize_target"], 504)

    written, skipped = 0, []
    for name, t in net.backbone.named_parameters():
        canon = name.split("pretrained.")[-1] if "pretrained." in name else name
        g = K.rename_backbone(canon)
        if g is None:
            skipped.append(name)
            continue
        arr = np.ascontiguousarray(
            t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
        )
        w.add_tensor(g, arr)
        written += 1
    if written == 0:
        raise SystemExit("error: no backbone tensors mapped; check rename_backbone prefix")
    # The backbone loop iterates only backbone params, so every one should map.
    # A nonzero skip means rename_backbone is missing a rule (e.g. a new variant's
    # mask_token / register_tokens) and would silently drop weights — fail loudly.
    if skipped:
        raise SystemExit(
            f"error: {len(skipped)} backbone param(s) unmapped (rename_backbone gap): {skipped[:5]}"
        )

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.output}: backbone_tensors={written} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
