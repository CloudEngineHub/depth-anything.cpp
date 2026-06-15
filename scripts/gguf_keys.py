"""Single source of truth for GGUF arch, KV keys, and tensor renames.
Both the C++ loader (via generated include/da_gguf_keys.h) and the Python
converter import from here, so they cannot drift."""
import re

ARCH = "depthanything3"

# short-key -> full GGUF KV string
KV = {
    "arch":                 f"{ARCH}.arch",
    "checkpoint_name":      f"{ARCH}.checkpoint_name",
    "patch_size":           f"{ARCH}.patch_size",
    "image_size":           f"{ARCH}.image_size",
    "task_caps":            f"{ARCH}.task_caps",          # bitmask of heads present
    # backbone (DINOv2)
    "vit.embed_dim":        f"{ARCH}.vit.embed_dim",
    "vit.depth":            f"{ARCH}.vit.depth",
    "vit.num_heads":        f"{ARCH}.vit.num_heads",
    "vit.head_dim":         f"{ARCH}.vit.head_dim",
    "vit.mlp_hidden":       f"{ARCH}.vit.mlp_hidden",
    "vit.num_register":     f"{ARCH}.vit.num_register_tokens",
    "vit.init_values":      f"{ARCH}.vit.init_values",
    "vit.alt_start":        f"{ARCH}.vit.alt_start",
    "vit.rope_start":       f"{ARCH}.vit.rope_start",
    "vit.qknorm_start":     f"{ARCH}.vit.qknorm_start",
    "vit.rope_freq":        f"{ARCH}.vit.rope_freq",
    "vit.cat_token":        f"{ARCH}.vit.cat_token",
    "vit.qkv_bias":         f"{ARCH}.vit.qkv_bias",
    "vit.ln_eps":           f"{ARCH}.vit.ln_eps",
    "vit.interp_offset":    f"{ARCH}.vit.interpolate_offset",
    "vit.interp_antialias": f"{ARCH}.vit.interpolate_antialias",
    "vit.pos_embed_grid":   f"{ARCH}.vit.pos_embed_grid",  # M where pos_embed has M*M+1 rows
    "vit.out_layers":       f"{ARCH}.vit.out_layers",
    # preprocessing
    "img.mean":             f"{ARCH}.img.mean",
    "img.std":              f"{ARCH}.img.std",
    "img.resize_mode":      f"{ARCH}.img.resize_mode",
    "img.resize_target":    f"{ARCH}.img.resize_target",   # target long/short side, multiple of patch
}

def rename_backbone(name: str):
    """HF backbone param name (with 'pretrained.' prefix already stripped) ->
    GGUF tensor name, or None if not a backbone tensor."""
    n = name
    if n == "patch_embed.proj.weight": return "vit.patch_embed.weight"
    if n == "patch_embed.proj.bias":   return "vit.patch_embed.bias"
    if n == "cls_token":               return "vit.cls_token"
    if n == "camera_token":            return "vit.camera_token"
    if n == "pos_embed":               return "vit.pos_embed"
    if n == "norm.weight":             return "vit.norm.weight"
    if n == "norm.bias":               return "vit.norm.bias"
    m = re.match(r"^blocks\.(\d+)\.(.+)$", n)
    if m:
        i, rest = m.group(1), m.group(2)
        table = {
            "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
            "attn.qkv.weight": "attn_qkv.weight", "attn.qkv.bias": "attn_qkv.bias",
            "attn.proj.weight": "attn_proj.weight", "attn.proj.bias": "attn_proj.bias",
            "attn.q_norm.weight": "attn_qnorm.weight", "attn.q_norm.bias": "attn_qnorm.bias",
            "attn.k_norm.weight": "attn_knorm.weight", "attn.k_norm.bias": "attn_knorm.bias",
            "ls1.gamma": "ls1", "ls2.gamma": "ls2",
            "mlp.fc1.weight": "mlp_fc1.weight", "mlp.fc1.bias": "mlp_fc1.bias",
            "mlp.fc2.weight": "mlp_fc2.weight", "mlp.fc2.bias": "mlp_fc2.bias",
        }
        if rest in table:
            return f"vit.blk.{i}.{table[rest]}"
    return None
