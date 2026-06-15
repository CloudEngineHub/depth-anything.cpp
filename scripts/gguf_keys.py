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
    # DualDPT depth head (main path)
    "head.features":        f"{ARCH}.head.features",
    "head.out_channels":    f"{ARCH}.head.out_channels",
    "head.output_dim":      f"{ARCH}.head.output_dim",
    "head.pos_embed":       f"{ARCH}.head.pos_embed",
    "head.down_ratio":      f"{ARCH}.head.down_ratio",
    "head.activation":      f"{ARCH}.head.activation",
    "head.conf_activation": f"{ARCH}.head.conf_activation",
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


# Aux (ray/sky) head tensors live in the same module as the main depth path but
# belong to M3. M2 intentionally skips them; this matches the prefixes for
# scratch.refinenet{i}_aux.*, scratch.output_conv1_aux.*, scratch.output_conv2_aux.*
_HEAD_AUX_RE = re.compile(r"(refinenet\d+_aux|output_conv1_aux|output_conv2_aux)")


def is_head_aux(name: str) -> bool:
    """True if `name` is a DualDPT auxiliary-head param (intentionally skipped in M2)."""
    return _HEAD_AUX_RE.search(name) is not None


def rename_head(name: str):
    """HF DualDPT head param name (already without 'head.' prefix, e.g. 'norm.weight',
    'projects.0.weight', 'scratch.refinenet1.resConfUnit1.conv1.weight') ->
    GGUF tensor name, or None if it is an aux/unknown tensor (caller decides whether
    a None is an intentional aux skip via is_head_aux, or a hard error)."""
    n = name
    if n in ("norm.weight", "norm.bias"):
        return f"head.{n}"
    m = re.match(r"^projects\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"head.proj.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^resize_layers\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"head.resize.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^scratch\.layer(\d+)_rn\.(weight|bias)$", n)
    if m:
        return f"head.scratch.layer{m.group(1)}_rn.{m.group(2)}"
    # refinenet{i} (main only; aux is handled by is_head_aux and returns None here)
    m = re.match(
        r"^scratch\.refinenet(\d+)\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)$", n)
    if m and "_aux" not in n:
        i, unit, conv, wb = m.groups()
        return f"head.scratch.rn{i}.rc{unit}.c{conv}.{wb}"
    m = re.match(r"^scratch\.refinenet(\d+)\.out_conv\.(weight|bias)$", n)
    if m and "_aux" not in n:
        return f"head.scratch.rn{m.group(1)}.out.{m.group(2)}"
    if re.match(r"^scratch\.output_conv1\.(weight|bias)$", n):
        return "head.scratch.out1." + n.rsplit(".", 1)[-1]
    m = re.match(r"^scratch\.output_conv2\.(\d+)\.(weight|bias)$", n)
    if m:
        sub = {"0": "out2a", "2": "out2b"}.get(m.group(1))
        if sub is not None:
            return f"head.scratch.{sub}.{m.group(2)}"
    return None
