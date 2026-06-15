import os, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/depth-anything-base-f32.gguf"


@pytest.mark.skipif(not (ROOT / "models/DA3-BASE").exists(), reason="weights not downloaded")
def test_convert_and_read_back():
    subprocess.check_call([sys.executable, str(ROOT / "scripts/convert_da3_to_gguf.py")])
    import gguf
    r = gguf.GGUFReader(str(GGUF))
    keys = {f.name for f in r.fields.values()}
    assert "depthanything3.vit.embed_dim" in keys
    assert "depthanything3.vit.depth" in keys
    names = {t.name for t in r.tensors}
    assert "vit.patch_embed.weight" in names
    assert "vit.blk.0.attn_qkv.weight" in names
    assert "vit.blk.11.mlp_fc2.weight" in names
    # DualDPT depth-head main-path tensors must be present.
    for h in (
        "head.norm.weight",
        "head.proj.0.weight",
        "head.proj.3.weight",
        "head.resize.0.weight",
        "head.resize.3.weight",
        "head.scratch.layer1_rn.weight",
        "head.scratch.rn4.rc2.c1.weight",
        "head.scratch.rn1.rc1.c1.weight",
        "head.scratch.rn1.out.weight",
        "head.scratch.out1.weight",
        "head.scratch.out2a.weight",
        "head.scratch.out2b.weight",
    ):
        assert h in names, f"missing head tensor {h}"
    # No aux-head tensors should leak into M2's GGUF.
    assert not any("_aux" in n for n in names), "aux-head tensors must be skipped in M2"
    # Regression guard: DA3-BASE backbone has exactly 207 tensors; the DualDPT
    # depth head adds 62 main-path tensors (norm 2 + projects 8 + resize 6 +
    # layer*_rn 4 + refinenet1..3 30 + refinenet4 6 + output_conv1/2 6).
    BACKBONE, HEAD_MAIN = 207, 62
    assert len(r.tensors) == BACKBONE + HEAD_MAIN, (
        f"expected {BACKBONE + HEAD_MAIN} tensors, got {len(r.tensors)}")
