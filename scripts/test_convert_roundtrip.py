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
