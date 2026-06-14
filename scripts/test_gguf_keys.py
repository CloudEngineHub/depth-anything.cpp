import subprocess, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.gen_gguf_keys_header as G

def test_header_matches_source():
    # Generated text must equal the committed header (no drift).
    subprocess.check_call([sys.executable, str(ROOT / "scripts/gen_gguf_keys_header.py")])
    committed = (ROOT / "include/da_gguf_keys.h").read_text()
    assert committed == G.render()
    assert 'DA_KV_VIT_EMBED_DIM "depthanything3.vit.embed_dim"' in committed
    assert 'DA_ARCH "depthanything3"' in committed
