import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.gen_gguf_keys_header as G


def test_header_matches_source():
    # Compare the COMMITTED header against render() WITHOUT regenerating it, so a
    # stale committed header fails the test (the whole point of the drift guard).
    committed = (ROOT / "include/da_gguf_keys.h").read_text()
    assert committed == G.render(), "da_gguf_keys.h is stale; run scripts/gen_gguf_keys_header.py"
    assert 'DA_KV_VIT_EMBED_DIM "depthanything3.vit.embed_dim"' in committed
    assert 'DA_ARCH "depthanything3"' in committed
