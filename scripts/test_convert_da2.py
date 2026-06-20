import os, pytest, numpy as np

CKPT = "models/da2/depth_anything_v2_vitl.pth"
OUT  = "dumps/da2_vitl_test.gguf"

@pytest.mark.skipif(not os.path.exists(CKPT), reason="vitl .pth not downloaded")
def test_convert_vitl_roundtrip():
    import gguf
    from scripts.da2_reference import load_da2_model
    from scripts.convert_da2_to_gguf import write_da2_gguf
    os.makedirs("dumps", exist_ok=True)
    net = load_da2_model("vitl", CKPT)
    stats = write_da2_gguf(net, "vitl", OUT, "Depth-Anything-V2-Large")
    assert stats["unmapped"] == 0 and stats["backbone"] > 0 and stats["head"] > 0
    r = gguf.GGUFReader(OUT)
    kv = {f.name: f for f in r.fields.values()}
    def s(n): return bytes(kv[n].parts[kv[n].data[-1]]).decode()
    assert s("depthanything3.arch") == "depthanything2"
    assert s("depthanything3.img.resize_mode") == "lower_bound"
    def u(n): return int(kv[n].parts[kv[n].data[-1]][0])
    assert u("depthanything3.vit.embed_dim") == 1024
    assert u("depthanything3.head.output_dim") == 1
    assert u("depthanything3.img.resize_target") == 518
    # every depth_head.* and pretrained.* tensor must be mapped (no silent drop)
    names = {t.name for t in r.tensors}
    assert "vit.patch_embed.weight" in names and "head.scratch.out2b.weight" in names
