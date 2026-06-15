# DA3 ggml port — M3 (camera pose) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Produce camera extrinsics + intrinsics numerically equal to the reference, via the **default** pose path (`cam_dec` MLP on the last-layer camera token → `pose_encoding_to_extri_intri` → `affine_inverse`), wired into the engine/CLI as a `DEPTH_POSE` task with pose export (json), and e2e-verified vs the original.

**Architecture (from `/tmp/da3-src` cam_dec.py, model/utils/transform.py, utils/geometry.py, da3.py `_process_camera_estimation`):**
Default forward (use_ray_pose=False): `pose_enc = cam_dec(feats[-1][1])`; `c2w, ixt = pose_encoding_to_extri_intri(pose_enc,(H,W))`; `output.extrinsics = affine_inverse(c2w)`; `output.intrinsics = ixt`. The DualDPT aux ray output is computed but DELETED in this path; **M3 does not need the aux head** (that's the alternate `use_ray_pose` path — deferred). da3-base DualDPT produces no `sky`, so `_process_mono_sky_estimation` is a no-op.

- `cam_dec` input = `feats[-1][1]` = the layer-11 camera token = **the already-dumped `cam_token_11`** `[1,1,1536]`.
- `CameraDec`: `backbone = Sequential(Linear(1536,1536), ReLU, Linear(1536,1536), ReLU)`; `fc_t=Linear(1536,3)`; `fc_qvec=Linear(1536,4)`; `fc_fov=Sequential(Linear(1536,2), ReLU)`. `pose_enc = cat([t(3), qvec(4), fov(2)])` → `[9]`. (Note `feat.float()` casts before fc_*; we're f32 throughout so it's a no-op.)
- `pose_encoding_to_extri_intri(pe,(H,W))` (HOST math): `T=pe[:3]`, `quat=pe[3:7]` (XYZW scalar-last), `fov_h=pe[7]`, `fov_w=pe[8]`. `R=quat_to_mat(quat)`; `c2w=[R | T]` (3×4). `fy=(H/2)/clamp(tan(fov_h/2),1e-6)`, `fx=(W/2)/clamp(tan(fov_w/2),1e-6)`; `K=[[fx,0,W/2],[0,fy,H/2],[0,0,1]]`.
  `quat_to_mat(i,j,k,r)`: `two_s = 2/(i²+j²+k²+r²)`; `R = [[1-two_s(j²+k²), two_s(ij-kr), two_s(ik+jr)], [two_s(ij+kr), 1-two_s(i²+k²), two_s(jk-ir)], [two_s(ik-jr), two_s(jk+ir), 1-two_s(i²+j²)]]`.
- `affine_inverse(c2w 3×4)`: `R=c2w[:3,:3]`, `T=c2w[:3,3:]`; returns `[Rᵀ | -Rᵀ·T]` (3×4) = w2c. So **`output.extrinsics` = `[Rᵀ | -Rᵀ·T]` (3×4)**, `output.intrinsics = K` (3×3).

**Tensor names (HF `model.cam_dec.*` → GGUF `cam.*`):**
```
backbone.0.{weight,bias} -> cam.bb0.{w,b}    (Linear 1536->1536)
backbone.2.{weight,bias} -> cam.bb2.{w,b}    (Linear 1536->1536)   [.1,.3 = ReLU]
fc_t.{weight,bias}       -> cam.fc_t.{w,b}    (Linear 1536->3)
fc_qvec.{weight,bias}    -> cam.fc_q.{w,b}    (Linear 1536->4)
fc_fov.0.{weight,bias}   -> cam.fc_fov.{w,b}  (Linear 1536->2)      [.1 = ReLU]
```

**Parity gates:** dump `pose_enc` `[9]`, `extrinsics` `[3,4]`, `intrinsics` `[3,3]` from the real `net.forward(x)` (default args). Gate C++ at 2e-3.

---

## Task list

### M3-T1: converter cam tensors + dump pose refs
**Files:** `scripts/gguf_keys.py` (`rename_cam`, cam KV), `scripts/convert_da3_to_gguf.py` (cam loop), `scripts/dump_reference.py` (pose refs), `scripts/test_convert_roundtrip.py`.
- [ ] Add `rename_cam(name)` per the table; add KV `cam.dim_in`(1536). Regenerate header (drift test passes).
- [ ] Converter: loop `net.cam_dec.named_parameters()`, map via `rename_cam`, write f32; fail on any unmapped cam param. Print `cam_tensors=N`. (cam_enc is NOT needed — skip it entirely.)
- [ ] Dump: run `out = net(x)` (default forward; `net` is the DepthAnything3Net) under no_grad; capture `pose_enc` via a forward hook on `net.cam_dec` (its return), and `extrinsics`=out.extrinsics squeezed to (3,4), `intrinsics`=out.intrinsics squeezed to (3,3). Also dump `cam_token_in` = the exact tensor passed to cam_dec (hook its input `feats[-1][1]`) to confirm it equals `cam_token_11`. (H=W=224.)
- [ ] roundtrip test: assert `cam.bb0.weight`, `cam.fc_t.weight`, `cam.fc_q.weight`, `cam.fc_fov.weight` present.
- [ ] Verify: extrinsics is a valid pose (R orthonormal: RᵀR≈I), intrinsics has fx,fy>0. Commit scripts.

Gate: cam tensors in GGUF; pose refs dumped; extrinsics rotation orthonormal.

### M3-T2: cam_dec + pose conversion + parity gate
**Files:** `src/cam_pose.{hpp,cpp}`, `tests/test_cam_pose.cpp`.
- [ ] `CamPose(ModelLoader&, Backend&)` with `bool pose(const std::vector<float>& cam_token /*1536*/, int H, int W, std::array<float,9>& pose_enc, std::array<float,12>& extrinsics /*3x4 row-major*/, std::array<float,9>& intrinsics)`. cam_dec MLP via ggml `linear`+`relu` (input [1536,1]); then host `pose_encoding_to_extri_intri` + `affine_inverse` exactly per the architecture (quat_to_mat, clamp tan, K, Rᵀ/-RᵀT).
- [ ] Gate `test_cam_pose`: load `cam_token_11` from dump, run `pose`, compare `pose_enc` vs dumped `pose_enc`, `extrinsics` vs dumped `extrinsics`, `intrinsics` vs dumped `intrinsics`, all at 2e-3. Add `da_add_test`.
- [ ] Commit.

Gate: pose_enc + extrinsics + intrinsics match reference <2e-3 (M3 component gate).

### M3-T3: engine pose path + CLI + export + e2e
**Files:** `src/engine.{hpp,cpp}` (`pose()` from image; `depth_pose()`), `src/pose_export.{hpp,cpp}` (json), `src/cli.*`+`examples/cli/main.cpp` (`--pose <out.json>` flag on `depth`, and `TaskMode` DEPTH_POSE), `include/da_capi.h`/`src/da_capi.cpp`, `tests/test_engine_pose.cpp`, `scripts/e2e_verify.py` (+pose compare), `docs/M3-e2e.md`.
- [ ] `Engine::pose(image)`: preprocess → backbone (capture cam_token_11) → CamPose. Note: `backbone_features` currently returns only feats; extend `DinoBackbone::forward`/engine to also surface the layer-11 cam token (it's already computed as `cam_tokens[3]`). Add `Engine::depth_pose(image, depth, conf, ext, intr, H, W)`.
- [ ] CLI: `da3 depth … --pose poses.json` writes `{"extrinsics":[3x4],"intrinsics":[3x3]}`. 
- [ ] Gate `test_engine_pose`: feed dumped `raw_image` through `Engine::pose`, compare extrinsics/intrinsics to dump <5e-3 (accumulated).
- [ ] Extend `scripts/e2e_verify.py` to also compare pose (run real `net(x).extrinsics/intrinsics` vs C++ `--pose` json) on the real image; document in `docs/M3-e2e.md`.
- [ ] Commit.

Gate: full image→pose matches reference; e2e pose vs original documented.

---

## Notes
- The MLP is tiny (1 token); doing it in ggml keeps the device path uniform, but a host matmul is acceptable — pick one, gate proves it.
- quaternion is XYZW (scalar-LAST): r is the 4th element.
- `extrinsics` output is **w2c** (`affine_inverse` of c2w); don't skip the affine_inverse.
- Deferred: the `use_ray_pose` alternate path (DualDPT aux ray head + `get_extrinsic_from_camray`) — not the default; add later if needed for ray-pose checkpoints.
