# Depth Anything 3 — LocalAI gRPC backend

A LocalAI [Backend](https://localai.io) gRPC service wrapping the Depth
Anything 3 C++/ggml port. It loads a GGUF model through a **purego** binding
(cgo-less) over the static-linked `libdepthanything.so` and serves monocular
depth + camera pose over the LocalAI Backend gRPC contract.

Depth has no native OpenAI endpoint, so the model is exposed two ways:

| RPC | Input | Output |
|-----|-------|--------|
| `GenerateImage` | `src` image path, `dst` output path | writes a min-max normalized 8-bit grayscale depth PNG to `dst` |
| `Predict` | `Images[0]` (path or base64) | `Reply.Message` = JSON `{depth_w, depth_h, depth_min, depth_max, extrinsics[12], intrinsics[9]}` |
| `LoadModel` | `ModelFile` (GGUF path), `Threads` | loads the model |
| `Health` | — | `"OK"` |
| `Status` | — | `READY` once a model is loaded, else `UNINITIALIZED` |

The C side is **not reentrant** (shared ggml graph allocator), so all
inference is serialized behind a single mutex (matching LocalAI's
`base.SingleThread`).

## Layout

```
backend/
  Makefile        # build the .so + the Go gRPC binary
  Dockerfile      # multi-stage build (pinned to a depth-anything.cpp commit)
  gallery.yaml    # LocalAI model-gallery entry (depth-anything-3-base)
  index.yaml      # backend-matrix snippet (to merge into LocalAI/backend/index.yaml)
  metadata.json   # LocalAI backend metadata
  go/
    da3/          # purego binding to libdepthanything.so (M8-T1)
    proto/        # vendored LocalAI Backend proto + generated Go stubs
    grpc/         # the gRPC server (main.go + smoke_test.go)
```

The proto stubs in `go/proto/` are **vendored** from LocalAI
(`pkg/grpc/proto/backend{,_grpc}.pb.go`, package `proto`) so the backend builds
standalone without depending on the full LocalAI Go module. To regenerate after
a proto change: `protoc --go_out=. --go-grpc_out=. backend.proto`.

## Build

```sh
make -C backend all        # builds libdepthanything.so + backend/da3-grpc
make -C backend ldd-check  # asserts the .so has NO external libggml
```

`libso` static-links ggml (`-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON`)
so the runtime needs no `libggml.so`. The Makefile pins `DA3_COMMIT` for
reproducible builds.

## Run

```sh
DA3_SO=build-shared/libdepthanything.so ./backend/da3-grpc --addr 127.0.0.1:50051
# or:
make -C backend run ADDR=127.0.0.1:50051
```

The shared library path comes from `--so` or the `DA3_SO` env var. Then point
LocalAI at the external backend (`localai run --external-grpc-backends \
depth-anything:127.0.0.1:50051`) and `LoadModel` a GGUF.

## Test / smoke

```sh
make -C backend test     # da3 binding + grpc smoke (inference tests skip w/o envs)

# Full smoke (loads a real model + produces a depth PNG over gRPC):
cd backend/go
DA3_SO=$PWD/../../build-shared/libdepthanything.so \
DA3_TEST_GGUF=$PWD/../../models/depth-anything-base-q4_k.gguf \
DA3_TEST_IMAGE=$PWD/../../dumps/e2e_input.png \
  go test ./grpc/ -run TestSmokeBackend -v
```

The smoke test dials an in-process server and asserts `Health=OK`,
`LoadModel=Success`, `GenerateImage` writes a non-empty PNG, and `Predict`
returns JSON.

## Docker

```sh
docker build -f backend/Dockerfile -t da3-backend .
docker run --rm -p 50051:50051 -v $PWD/models:/models da3-backend --addr 0.0.0.0:50051
```

## LocalAI monorepo integration (L3–L5)

These steps register the backend in the LocalAI monorepo (out of this repo's
scope; documented here for the integrator). Mirror the existing
`locate-anything-cpp` backend by the same author:

1. **Vendor the backend** into `LocalAI/backend/go/depth-anything-cpp/` (the
   `go/` tree here), with a `Makefile` that clones depth-anything.cpp pinned to
   `DA3_COMMIT`, builds the AVX/AVX2/AVX512/fallback `.so` variants, and builds
   the Go binary. Add a `run.sh` that picks the right `.so` by `/proc/cpuinfo`
   and exports it as `DA3_SO`.
2. **Backend matrix** (`.github/backend-matrix.yml`): add `depth-anything-cpp`
   build rows (`tag-suffix: '-cpu-depth-anything-cpp'`, etc.,
   `dockerfile: ./backend/Dockerfile.golang`, `backend: depth-anything-cpp`).
3. **Backend index** (`backend/index.yaml`): add the meta + per-capability
   entries from this repo's `backend/index.yaml` snippet.
4. **Model gallery** (`gallery/index.yaml`): add the `gallery.yaml` entry here
   (fill in the published GGUF `uri`/`sha256`).
5. **Top-level Makefile**: add `$(MAKE) -C backend/go/depth-anything-cpp` to the
   backend build target and `... test` to `test-extra`.
6. **changed-backends / bump_deps**: include `depth-anything-cpp` in the
   changed-backends detection and the `bump_deps` rotation so the pinned
   `DA3_COMMIT` is bumped on release.

### Commit-SHA pinning

The native version is pinned by **commit SHA** (`DA3_COMMIT`,
`Makefile`/`Dockerfile`) rather than a branch, because a squash merge upstream
can orphan a branch. Tag the pinned commit in depth-anything.cpp and bump
`DA3_COMMIT` deliberately on each backend release.
