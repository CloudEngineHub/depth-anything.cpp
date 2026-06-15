# Makefile for the Depth Anything 3 LocalAI gRPC backend.
#
# Builds the static-linked shared library (libdepthanything.so, no external
# libggml) and the Go gRPC Backend server (da3-grpc) that wraps it via the
# purego binding.
#
# This Makefile lives inside the depth-anything.cpp repo and builds the .so
# from the parent source tree (REPO_ROOT). When this backend is vendored into
# the LocalAI monorepo it should instead clone depth-anything.cpp pinned to
# DA3_COMMIT (see README "LocalAI integration").

# Pin to a specific depth-anything.cpp commit for reproducible builds. A squash
# merge upstream can orphan a branch, so the native version is pinned by SHA.
DA3_COMMIT ?= 7be8d95beda7707340e5ebc196a6ae31e2130ce8

GOCMD     ?= go
JOBS      ?= $(shell nproc --ignore=1 2>/dev/null || echo 4)
CMAKE_ARGS ?=

# Repo root (one level up from backend/).
REPO_ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD_DIR  := $(REPO_ROOT)/build-shared
SO          := $(BUILD_DIR)/libdepthanything.so
GRPC_BIN    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))/da3-grpc

# Runtime defaults for `make run`.
DA3_SO   ?= $(SO)
ADDR     ?= 127.0.0.1:50051

.PHONY: all libso grpc run test clean ldd-check

all: libso grpc

## libso: build the static-linked shared library (ggml linked statically).
libso: $(SO)

$(SO):
	cmake -B $(BUILD_DIR) -S $(REPO_ROOT) \
		-DDA_SHARED=ON -DDA_BUILD_CLI=OFF -DDA_BUILD_TESTS=OFF \
		-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		$(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(JOBS)

## ldd-check: assert the .so does NOT pull in an external libggml.
ldd-check: $(SO)
	@if ldd $(SO) | grep -qi ggml; then \
		echo "FAIL: $(SO) links an external libggml (must be static)"; \
		ldd $(SO) | grep -i ggml; exit 1; \
	else \
		echo "OK: $(SO) is self-contained (no external libggml)"; \
	fi

## grpc: build the Go gRPC backend server.
grpc: $(GRPC_BIN)

$(GRPC_BIN):
	cd go && $(GOCMD) build -o $(GRPC_BIN) ./grpc/

## run: start the gRPC backend (DA3_SO -> .so, ADDR -> listen address).
run: grpc
	DA3_SO=$(DA3_SO) $(GRPC_BIN) --addr $(ADDR)

## test: run the Go tests (da3 binding + grpc smoke). Set DA3_SO,
## DA3_TEST_GGUF and DA3_TEST_IMAGE to exercise real inference; otherwise the
## inference tests skip.
test:
	cd go && $(GOCMD) test ./...

clean:
	rm -rf $(BUILD_DIR) $(GRPC_BIN)
