# Multi-stage build for the Depth Anything 3 LocalAI gRPC backend.
#
# Stage 1 clones depth-anything.cpp (pinned to DA3_COMMIT) and builds the
# static-linked shared library + the Go gRPC server. Stage 2 is a slim runtime
# image carrying just the .so, the binary and the gRPC entrypoint.
#
#   docker build -f backend/Dockerfile -t da3-backend .
#   docker run --rm -p 50051:50051 -v $PWD/models:/models da3-backend \
#       --addr 0.0.0.0:50051
#
# At runtime point LocalAI's external backend at the container's gRPC address
# and load a GGUF via LoadModel(ModelFile=/models/<file>.gguf).

ARG DA3_REPO=https://github.com/mudler/depth-anything.cpp.git
ARG DA3_COMMIT=7be8d95beda7707340e5ebc196a6ae31e2130ce8
ARG GO_VERSION=1.26

# ---------- Stage 1: build ----------
FROM golang:${GO_VERSION}-bookworm AS build
ARG DA3_REPO
ARG DA3_COMMIT

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone ${DA3_REPO} depth-anything.cpp \
    && cd depth-anything.cpp \
    && git checkout ${DA3_COMMIT} \
    && git submodule update --init --recursive --depth 1

WORKDIR /src/depth-anything.cpp

# Static-linked shared library (ggml linked statically -> no external libggml).
RUN cmake -B build-shared -S . \
        -DDA_SHARED=ON -DDA_BUILD_CLI=OFF -DDA_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && cmake --build build-shared -j"$(nproc)" \
    && ! ldd build-shared/libdepthanything.so | grep -qi ggml

# Go gRPC server.
RUN cd backend/go && go build -o /src/da3-grpc ./grpc/

# ---------- Stage 2: runtime ----------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /backend
COPY --from=build /src/depth-anything.cpp/build-shared/libdepthanything.so /backend/libdepthanything.so
COPY --from=build /src/da3-grpc /backend/da3-grpc

ENV DA3_SO=/backend/libdepthanything.so
EXPOSE 50051
ENTRYPOINT ["/backend/da3-grpc"]
CMD ["--addr", "0.0.0.0:50051"]
