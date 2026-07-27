#!/usr/bin/env bash
# Real-clip regression for the streaming de-ghosting toggles (tasks A/B/C).
# Extracts frames from a video (ffmpeg, mirroring the demo server's fps/scale) and
# runs build/tests/stream_regress over the A/B/C variants, printing per-variant
# point count, GT-free surface thickness (ghosting proxy), and wall-clock time.
#
# Needs a pose-capable DA3 model (e.g. models/depth-anything-giant-f32.gguf).
# Build the harness first:
#   cmake --build build --target stream_regress
# Usage:
#   scripts/stream_regress.sh <video> [model.gguf] [max_frames=48] [fps=6]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="${1:?usage: stream_regress.sh <video> [model] [max_frames] [fps]}"
MODEL="${2:-$ROOT/models/depth-anything-giant-f32.gguf}"
MAXF="${3:-48}"
FPS="${4:-6}"
BIN="$ROOT/build/tests/stream_regress"

[ -x "$BIN" ] || { echo "harness missing: cmake --build build --target stream_regress" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

FR="$(mktemp -d)"
trap 'rm -rf "$FR"' EXIT
echo "extracting frames from $VIDEO (fps=$FPS) ..." >&2
ffmpeg -y -loglevel error -i "$VIDEO" -vf "fps=$FPS,scale=640:-2" "$FR/all%05d.jpg"
LD_LIBRARY_PATH="$ROOT/build:${LD_LIBRARY_PATH:-}" "$BIN" "$MODEL" "$FR" "$MAXF"
