#!/usr/bin/env bash
# Reference-oracle Python runner for the streaming-alignment validation suite
# (tasks A/B/C). Provides open3d (ICP + voxel fusion + pose-graph), gtsam
# (Sim3 pose-graph), and scipy/numpy — the independent implementations we
# differentially test our C++ against.
#
# NixOS note: the manylinux open3d wheel dlopen's X11/GL at import, which are
# not on the default loader path here. We assemble LD_LIBRARY_PATH from the
# store (cached next to the venv) so the wheel loads. gtsam pins numpy<2, so
# the whole stack is pinned to a numpy 1.26 / scipy 1.13 line.
#
# Usage:
#   scripts/oracle.sh setup             # (re)create venv + install pinned deps
#   scripts/oracle.sh python script.py  # run a script with the oracle env
#   scripts/oracle.sh -c "import open3d" # inline
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.oracle-venv"
PY="$VENV/bin/python"
LDP_CACHE="$VENV/.ldpath"

# Libraries the open3d wheel needs at import (X11/GL cluster + toolchain).
NEED_LIBS=(
  libX11.so.6 libXext.so.6 libXrandr.so.2 libXinerama.so.1 libXcursor.so.1
  libXi.so.6 libXfixes.so.3 libXrender.so.1 libGL.so.1 libEGL.so.1
  libGLX.so.0 libGLdispatch.so.0 libgomp.so.1 libstdc++.so.6 libgcc_s.so.1
  libxcb.so.1 libXau.so.6 libXdmcp.so.6
)

# True (0) iff $1 is a 64-bit ELF (byte 4 of the header == 2). The store also
# holds 32-bit multilib copies (lib32/) that must not leak onto the path.
is_elf64() {
  [ "$(od -An -t u1 -j4 -N1 "$1" 2>/dev/null | tr -d ' ')" = "2" ]
}

compute_ldpath() {
  local dirs="" cand
  for lib in "${NEED_LIBS[@]}"; do
    while IFS= read -r cand; do
      if is_elf64 "$cand"; then dirs="$dirs:$(dirname "$cand")"; break; fi
    done < <(find /nix/store -maxdepth 3 -name "$lib" 2>/dev/null)
  done
  echo "$dirs" | tr ':' '\n' | sort -u | grep -v '^$' | paste -sd:
}

ensure_ldpath() {
  if [ ! -s "$LDP_CACHE" ]; then compute_ldpath > "$LDP_CACHE"; fi
  cat "$LDP_CACHE"
}

case "${1:-python}" in
  setup)
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$PY" \
      "numpy==1.26.4" "scipy==1.13.1" "open3d==0.19.0" "gtsam==4.2.1"
    compute_ldpath > "$LDP_CACHE"
    echo "oracle env ready: $VENV"
    ;;
  python)
    shift
    LD_LIBRARY_PATH="$(ensure_ldpath):${LD_LIBRARY_PATH:-}" exec "$PY" "$@"
    ;;
  *)
    # treat all args as python args (e.g. -c "...")
    LD_LIBRARY_PATH="$(ensure_ldpath):${LD_LIBRARY_PATH:-}" exec "$PY" "$@"
    ;;
esac
