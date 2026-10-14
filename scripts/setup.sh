#!/usr/bin/env bash
# Fetches the pinned BanjoRecomp checkout with all of its submodules and applies the hybrid ray tracing patch to RT64.
# Safe to run more than once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT64_DIR="$ROOT/BanjoRecomp/lib/rt64"
PATCH="$ROOT/patches/rt64-hybrid-raytracing.patch"

cd "$ROOT"
git submodule update --init --recursive BanjoRecomp

if git -C "$RT64_DIR" apply --reverse --check "$PATCH" 2> /dev/null; then
    echo "The hybrid ray tracing patch is already applied to $RT64_DIR."
else
    git -C "$RT64_DIR" apply --whitespace=nowarn "$PATCH"
    echo "Applied the hybrid ray tracing patch to $RT64_DIR."
fi
