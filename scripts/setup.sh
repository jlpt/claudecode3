#!/usr/bin/env bash
# Fetches the pinned BanjoRecomp checkout with all of its submodules and applies the hybrid ray tracing patch to RT64.
# Safe to run more than once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT64_DIR="$ROOT/BanjoRecomp/lib/rt64"
PATCH="$ROOT/patches/rt64-hybrid-raytracing.patch"

cd "$ROOT"

# Check the sources out exactly as they are stored (LF line endings) so the patch applies cleanly.
git -c core.autocrlf=false submodule update --init --recursive BanjoRecomp

# --ignore-whitespace covers checkouts that were already converted to CRLF line endings.
if git -C "$RT64_DIR" apply --reverse --check --ignore-whitespace "$PATCH" 2> /dev/null; then
    echo "The hybrid ray tracing patch is already applied to $RT64_DIR."
elif git -C "$RT64_DIR" apply --whitespace=nowarn "$PATCH" 2> /dev/null || git -C "$RT64_DIR" apply --whitespace=nowarn --ignore-whitespace "$PATCH"; then
    echo "Applied the hybrid ray tracing patch to $RT64_DIR."
else
    echo "Failed to apply $PATCH to $RT64_DIR." >&2
    exit 1
fi
