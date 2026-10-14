#!/usr/bin/env bash
# Builds Banjo: Recompiled with the hybrid ray tracing patch on Linux.
#
# Usage: scripts/build.sh /path/to/your/banjo-kazooie-us-1.0-rom [Release|Debug]
#
# The ROM must be the NTSC-U 1.0 release dumped from a cartridge you own (z64, n64 or v64 byte order). It is only
# read to produce the decompressed copy that N64Recomp needs, which is written inside BanjoRecomp/ and is ignored by git.
#
# Requirements (Ubuntu): sudo apt-get install git cmake ninja-build libsdl2-dev libgtk-3-dev lld llvm clang cargo
set -euo pipefail

ROM="${1:?Usage: scripts/build.sh /path/to/banjo-kazooie-us-1.0-rom [Release|Debug]}"
BUILD_TYPE="${2:-Release}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BK_DIR="$ROOT/BanjoRecomp"
TOOLS_DIR="$ROOT/build-tools"

# Versions the patch was built and tested against.
N64RECOMP_URL="https://github.com/N64Recomp/N64Recomp.git"
N64RECOMP_COMMIT="2b6f05688de2abc7d86da5b4a89b84c2c6acbabe"
BK_ROM_COMPRESSOR_URL="https://github.com/MittenzHugg/bk_rom_compressor.git"
BK_ROM_COMPRESSOR_COMMIT="d0d464e2f1ffc71497299288068a6925cd530789"

if [ ! -f "$ROM" ]; then
    echo "ROM not found: $ROM" >&2
    exit 1
fi

ROM="$(cd "$(dirname "$ROM")" && pwd)/$(basename "$ROM")"

checkout() {
    local url="$1" commit="$2" dir="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    fi

    git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
    git -C "$dir" submodule update --init --recursive
}

echo "==> Fetching BanjoRecomp and applying the patch"
"$ROOT/scripts/setup.sh"

mkdir -p "$TOOLS_DIR"

echo "==> Decompressing the ROM"
checkout "$BK_ROM_COMPRESSOR_URL" "$BK_ROM_COMPRESSOR_COMMIT" "$TOOLS_DIR/bk_rom_compressor"
cargo build --release --manifest-path "$TOOLS_DIR/bk_rom_compressor/Cargo.toml" --bin bk_rom_decompress
"$TOOLS_DIR/bk_rom_compressor/target/release/bk_rom_decompress" "$ROM" "$BK_DIR/banjo.us.v10.decompressed.z64"

echo "==> Building N64Recomp and RSPRecomp"
checkout "$N64RECOMP_URL" "$N64RECOMP_COMMIT" "$TOOLS_DIR/N64Recomp"
cmake -S "$TOOLS_DIR/N64Recomp" -B "$TOOLS_DIR/N64Recomp/cmake-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build "$TOOLS_DIR/N64Recomp/cmake-build" --target N64RecompCLI RSPRecomp -j "$(nproc)"
cp "$TOOLS_DIR/N64Recomp/cmake-build/N64Recomp" "$TOOLS_DIR/N64Recomp/cmake-build/RSPRecomp" "$BK_DIR/"

echo "==> Recompiling the game code"
(cd "$BK_DIR" && ./N64Recomp banjo.us.rev0.toml && ./RSPRecomp n_aspMain.us.rev0.toml)

echo "==> Building BanjoRecompiled"
cmake -S "$BK_DIR" -B "$BK_DIR/build-cmake" -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build "$BK_DIR/build-cmake" --target BanjoRecompiled -j "$(nproc)"

echo
echo "Done: $BK_DIR/build-cmake/BanjoRecompiled"
echo "Run it from $BK_DIR (it needs the assets folder) and pick your ROM in the launcher."
