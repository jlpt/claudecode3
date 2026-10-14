#!/usr/bin/env python3
"""Decompresses the code overlays of a Banjo-Kazooie NTSC-U 1.0 ROM for N64Recomp.

Usage: python decompress_rom.py <your rom (.z64, .n64 or .v64)> <output .z64>

This is a dependency-free port of bk_rom_decompress from https://github.com/MittenzHugg/bk_rom_compressor, so it also
runs on Windows without Rust, make or a C compiler. It produces the same file byte for byte.
"""

import hashlib
import struct
import sys
import zlib

# MD5 of the big endian NTSC-U 1.0 ROM.
USA_MD5 = "b29599651a13f681c9923d69354bf4a3"

# Start and end of the compressed code and data of every overlay, in ROM order.
USA_OFFSETS = [
    0xF19250, 0xF19250 + 0x1D09B,  # core1
    0xF37F90, 0xF37F90 + 0x64B50,  # core2
    0xFA3FD0, 0xFA3FD0 + 0x1DC6,   # whale
    0xFA5F50, 0xFA5F50 + 0x2D96,   # haunted
    0xFA9150, 0xFA9150 + 0x512E,   # desert
    0xFAE860, 0xFAE860 + 0x328B,   # beach
    0xFB24A0, 0xFB24A0 + 0x1E39,   # jungle
    0xFB44E0, 0xFB44E0 + 0x5130,   # swamp
    0xFB9A30, 0xFB9A30 + 0x4BB2,   # ship
    0xFBEBE0, 0xFBEBE0 + 0x540F,   # snow
    0xFC4810, 0xFC4810 + 0x23FF,   # training
    0xFC6F20, 0xFC6F20 + 0x1BDC,   # intro
    0xFC9150, 0xFC9150 + 0x6548,   # witch
    0xFD0420, 0xFD0420 + 0x5640,   # battle
    0xFD6190, 0xFD6190 + 0x416F,   # tree
    0xFDAA10, 0xFDAA10 + 0xE,      # coshow
    0xFDAA30,
]


def to_big_endian(rom):
    signature = rom[0:4]
    if signature == b"\x80\x37\x12\x40":
        return rom
    if signature == b"\x40\x12\x37\x80":
        swapped = bytearray(len(rom))
        for i in range(4):
            swapped[i::4] = rom[3 - i::4]
        return bytes(swapped)
    if signature == b"\x37\x80\x40\x12":
        swapped = bytearray(len(rom))
        swapped[0::2] = rom[1::2]
        swapped[1::2] = rom[0::2]
        return bytes(swapped)
    raise ValueError("this file is not an N64 ROM")


def bk_unzip(data):
    if data[0:2] != b"\x11\x72":
        raise ValueError("overlay is missing the compression header")

    expected_length = struct.unpack(">I", data[2:6])[0]
    output = zlib.decompressobj(-zlib.MAX_WBITS).decompress(data[6:])
    if len(output) != expected_length:
        raise ValueError("overlay decompressed to 0x%X bytes instead of 0x%X" % (len(output), expected_length))

    return output


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())

    with open(sys.argv[1], "rb") as rom_file:
        rom = to_big_endian(rom_file.read())

    if hashlib.md5(rom).hexdigest() != USA_MD5:
        sys.exit("This isn't the NTSC-U 1.0 Banjo-Kazooie ROM (MD5 %s)." % hashlib.md5(rom).hexdigest())

    overlays = [bk_unzip(rom[start:end]) for start, end in zip(USA_OFFSETS, USA_OFFSETS[1:])]

    # The decompressed ROM places the desert overlay before the haunted one.
    overlays[6], overlays[8] = overlays[8], overlays[6]
    overlays[7], overlays[9] = overlays[9], overlays[7]

    with open(sys.argv[2], "wb") as output_file:
        output_file.write(rom[:USA_OFFSETS[0]])
        for overlay in overlays:
            output_file.write(overlay)

    print("Wrote %s" % sys.argv[2])


if __name__ == "__main__":
    main()
