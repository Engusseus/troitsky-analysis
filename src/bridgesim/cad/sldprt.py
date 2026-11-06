"""Read the Parasolid geometry stored inside a SolidWorks part file (``.SLDPRT``).

SolidWorks 2015+ part files are a private container: a sequence of sections, each starting
with the marker ``14 00 06 00 08 00``, then four little-endian uint32 values (flags, CRC,
compressed size, uncompressed size), the length of the section name, the name with every
byte's nibbles swapped, and a raw-deflate payload. The solid geometry lives in the section
``Contents/Config-<n>-Partition``: a short header followed by zlib streams that together
are a Parasolid *transmit file* in neutral binary form (``PS ... TRANSMIT FILE
(partition)``). :mod:`bridgesim.cad.parasolid` reads that.

No SolidWorks licence or library is needed. The format is not documented by Dassault;
this reader was written against real files and checks itself (every face must close,
every body must have a positive volume), so a file it cannot read fails loudly rather
than producing wrong geometry.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MARKER = b"\x14\x00\x06\x00\x08\x00"


@dataclass
class Section:
    name: str
    data: bytes


class SolidWorksFormatError(ValueError):
    """The file is not a SolidWorks part this reader understands."""


def _swap_nibbles(b: bytes) -> bytes:
    return bytes(((c << 4) & 0xF0) | (c >> 4) for c in b)


def read_sections(raw: bytes) -> list[Section]:
    """Every section of a SolidWorks 2015+ document, decompressed, in file order."""
    out: list[Section] = []
    i = 0
    while True:
        j = raw.find(MARKER, i)
        if j < 0:
            break
        i = j + len(MARKER)
        if j + 26 > len(raw):
            break
        _flags, _crc, csize, usize, nlen = struct.unpack("<5I", raw[j + 6 : j + 26])
        if nlen == 0 or nlen > 256 or csize > len(raw):
            continue
        name_b = _swap_nibbles(raw[j + 26 : j + 26 + nlen])
        if not name_b.isascii() or b"\x00" in name_b:
            continue
        start = j + 26 + nlen
        try:
            data = zlib.decompressobj(-15).decompress(raw[start : start + csize])
        except zlib.error:
            continue
        if len(data) != usize:
            continue
        out.append(Section(name_b.decode("ascii"), data))
        i = start + csize
    return out


def _inflate_chain(blob: bytes) -> bytes:
    """Concatenate the zlib streams of a partition section (separated by small headers)."""
    k = blob.find(b"x\x01")
    if k < 0:
        k = blob.find(b"x\x9c")
    if k < 0:
        raise SolidWorksFormatError("no zlib stream in the partition section")
    parts: list[bytes] = []
    rest = blob[k:]
    while rest:
        o = zlib.decompressobj()
        try:
            parts.append(o.decompress(rest))
        except zlib.error:
            break
        rest = o.unused_data
        if not rest:
            break
        nxt = -1
        for sig in (b"x\x01", b"x\x9c", b"x\xda"):
            p = rest.find(sig, 0, 64)
            if p >= 0 and (nxt < 0 or p < nxt):
                nxt = p
        if nxt < 0:
            break
        rest = rest[nxt:]
    return b"".join(parts)


def partition_streams(raw: bytes, config: int = 0) -> list[bytes]:
    """The Parasolid transmit streams of configuration ``config`` (the first one is the
    model; any later ones are deltas)."""
    secs = read_sections(raw)
    if not secs:
        raise SolidWorksFormatError(
            "not a SolidWorks 2015+ document (no sections found). Save it again in a recent "
            "SolidWorks, or export STL (one file, all bodies) and load that instead.")
    want = f"Contents/Config-{config}-Partition"
    cands = [s for s in secs if s.name == want and len(s.data) > 64]
    if not cands:
        names = sorted({s.name for s in secs})
        raise SolidWorksFormatError(f"no {want} section; sections present: {names}")
    blob = max(cands, key=lambda s: len(s.data)).data
    data = _inflate_chain(blob)
    streams: list[bytes] = []
    starts = [i for i in _find_all(data, b"PS\x00\x00\x00")]
    if not starts:
        raise SolidWorksFormatError("partition section holds no Parasolid transmit data")
    starts.append(len(data))
    for a, b in zip(starts, starts[1:], strict=False):
        streams.append(data[a:b])
    return streams


def _find_all(data: bytes, sub: bytes):
    i = data.find(sub)
    while i >= 0:
        yield i
        i = data.find(sub, i + 1)


def read_part(path: str | Path, config: int = 0) -> tuple[bytes, dict[str, bytes]]:
    """Return (Parasolid model stream, other useful sections) of a ``.SLDPRT`` file."""
    raw = Path(path).read_bytes()
    streams = partition_streams(raw, config)
    model = next((s for s in streams if b"(partition)" in s[:80]), streams[0])
    extras = {s.name: s.data for s in read_sections(raw)
              if s.name in ("PreviewPNG", "docProps/ISolidWorksInformation.xml")}
    return model, extras
