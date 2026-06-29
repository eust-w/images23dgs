from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


THREEDGS_REQUIRED_FIELDS = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    format: str
    vertex_count: int
    properties: list[str]
    header_bytes: int

    @property
    def has_3dgs_fields(self) -> bool:
        return THREEDGS_REQUIRED_FIELDS.issubset(set(self.properties))

    @property
    def has_rgb_fields(self) -> bool:
        return {"red", "green", "blue"}.issubset(set(self.properties))

    def to_jsonable(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "format": self.format,
            "vertex_count": self.vertex_count,
            "property_count": len(self.properties),
            "properties": self.properties,
            "has_3dgs_fields": self.has_3dgs_fields,
            "has_rgb_fields": self.has_rgb_fields,
            "header_bytes": self.header_bytes,
        }


def read_ply_header(path: Path) -> PlyHeader:
    if not path.is_file():
        raise FileNotFoundError(path)

    header = bytearray()
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"unexpected EOF before PLY end_header: {path}")
            header.extend(line)
            if line.strip() == b"end_header":
                break

    text = header.decode("ascii", errors="replace")
    if not text.startswith("ply\n"):
        raise ValueError(f"not a PLY file: {path}")
    fmt_match = re.search(r"^format\s+(\S+)\s+1\.0$", text, re.MULTILINE)
    vertex_match = re.search(r"^element\s+vertex\s+(\d+)$", text, re.MULTILINE)
    if not fmt_match:
        raise ValueError(f"PLY format line not found: {path}")
    if not vertex_match:
        raise ValueError(f"PLY vertex element not found: {path}")

    properties: list[str] = []
    in_vertex = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["element", "vertex"]:
            in_vertex = True
            continue
        if parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and parts[0] == "property" and len(parts) >= 3:
            properties.append(parts[-1])

    return PlyHeader(
        path=path,
        format=fmt_match.group(1),
        vertex_count=int(vertex_match.group(1)),
        properties=properties,
        header_bytes=len(header),
    )
