from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DepthPointCloudResult:
    output_ply: Path
    points: int
    frames_used: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "output_ply": str(self.output_ply),
            "points": self.points,
            "frames_used": self.frames_used,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


def generate_depth_point_cloud(
    *,
    prep_root: Path,
    output_ply: Path,
    predicted_frames_dir: Path | None = None,
    max_points: int = 350_000,
    opacity_threshold: float = 0.05,
    z_sign: float = -1.0,
    seed: int = 7,
) -> DepthPointCloudResult:
    np, Image = _load_runtime_deps()
    transforms_path = _find_transforms(prep_root)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames found in {transforms_path}")

    depth_dir = _find_artifixer_output_dir(prep_root, "depth")
    renders_dir = _find_artifixer_output_dir(prep_root, "renders", required=False)
    opacity_dir = _find_artifixer_output_dir(prep_root, "opacity", required=False)
    depth_files = sorted(depth_dir.glob("*.npy"))
    if not depth_files:
        raise ValueError(f"no .npy depth files found in {depth_dir}")

    frames_by_index = _frames_by_index(frames)
    per_frame_budget = max(1, math.ceil(max_points / max(1, len(depth_files))))
    rng = np.random.default_rng(seed)
    all_points: list[Any] = []
    all_colors: list[Any] = []
    frames_used = 0

    for order, depth_path in enumerate(depth_files):
        frame_index = _index_from_stem(depth_path.stem)
        frame = frames_by_index.get(frame_index)
        if frame is None and order < len(frames):
            frame = frames[order]
        if not isinstance(frame, dict):
            continue

        depth = np.load(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype("float32", copy=False)
        height, width = depth.shape[:2]
        mask = np.isfinite(depth) & (depth > 0)
        if opacity_dir is not None:
            opacity_path = _image_for_stem(opacity_dir, depth_path.stem)
            if opacity_path is not None:
                opacity = np.asarray(Image.open(opacity_path).convert("L").resize((width, height)))
                mask &= (opacity.astype("float32") / 255.0) >= opacity_threshold
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if ys.size > per_frame_budget:
            chosen = rng.choice(ys.size, size=per_frame_budget, replace=False)
            ys = ys[chosen]
            xs = xs[chosen]

        rgb_path = _resolve_rgb_path(
            transforms_path=transforms_path,
            frame=frame,
            stem=depth_path.stem,
            predicted_frames_dir=predicted_frames_dir,
            renders_dir=renders_dir,
        )
        colors = _read_rgb(np, Image, rgb_path, width, height)
        sample_colors = colors[ys, xs]
        sample_points = _backproject(np, transforms, frame, xs, ys, depth[ys, xs], width, height, z_sign)
        all_points.append(sample_points)
        all_colors.append(sample_colors)
        frames_used += 1

    if not all_points:
        raise ValueError(f"no valid depth samples were found under {depth_dir}")
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    if points.shape[0] > max_points:
        chosen = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[chosen]
        colors = colors[chosen]

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_rgb_ply(output_ply, points, colors)
    bounds_min = tuple(float(value) for value in points.min(axis=0))
    bounds_max = tuple(float(value) for value in points.max(axis=0))
    return DepthPointCloudResult(output_ply, int(points.shape[0]), frames_used, bounds_min, bounds_max)


def _load_runtime_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("true-geometry requires numpy and Pillow") from exc
    return np, Image


def _find_transforms(prep_root: Path) -> Path:
    candidates = [
        prep_root / "transforms.json",
        *sorted((prep_root / "3dgrut_input").glob("*/nerfstudio/transforms.json")),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find nerfstudio transforms.json under {prep_root}")


def _find_artifixer_output_dir(prep_root: Path, name: str, *, required: bool = True) -> Path | None:
    candidates = sorted((prep_root / "recon_results").glob(f"*/reconstruction/*/ours_*/{name}"))
    for path in candidates:
        if path.is_dir():
            return path
    if required:
        raise FileNotFoundError(f"could not find ArtiFixer {name}/ output under {prep_root}")
    return None


def _frames_by_index(frames: list[Any]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for order, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        value = frame.get("file_path")
        index = _index_from_stem(Path(str(value)).stem) if value is not None else order
        indexed[index] = frame
    return indexed


def _index_from_stem(stem: str) -> int:
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _resolve_rgb_path(
    *,
    transforms_path: Path,
    frame: dict[str, Any],
    stem: str,
    predicted_frames_dir: Path | None,
    renders_dir: Path | None,
) -> Path | None:
    for root in [predicted_frames_dir, renders_dir]:
        if root is None:
            continue
        candidate = _image_for_stem(root, stem)
        if candidate is not None:
            return candidate
    file_path = frame.get("file_path")
    if isinstance(file_path, str) and file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = transforms_path.parent / path
        if path.is_file():
            return path
    return None


def _image_for_stem(root: Path, stem: str) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]:
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _read_rgb(np: Any, Image: Any, path: Path | None, width: int, height: int) -> Any:
    if path is None:
        return np.full((height, width, 3), 220, dtype=np.uint8)
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height))
    return np.asarray(image, dtype=np.uint8)


def _backproject(
    np: Any,
    root: dict[str, Any],
    frame: dict[str, Any],
    xs: Any,
    ys: Any,
    depth: Any,
    width: int,
    height: int,
    z_sign: float,
) -> Any:
    source_width = float(frame.get("w") or root.get("w") or width)
    source_height = float(frame.get("h") or root.get("h") or height)
    scale_x = width / source_width if source_width > 0 else 1.0
    scale_y = height / source_height if source_height > 0 else 1.0
    fx = float(frame.get("fl_x") or root.get("fl_x") or root.get("fx") or width) * scale_x
    fy = float(frame.get("fl_y") or root.get("fl_y") or root.get("fy") or fx) * scale_y
    cx = float(frame.get("cx") or root.get("cx") or (source_width * 0.5)) * scale_x
    cy = float(frame.get("cy") or root.get("cy") or (source_height * 0.5)) * scale_y

    z = depth.astype("float32") * float(z_sign)
    x = ((xs.astype("float32") - cx) / fx) * depth
    y = -((ys.astype("float32") - cy) / fy) * depth
    camera_points = np.stack([x, y, z, np.ones_like(z, dtype="float32")], axis=0)
    c2w = np.asarray(frame.get("transform_matrix"), dtype="float32")
    if c2w.shape != (4, 4):
        raise ValueError("frame transform_matrix must be 4x4")
    world = (c2w @ camera_points).T[:, :3]
    return world.astype("float32", copy=False)


def _write_binary_rgb_ply(path: Path, points: Any, colors: Any) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for point, color in zip(points, colors, strict=True):
            handle.write(struct.pack("<fffBBB", float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])))
