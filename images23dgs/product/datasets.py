from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from images23dgs.colmap import IMAGE_SUFFIXES


DEPTH_DIR_NAMES = {"depth", "depths", "frames2", "depth_selected"}
POSE_KEYS = {"camera_to_world", "transform_matrix", "colmap_transform_matrix"}
RGB_DIR_NAMES = {"rgb", "images", "color", "colors"}


@dataclass(frozen=True)
class DatasetScan:
    path: Path
    image_count: int
    depth_count: int
    video_count: int
    has_pose: bool
    has_intrinsics: bool
    pose_source: str
    photo_risk: str
    metadata_files: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "image_count": self.image_count,
            "depth_count": self.depth_count,
            "video_count": self.video_count,
            "has_pose": self.has_pose,
            "has_intrinsics": self.has_intrinsics,
            "pose_source": self.pose_source,
            "photo_risk": self.photo_risk,
            "metadata_files": self.metadata_files,
        }


def scan_dataset(path: Path) -> DatasetScan:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and not _ignored(p)]
    depth = [p for p in files if p.suffix.lower() in {".png", ".npy", ".exr"} and _is_depth_path(p)]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES and not _is_depth_path(p)]
    videos = [p for p in files if p.suffix.lower() in {".mov", ".mp4", ".m4v"}]
    metadata = [p for p in files if p.name in {"metadata.json", "calibration.json", "transforms.json"} or p.suffix.lower() == ".jsonl"]
    pose = _detect_pose(metadata)
    intrinsics = _detect_intrinsics(metadata)
    if pose:
        risk = "低"
        pose_source = pose
    elif depth:
        risk = "中"
        pose_source = "RGBD-PnP估计"
    else:
        risk = "高"
        pose_source = "COLMAP"
    return DatasetScan(
        path=root,
        image_count=len(images),
        depth_count=len(depth),
        video_count=len(videos),
        has_pose=pose not in {"", None},
        has_intrinsics=intrinsics,
        pose_source=pose_source,
        photo_risk=risk,
        metadata_files=[str(p.relative_to(root)) if root.is_dir() else p.name for p in metadata[:20]],
    )


def import_path_dataset(source: Path, datasets_dir: Path) -> tuple[Path, DatasetScan]:
    scan = scan_dataset(source)
    dataset_dir = datasets_dir / _safe_name(source.stem if source.is_file() else source.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    marker = dataset_dir / "source_path.txt"
    marker.write_text(str(source.resolve()) + "\n", encoding="utf-8")
    (dataset_dir / "scan.json").write_text(json.dumps(scan.to_jsonable(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset_dir, scan


def ingest_upload(upload_path: Path, datasets_dir: Path) -> tuple[Path, DatasetScan]:
    dataset_dir = datasets_dir / _safe_name(upload_path.stem)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if upload_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(upload_path) as archive:
            archive.extractall(dataset_dir)
    else:
        shutil.copy2(upload_path, dataset_dir / upload_path.name)
    scan = scan_dataset(dataset_dir)
    (dataset_dir / "scan.json").write_text(json.dumps(scan.to_jsonable(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset_dir, scan


def _detect_pose(metadata_files: list[Path]) -> str:
    for path in metadata_files:
        text = _read_head(path)
        if "ARFrame.camera.transform" in text or "camera_to_world" in text:
            return "ARKit"
        if '"poses"' in text or any(key in text for key in POSE_KEYS):
            return "metadata"
    return ""


def _detect_intrinsics(metadata_files: list[Path]) -> bool:
    for path in metadata_files:
        text = _read_head(path)
        if "intrinsics" in text or "fx" in text or "cameraCalibrationData" in text or "perFrameIntrinsicCoeffs" in text or '"K"' in text:
            return True
    return False


def export_exr_rgbd_package(dataset_path: Path, output_zip: Path) -> Path:
    source = _content_root(dataset_path)
    rgb_files = _files_in_named_dirs(source, RGB_DIR_NAMES, IMAGE_SUFFIXES)
    depth_files = _files_in_named_dirs(source, DEPTH_DIR_NAMES, {".exr", ".png", ".npy"})
    if not rgb_files:
        raise RuntimeError(f"EXR_RGBD 导出失败，未找到 rgb/images 图片目录: {source}")
    if not depth_files:
        raise RuntimeError(f"EXR_RGBD 导出失败，未找到 depth/depths/frames2 目录: {source}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    metadata = _find_metadata(source)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _sort_frame_files(rgb_files):
            archive.write(path, Path("EXR_RGBD") / "rgb" / path.name)
        for path in _sort_frame_files(depth_files):
            archive.write(path, Path("EXR_RGBD") / "depth" / path.name)
        if metadata:
            archive.write(metadata, Path("EXR_RGBD") / "metadata.json")
        else:
            archive.writestr("EXR_RGBD/metadata.json", json.dumps(_basic_export_metadata(rgb_files, depth_files), indent=2) + "\n")
    return output_zip


def _read_head(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
    except OSError:
        return ""


def _safe_name(value: str) -> str:
    allowed = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value]
    return "".join(allowed).strip("._") or "dataset"


def _ignored(path: Path) -> bool:
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _is_depth_path(path: Path) -> bool:
    return any(part.lower() in DEPTH_DIR_NAMES for part in path.parts)


def _content_root(path: Path) -> Path:
    root = path.resolve()
    marker = root / "source_path.txt"
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip()).resolve()
    children = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    if len(children) == 1 and ((children[0] / "rgb").is_dir() or (children[0] / "images").is_dir()):
        return children[0].resolve()
    return root


def _files_in_named_dirs(root: Path, dir_names: set[str], suffixes: set[str]) -> list[Path]:
    files = []
    for directory in [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in dir_names]:
        files.extend(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    return files


def _sort_frame_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.name))


def _find_metadata(root: Path) -> Path | None:
    for name in ["metadata.json", "transforms.json", "calibration.json"]:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _basic_export_metadata(rgb_files: list[Path], depth_files: list[Path]) -> dict[str, Any]:
    return {
        "schema": "images23dgs.exr_rgbd.v1",
        "rgb_frames": len(rgb_files),
        "depth_frames": len(depth_files),
        "note": "Generated during EXR_RGBD export; no source pose/intrinsics metadata was available.",
    }
