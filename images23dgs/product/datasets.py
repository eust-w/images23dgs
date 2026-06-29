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
        if any(key in text for key in POSE_KEYS):
            return "metadata"
    return ""


def _detect_intrinsics(metadata_files: list[Path]) -> bool:
    for path in metadata_files:
        text = _read_head(path)
        if "intrinsics" in text or "fx" in text or "cameraCalibrationData" in text:
            return True
    return False


def _read_head(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:200_000]
    except OSError:
        return ""


def _safe_name(value: str) -> str:
    allowed = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value]
    return "".join(allowed).strip("._") or "dataset"


def _ignored(path: Path) -> bool:
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _is_depth_path(path: Path) -> bool:
    return any(part.lower() in DEPTH_DIR_NAMES for part in path.parts)
