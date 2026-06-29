from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .external import CommandRunner


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ColmapQuality:
    source_images: int
    registered_images: int | None
    registered_ratio: float | None
    sparse_text: Path | None
    status: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "source_images": self.source_images,
            "registered_images": self.registered_images,
            "registered_ratio": self.registered_ratio,
            "sparse_text": str(self.sparse_text) if self.sparse_text else None,
            "status": self.status,
        }


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {path}")
        if _is_ignored_image_sidecar(path):
            raise ValueError(f"ignored image sidecar: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = [
        p
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not _is_ignored_image_sidecar(p)
    ]
    if not images:
        raise ValueError(f"no supported images found under {path}")
    return images


def materialize_images(images: Iterable[Path], output_dir: Path, *, copy: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for index, src in enumerate(images, start=1):
        name = src.name
        if name in seen:
            name = f"{index:04d}_{name}"
        seen.add(name)
        dst = output_dir / name
        if dst.exists():
            written.append(dst)
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        written.append(dst)
    return written


def run_colmap(
    runner: CommandRunner,
    *,
    image_dir: Path,
    colmap_dir: Path,
    colmap_binary: str = "colmap",
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    num_threads: int | None = None,
    max_image_size: int | None = None,
    single_camera: bool = True,
    resume: bool = False,
) -> ColmapQuality:
    source_count = len(collect_images(image_dir))
    sparse_text = colmap_dir / "sparse_text" / "images.txt"
    if resume and sparse_text.exists():
        return analyze_colmap_quality(image_dir, sparse_text)

    if shutil.which(colmap_binary) is None and not runner.dry_run:
        return ColmapQuality(
            source_images=source_count,
            registered_images=None,
            registered_ratio=None,
            sparse_text=None,
            status=f"missing_colmap_binary:{colmap_binary}",
        )

    database = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    text = colmap_dir / "sparse_text"
    database.parent.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    text.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"
    option_names = detect_colmap_option_names(colmap_binary, matcher=matcher)
    feature_command = [
        colmap_binary,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
        f"--{option_names.feature_prefix}.use_gpu",
        gpu_flag,
    ]
    if num_threads is not None and num_threads > 0:
        feature_command.extend([f"--{option_names.feature_prefix}.num_threads", str(num_threads)])
    if max_image_size is not None and max_image_size > 0:
        feature_command.extend([f"--{option_names.feature_prefix}.max_image_size", str(max_image_size)])
    runner.run("colmap.feature_extractor", feature_command)
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    matching_command = [
        colmap_binary,
        matcher_command,
        "--database_path",
        str(database),
        f"--{option_names.matching_prefix}.use_gpu",
        gpu_flag,
    ]
    if num_threads is not None and num_threads > 0:
        matching_command.extend([f"--{option_names.matching_prefix}.num_threads", str(num_threads)])
    runner.run(f"colmap.{matcher_command}", matching_command)
    runner.run(
        "colmap.mapper",
        [
            colmap_binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse),
        ],
    )
    if runner.dry_run:
        runner.run(
            "colmap.model_converter",
            [
                colmap_binary,
                "model_converter",
                "--input_path",
                str(sparse / "0"),
                "--output_path",
                str(text),
                "--output_type",
                "TXT",
            ],
        )
        return ColmapQuality(
            source_images=source_count,
            registered_images=None,
            registered_ratio=None,
            sparse_text=sparse_text,
            status="dry_run",
        )
    selected_sparse_text = convert_and_select_sparse_model(
        runner,
        image_dir=image_dir,
        colmap_binary=colmap_binary,
        sparse_dir=sparse,
        output_text_dir=text,
    )
    if selected_sparse_text is None:
        return ColmapQuality(
            source_images=source_count,
            registered_images=None,
            registered_ratio=None,
            sparse_text=None,
            status="missing_sparse_model",
        )
    return analyze_colmap_quality(image_dir, sparse_text)


def analyze_colmap_quality(image_dir: Path, images_txt: Path | None) -> ColmapQuality:
    source_count = len(collect_images(image_dir))
    if images_txt is None or not images_txt.is_file():
        return ColmapQuality(source_count, None, None, images_txt, "missing_sparse_text")
    registered = len(parse_colmap_registered_image_names(images_txt))
    ratio = registered / source_count if source_count else 0.0
    return ColmapQuality(source_count, registered, ratio, images_txt, "ok")


def parse_colmap_registered_image_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    for raw_line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # COLMAP image rows have:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10 and _looks_int(parts[0]) and _looks_int(parts[8]):
            names.append(" ".join(parts[9:]))
    return names


def convert_and_select_sparse_model(
    runner: CommandRunner,
    *,
    image_dir: Path,
    colmap_binary: str,
    sparse_dir: Path,
    output_text_dir: Path,
) -> Path | None:
    candidates = sorted(
        [path for path in sparse_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if not candidates:
        return None

    candidate_text_root = output_text_dir.parent / "sparse_text_candidates"
    if candidate_text_root.exists():
        shutil.rmtree(candidate_text_root)
    candidate_text_root.mkdir(parents=True, exist_ok=True)

    best_sparse: Path | None = None
    best_text: Path | None = None
    best_registered = -1
    for candidate in candidates:
        candidate_text = candidate_text_root / candidate.name
        candidate_text.mkdir(parents=True, exist_ok=True)
        runner.run(
            f"colmap.model_converter.{candidate.name}",
            [
                colmap_binary,
                "model_converter",
                "--input_path",
                str(candidate),
                "--output_path",
                str(candidate_text),
                "--output_type",
                "TXT",
            ],
        )
        quality = analyze_colmap_quality(image_dir, candidate_text / "images.txt")
        registered = quality.registered_images or 0
        if registered > best_registered:
            best_sparse = candidate
            best_text = candidate_text
            best_registered = registered

    if best_sparse is None or best_text is None:
        return None

    if output_text_dir.exists():
        shutil.rmtree(output_text_dir)
    shutil.copytree(best_text, output_text_dir)
    _replace_sparse_zero_with_best_model(sparse_dir, best_sparse)
    return output_text_dir / "images.txt"


def _replace_sparse_zero_with_best_model(sparse_dir: Path, best_sparse: Path) -> None:
    sparse0 = sparse_dir / "0"
    if best_sparse == sparse0:
        return

    original = sparse_dir / "0_original"
    if original.exists():
        shutil.rmtree(original)
    if sparse0.exists():
        shutil.move(str(sparse0), str(original))
    shutil.copytree(best_sparse, sparse0)


def _looks_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ColmapOptionNames:
    feature_prefix: str
    matching_prefix: str


def detect_colmap_option_names(colmap_binary: str, *, matcher: str = "exhaustive", allow_probe: bool = True) -> ColmapOptionNames:
    if not allow_probe:
        return ColmapOptionNames("FeatureExtraction", "FeatureMatching")

    feature_help = _colmap_help(colmap_binary, "feature_extractor")
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    matcher_help = _colmap_help(colmap_binary, matcher_command)

    feature_prefix = "FeatureExtraction" if "--FeatureExtraction.use_gpu" in feature_help else "SiftExtraction"
    matching_prefix = "FeatureMatching" if "--FeatureMatching.use_gpu" in matcher_help else "SiftMatching"
    return ColmapOptionNames(feature_prefix, matching_prefix)


def _colmap_help(colmap_binary: str, command: str) -> str:
    if shutil.which(colmap_binary) is None:
        return ""
    completed = subprocess.run(
        [colmap_binary, command, "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def _is_ignored_image_sidecar(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")
