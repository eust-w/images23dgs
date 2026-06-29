from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .external import CommandRunner


@dataclass(frozen=True)
class ArtifixerConfig:
    root: Path
    checkpoint: Path | None
    python: str = "python"
    render_trajectory: str = "all_frames"
    prepare_phases: str | None = None
    reconstruction_steps: int = 10000
    fetch_if_missing: bool = False


@dataclass(frozen=True)
class ArtifixerResult:
    used: bool
    prepared_scene: Path | None
    corrected_frames: Path | None
    status: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "used": self.used,
            "prepared_scene": str(self.prepared_scene) if self.prepared_scene else None,
            "corrected_frames": str(self.corrected_frames) if self.corrected_frames else None,
            "status": self.status,
        }


def should_use_artifixer(
    *,
    source_images: int,
    registered_ratio: float | None,
    min_images_for_direct: int,
    min_registered_ratio: float,
    force: bool = False,
    skip: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if skip:
        return False, ["skip_artifixer"]
    if force:
        reasons.append("force_artifixer")
    if source_images < min_images_for_direct:
        reasons.append(f"source_images<{min_images_for_direct}")
    if registered_ratio is not None and registered_ratio < min_registered_ratio:
        reasons.append(f"registered_ratio<{min_registered_ratio}")
    return bool(reasons), reasons


def run_artifixer(
    runner: CommandRunner,
    *,
    config: ArtifixerConfig,
    colmap_scene: Path,
    output_root: Path,
    scene_name: str,
    selected_images_file: Path | None = None,
) -> ArtifixerResult:
    if not config.root.exists():
        if not config.fetch_if_missing and not runner.dry_run:
            raise RuntimeError(
                f"ArtiFixer root does not exist: {config.root}. "
                "Pass --artifixer-root or --fetch-artifixer."
            )
        if config.fetch_if_missing:
            runner.run(
                "artifixer.fetch",
                [
                    "git",
                    "clone",
                    "--recurse-submodules",
                    "https://github.com/nv-tlabs/ArtiFixer.git",
                    str(config.root),
                ],
            )

    checkpoint = config.checkpoint or _checkpoint_from_env()
    if checkpoint is None:
        if not runner.dry_run:
            raise RuntimeError("ArtiFixer requires --artifixer-checkpoint or CHECKPOINT_PT")
        checkpoint = Path("${CHECKPOINT_PT}")

    prep_root = output_root / "prep" / scene_name
    save_dir = output_root / "direct"
    plus_dir = output_root / "artifixer3d_plus"
    prep_root.parent.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    plus_dir.mkdir(parents=True, exist_ok=True)

    prepare_command = [
        config.python,
        "-m",
        "data_processing.prepare_colmap_artifixer_inputs",
        "--colmap_dir",
        str(colmap_scene),
        "--output_root",
        str(prep_root),
        "--reconstruction_steps",
        str(config.reconstruction_steps),
    ]
    if config.prepare_phases:
        prepare_command.extend(["--phases", config.prepare_phases])
    if selected_images_file:
        prepare_command.extend(["--selected_image_names_file", str(selected_images_file)])
    runner.run("artifixer.prepare_colmap", prepare_command, cwd=config.root)
    if not _prepare_phases_include_caption(config.prepare_phases):
        runner.run(
            "artifixer.write_empty_caption",
            [config.python, "-c", EMPTY_CAPTION_HELPER_CODE, str(prep_root)],
            cwd=config.root,
        )

    runner.run(
        "artifixer.inference",
        [
            config.python,
            "-m",
            "model_eval.run_inference",
            "--evalset",
            "reconstructed_colmap",
            "--checkpoint_pt",
            str(checkpoint),
            "--save_dir",
            str(save_dir),
            "--split_path",
            str(prep_root / "split.json"),
            "--render_trajectory",
            config.render_trajectory,
        ],
        cwd=config.root,
    )

    if runner.dry_run:
        predicted = Path("${ARTIFIXER_FRAMES_DIR}")
    else:
        predicted = find_artifixer_pred_frames(save_dir)
        if predicted is None:
            raise RuntimeError(f"ArtiFixer did not produce predicted frames under {save_dir}")

    runner.run(
        "artifixer.artifixer3d",
        [
            config.python,
            "-m",
            "data_processing.run_artifixer3d",
            "--scene_root",
            str(prep_root),
            "--artifixer_frames_dir",
            str(predicted),
        ],
        cwd=config.root,
    )

    runner.run(
        "artifixer.artifixer3d_plus",
        [
            config.python,
            "-m",
            "model_eval.run_inference",
            "--evalset",
            "reconstructed_colmap",
            "--checkpoint_pt",
            str(checkpoint),
            "--save_dir",
            str(plus_dir),
            "--split_path",
            str(prep_root / "split_artifixer3d_plus.json"),
            "--render_trajectory",
            config.render_trajectory,
        ],
        cwd=config.root,
    )

    if runner.dry_run:
        return ArtifixerResult(True, prep_root, predicted, "dry_run")

    plus_predicted = find_artifixer_pred_frames(plus_dir)
    corrected = plus_predicted or predicted
    return ArtifixerResult(True, prep_root, corrected, "ok")


def find_artifixer_pred_frames(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.glob("**/frames/batch_0000/pred")
        if path.is_dir() and any(child.suffix.lower() in {".png", ".jpg", ".jpeg"} for child in path.iterdir())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_empty_artifixer_caption_files(prep_root: Path) -> list[Path]:
    written: list[Path] = []
    for prompt_path in _caption_paths_from_split(prep_root):
        if _write_empty_caption_h5(prompt_path):
            written.append(prompt_path)
    return written


def _caption_paths_from_split(prep_root: Path) -> list[Path]:
    split_path = prep_root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"ArtiFixer split file is missing: {split_path}")

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    prompt_paths: list[Path] = []
    for split in split_data.values():
        if not isinstance(split, dict):
            continue
        for metadata in split.values():
            if not isinstance(metadata, dict) or not metadata.get("prompt_path"):
                continue
            prompt_path = Path(str(metadata["prompt_path"]))
            if not prompt_path.is_absolute():
                prompt_path = prep_root / prompt_path
            prompt_paths.append(prompt_path)
    return prompt_paths


def _write_empty_caption_h5(path: Path) -> bool:
    if path.is_file():
        return False
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Skipping ArtiFixer caption requires h5py and numpy to write a dummy prompt file") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("empty_prompt", data=np.zeros((1, 4096), dtype=np.uint16))
        dataset.attrs["caption"] = ""
    return True


def _prepare_phases_include_caption(phases: str | None) -> bool:
    if phases is None:
        return True
    return "caption" in {phase.strip() for phase in phases.split(",") if phase.strip()}


EMPTY_CAPTION_HELPER_CODE = r"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

prep_root = Path(sys.argv[1])
split_data = json.loads((prep_root / "split.json").read_text(encoding="utf-8"))
for split in split_data.values():
    if not isinstance(split, dict):
        continue
    for metadata in split.values():
        if not isinstance(metadata, dict) or not metadata.get("prompt_path"):
            continue
        path = Path(str(metadata["prompt_path"]))
        if not path.is_absolute():
            path = prep_root / path
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("empty_prompt", data=np.zeros((1, 4096), dtype=np.uint16))
            dataset.attrs["caption"] = ""
        print(path)
"""


def _checkpoint_from_env() -> Path | None:
    value = os.environ.get("CHECKPOINT_PT")
    return Path(value) if value else None
