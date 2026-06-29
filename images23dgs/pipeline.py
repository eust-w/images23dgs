from __future__ import annotations

import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifixer import ArtifixerConfig, ArtifixerResult, run_artifixer, should_use_artifixer
from .colmap import (
    ColmapQuality,
    analyze_colmap_quality,
    collect_images,
    materialize_images,
    parse_colmap_registered_image_names,
    run_colmap,
)
from .discoverse import DiscoverseResult, copy_final_scene, run_discoverse_real2sim
from .external import CommandRunner, write_json
from .viewer import write_layered_viewer


DEFAULT_DISCOVERSE_ROOT = Path("/Users/d-robotics/workSpace/DISCOVERSE")
DEFAULT_REAL2SIM_ROOT = Path("/Users/d-robotics/workSpace/gs_playground")


@dataclass(frozen=True)
class PipelineConfig:
    images: Path
    output: Path
    prompt: str
    scene_name: str = "scene"
    discoverse_root: Path = DEFAULT_DISCOVERSE_ROOT
    real2sim_root: Path = DEFAULT_REAL2SIM_ROOT
    python: str = sys.executable
    backend: str = "auto"
    config_file: Path | None = None
    enable_prune: bool = False
    dry_run: bool = False
    resume: bool = False
    copy_images: bool = True
    run_colmap_stage: bool = True
    colmap_binary: str = "colmap"
    colmap_matcher: str = "exhaustive"
    colmap_use_gpu: bool = True
    colmap_num_threads: int | None = None
    colmap_max_image_size: int | None = None
    colmap_single_camera: bool = True
    min_images_for_direct: int = 24
    min_registered_ratio: float = 0.65
    force_artifixer: bool = False
    skip_artifixer: bool = False
    artifixer_root: Path | None = None
    artifixer_python: str = "python"
    artifixer_checkpoint: Path | None = None
    artifixer_render_trajectory: str = "all_frames"
    artifixer_prepare_phases: str | None = None
    artifixer_anchor_count: int | None = None
    artifixer_reconstruction_steps: int = 10000
    fetch_artifixer: bool = False


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    output = config.output.resolve()
    reports_dir = output / "reports"
    input_images_dir = output / "input" / "images"
    colmap_dir = output / "colmap"
    colmap_images_dir = colmap_dir / "images"
    artifixer_dir = output / "artifixer"
    discoverse_package = output / "discoverse_package"
    discoverse_work = output / "work" / "real2sim"
    final_dir = output / "final"
    output.mkdir(parents=True, exist_ok=True)

    source_images = collect_images(config.images)
    materialized = materialize_images(source_images, input_images_dir, copy=config.copy_images)
    ensure_colmap_scene_images(input_images_dir, colmap_images_dir)

    runner = CommandRunner(dry_run=config.dry_run)
    if config.run_colmap_stage:
        colmap_quality = run_colmap(
            runner,
            image_dir=colmap_images_dir,
            colmap_dir=colmap_dir,
            colmap_binary=config.colmap_binary,
            matcher=config.colmap_matcher,
            use_gpu=config.colmap_use_gpu,
            num_threads=config.colmap_num_threads,
            max_image_size=config.colmap_max_image_size,
            single_camera=config.colmap_single_camera,
            resume=config.resume,
        )
    else:
        colmap_quality = analyze_colmap_quality(colmap_images_dir, colmap_dir / "sparse_text" / "images.txt")

    use_artifixer, artifixer_reasons = should_use_artifixer(
        source_images=len(materialized),
        registered_ratio=colmap_quality.registered_ratio,
        min_images_for_direct=config.min_images_for_direct,
        min_registered_ratio=config.min_registered_ratio,
        force=config.force_artifixer,
        skip=config.skip_artifixer,
    )

    active_images_dir = input_images_dir
    artifixer_result = ArtifixerResult(False, None, None, "not_needed")
    if use_artifixer:
        if config.skip_artifixer:
            artifixer_result = ArtifixerResult(False, None, None, "skipped")
        else:
            sparse0 = colmap_dir / "sparse" / "0"
            if not config.dry_run and not sparse0.is_dir():
                raise RuntimeError(
                    "Artifixer selected but COLMAP sparse/0 is missing. "
                    "Run COLMAP successfully first or pass --skip-artifixer."
                )
            root = config.artifixer_root or (output / "third_party" / "ArtiFixer")
            selected_images_file = write_artifixer_anchor_selection(
                colmap_quality,
                artifixer_dir,
                anchor_count=config.artifixer_anchor_count,
                dry_run=config.dry_run,
            )
            artifixer_result = run_artifixer(
                runner,
                config=ArtifixerConfig(
                    root=root,
                    checkpoint=config.artifixer_checkpoint,
                    python=config.artifixer_python,
                    render_trajectory=config.artifixer_render_trajectory,
                    prepare_phases=config.artifixer_prepare_phases,
                    reconstruction_steps=config.artifixer_reconstruction_steps,
                    fetch_if_missing=config.fetch_artifixer,
                ),
                colmap_scene=colmap_dir,
                output_root=artifixer_dir,
                scene_name=config.scene_name,
                selected_images_file=selected_images_file,
            )
            if artifixer_result.corrected_frames:
                active_images_dir = artifixer_result.corrected_frames

    discoverse_result = run_discoverse_real2sim(
        runner,
        images_dir=active_images_dir,
        output_dir=discoverse_package,
        work_dir=discoverse_work,
        prompt=config.prompt,
        discoverse_root=config.discoverse_root,
        real2sim_root=config.real2sim_root,
        python=config.python,
        backend=config.backend,
        config_file=config.config_file,
        resume=config.resume,
        enable_prune=config.enable_prune,
    )
    final_scene = None if config.dry_run else copy_final_scene(discoverse_result.scene_ply, final_dir)
    viewer_result = write_layered_viewer(
        output_root=output,
        discoverse_result=discoverse_result,
        final_scene=final_scene,
        scene_name=config.scene_name,
    )

    runner.write_script(output / "run_commands.sh")
    manifest = {
        "schema": "images23dgs.pipeline.v1",
        "output": str(output),
        "input": {
            "source": str(config.images),
            "materialized_images": len(materialized),
            "input_images_dir": str(input_images_dir),
            "colmap_images_dir": str(colmap_images_dir),
            "active_images_dir": str(active_images_dir),
        },
        "thresholds": {
            "min_images_for_direct": config.min_images_for_direct,
            "min_registered_ratio": config.min_registered_ratio,
        },
        "colmap": colmap_quality.to_jsonable(),
        "artifixer": {
            **artifixer_result.to_jsonable(),
            "selected": use_artifixer,
            "selection_reasons": artifixer_reasons,
        },
        "discoverse": discoverse_result.to_jsonable(),
        "final_scene": str(final_scene) if final_scene else None,
        "viewer": viewer_result.to_jsonable(),
        "commands": runner.to_jsonable(),
        "dry_run": config.dry_run,
    }
    write_json(reports_dir / "run_manifest.json", manifest)
    return manifest


def ensure_colmap_scene_images(source_dir: Path, colmap_images_dir: Path) -> None:
    if colmap_images_dir.exists():
        return
    colmap_images_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        colmap_images_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, colmap_images_dir)


def write_artifixer_anchor_selection(
    colmap_quality: ColmapQuality,
    output_dir: Path,
    *,
    anchor_count: int | None,
    dry_run: bool = False,
) -> Path | None:
    if dry_run or colmap_quality.sparse_text is None or not colmap_quality.sparse_text.is_file():
        return None
    names = parse_colmap_registered_image_names(colmap_quality.sparse_text)
    anchors = choose_artifixer_anchor_names(names, anchor_count=anchor_count)
    if anchors is None:
        return None
    path = output_dir / "selected_anchor_images.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(anchors) + "\n", encoding="utf-8")
    return path


def choose_artifixer_anchor_names(names: list[str], *, anchor_count: int | None) -> list[str] | None:
    if len(names) <= 2:
        return None
    count = anchor_count if anchor_count is not None and anchor_count > 0 else min(12, max(2, len(names) // 2))
    count = min(count, len(names) - 1)
    if count >= len(names):
        return None
    if count == 1:
        return [names[0]]
    indices = [round(index * (len(names) - 1) / (count - 1)) for index in range(count)]
    deduped_indices: list[int] = []
    for index in indices:
        if index not in deduped_indices:
            deduped_indices.append(index)
    candidate = 0
    while len(deduped_indices) < count and candidate < len(names):
        if candidate not in deduped_indices:
            deduped_indices.append(candidate)
        candidate += 1
    return [names[index] for index in sorted(deduped_indices[:count])]
