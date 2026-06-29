from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .external import CommandRunner, prepend_pythonpath
from .ply import PlyHeader, read_ply_header


@dataclass(frozen=True)
class DiscoverseResult:
    package_dir: Path
    scene_ply: Path | None
    ply_header: PlyHeader | None
    status: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "package_dir": str(self.package_dir),
            "scene_ply": str(self.scene_ply) if self.scene_ply else None,
            "ply_header": self.ply_header.to_jsonable() if self.ply_header else None,
            "status": self.status,
        }


def run_discoverse_real2sim(
    runner: CommandRunner,
    *,
    images_dir: Path,
    output_dir: Path,
    work_dir: Path,
    prompt: str,
    discoverse_root: Path,
    real2sim_root: Path,
    python: str,
    backend: str,
    config_file: Path | None = None,
    resume: bool = False,
    enable_prune: bool = False,
) -> DiscoverseResult:
    validate_discoverse_root(discoverse_root)
    validate_real2sim_root(real2sim_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python,
        "-m",
        "real2sim",
        "generate-scene",
        "--input",
        str(images_dir),
        "--output",
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--prompt",
        prompt,
        "--backend",
        backend,
    ]
    if config_file:
        command.extend(["--config", str(config_file)])
    if resume:
        command.append("--resume")
    if not enable_prune:
        command.append("--no-prune")

    runner.run(
        "discoverse.real2sim.generate_scene",
        command,
        cwd=real2sim_root,
        env=prepend_pythonpath(real2sim_root, discoverse_root),
    )

    scene_ply = output_dir / "3dgs" / "scene.ply"
    if runner.dry_run:
        return DiscoverseResult(output_dir, scene_ply, None, "dry_run")
    if not scene_ply.is_file():
        return DiscoverseResult(output_dir, scene_ply, None, "missing_scene_ply")
    header = read_ply_header(scene_ply)
    status = "ok" if header.has_3dgs_fields else "scene_ply_missing_3dgs_fields"
    return DiscoverseResult(output_dir, scene_ply, header, status)


def copy_final_scene(scene_ply: Path | None, final_dir: Path) -> Path | None:
    if scene_ply is None or not scene_ply.is_file():
        return None
    final_dir.mkdir(parents=True, exist_ok=True)
    dst = final_dir / "scene.ply"
    shutil.copy2(scene_ply, dst)
    return dst


def validate_discoverse_root(path: Path) -> None:
    if not (path / "discoverse").is_dir():
        raise RuntimeError(f"DISCOVERSE root is invalid or missing discoverse package: {path}")


def validate_real2sim_root(path: Path) -> None:
    if not (path / "real2sim" / "cli.py").is_file():
        raise RuntimeError(f"Real2Sim root is invalid or missing real2sim/cli.py: {path}")
