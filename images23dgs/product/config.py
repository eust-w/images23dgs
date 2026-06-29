from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_APP_ROOT = Path(os.environ.get("IMAGES23DGS_APP_ROOT", "/opt/images23dgs_app"))


@dataclass(frozen=True)
class ProductConfig:
    app_root: Path = DEFAULT_APP_ROOT
    workspace_dir: Path = DEFAULT_APP_ROOT / "workspace"
    discoverse_root: Path = DEFAULT_APP_ROOT / "DISCOVERSE"
    real2sim_root: Path = Path("/opt/gs_playground_real2sim_48q")
    gsplat_python: Path = Path("/opt/real2sim_paper_runtime/envs/anysplat/bin/python")
    gsplat_train_script: Path = Path("/opt/gs_playground_real2sim_48q/scripts/real2sim_pose_init_gsplat_train.py")
    aholo_splat_transform_binary: Path = DEFAULT_APP_ROOT / "node" / "bin" / "splat-transform"
    aholo_convert_format: str = "spz"
    artifixer_root: Path | None = None
    colmap_binary: Path = Path("/usr/local/bin/colmap")
    host: str = "0.0.0.0"
    port: int = 18123

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / "state.db"

    @property
    def datasets_dir(self) -> Path:
        return self.workspace_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_dir / "runs"

    @property
    def uploads_dir(self) -> Path:
        return self.workspace_dir / "uploads"


def default_config_path(app_root: Path = DEFAULT_APP_ROOT) -> Path:
    return app_root / "config.toml"


def load_config(path: Path | None = None) -> ProductConfig:
    config_path = path or Path(os.environ.get("IMAGES23DGS_CONFIG", default_config_path()))
    if not config_path.is_file():
        return ProductConfig()
    values = _parse_simple_toml(config_path.read_text(encoding="utf-8"))
    app_root = Path(str(values.get("app_root", DEFAULT_APP_ROOT)))
    workspace_dir = Path(str(values.get("workspace_dir", app_root / "workspace")))
    artifixer_value = values.get("artifixer_root")
    return ProductConfig(
        app_root=app_root,
        workspace_dir=workspace_dir,
        discoverse_root=Path(str(values.get("discoverse_root", app_root / "DISCOVERSE"))),
        real2sim_root=Path(str(values.get("real2sim_root", "/opt/gs_playground_real2sim_48q"))),
        gsplat_python=Path(str(values.get("gsplat_python", "/opt/real2sim_paper_runtime/envs/anysplat/bin/python"))),
        gsplat_train_script=Path(str(values.get("gsplat_train_script", "/opt/gs_playground_real2sim_48q/scripts/real2sim_pose_init_gsplat_train.py"))),
        aholo_splat_transform_binary=Path(str(values.get("aholo_splat_transform_binary", app_root / "node" / "bin" / "splat-transform"))),
        aholo_convert_format=str(values.get("aholo_convert_format", "spz")),
        artifixer_root=Path(str(artifixer_value)) if artifixer_value else None,
        colmap_binary=Path(str(values.get("colmap_binary", "/usr/local/bin/colmap"))),
        host=str(values.get("host", "0.0.0.0")),
        port=int(values.get("port", 18123)),
    )


def write_default_config(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = ProductConfig(app_root=path.parent, workspace_dir=path.parent / "workspace")
    path.write_text(
        "\n".join(
            [
                f'app_root = "{cfg.app_root.as_posix()}"',
                f'workspace_dir = "{cfg.workspace_dir.as_posix()}"',
                f'discoverse_root = "{cfg.discoverse_root.as_posix()}"',
                f'real2sim_root = "{cfg.real2sim_root.as_posix()}"',
                f'gsplat_python = "{cfg.gsplat_python.as_posix()}"',
                f'gsplat_train_script = "{cfg.gsplat_train_script.as_posix()}"',
                f'aholo_splat_transform_binary = "{(cfg.app_root / "node" / "bin" / "splat-transform").as_posix()}"',
                f'aholo_convert_format = "{cfg.aholo_convert_format}"',
                f'colmap_binary = "{cfg.colmap_binary.as_posix()}"',
                f'host = "{cfg.host}"',
                f"port = {cfg.port}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def ensure_workspace(config: ProductConfig) -> None:
    for path in [config.workspace_dir, config.datasets_dir, config.runs_dir, config.uploads_dir]:
        path.mkdir(parents=True, exist_ok=True)


def _parse_simple_toml(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            values[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                values[key] = value
    return values
