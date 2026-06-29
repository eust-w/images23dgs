from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import ProductConfig, ensure_workspace


def run_doctor(config: ProductConfig) -> dict[str, Any]:
    ensure_workspace(config)
    colmap = _command_probe([str(config.colmap_binary), "-h"])
    gsplat = _command_probe([str(config.gsplat_python), "-c", "import torch; from gsplat import rasterization; print('torch+gsplat rasterization ok')"])
    node_binary = _node_binary_for(config)
    node = _command_probe([str(node_binary), "--version"]) if node_binary else {"ok": False, "summary": "missing:node"}
    aholo = _command_probe([str(config.aholo_splat_transform_binary), "--version"], extra_path=config.aholo_splat_transform_binary.parent)
    nvidia = _command_probe(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"])
    real2sim_ok = (config.real2sim_root / "real2sim").exists() or (config.real2sim_root / "real2sim_video_motrix_scene.py").exists()
    disk = shutil.disk_usage(config.workspace_dir)
    payload = {
        "schema": "images23dgs.product.doctor.v1",
        "workspace_dir": str(config.workspace_dir),
        "config": {
            "real2sim_root": str(config.real2sim_root),
            "gsplat_python": str(config.gsplat_python),
            "gsplat_train_script": str(config.gsplat_train_script),
            "aholo_splat_transform_binary": str(config.aholo_splat_transform_binary),
            "aholo_convert_format": config.aholo_convert_format,
            "discoverse_root": str(config.discoverse_root),
            "colmap_binary": str(config.colmap_binary),
            "host": config.host,
            "port": config.port,
        },
        "checks": {
            "workspace_writable": _is_writable(config.workspace_dir),
            "colmap": {
                "ok": colmap["ok"],
                "path": str(config.colmap_binary),
                "summary": colmap["summary"],
            },
            "real2sim": {
                "ok": real2sim_ok,
                "path": str(config.real2sim_root),
            },
            "gsplat_training": {
                "ok": gsplat["ok"] and config.gsplat_train_script.is_file(),
                "python": str(config.gsplat_python),
                "script": str(config.gsplat_train_script),
                "summary": gsplat["summary"],
            },
            "node": {
                "ok": node["ok"] and _node_major_ok(str(node["summary"])),
                "path": str(node_binary) if node_binary else None,
                "summary": node["summary"],
            },
            "aholo_splat_transform": {
                "ok": aholo["ok"],
                "path": str(config.aholo_splat_transform_binary),
                "format": config.aholo_convert_format,
                "summary": aholo["summary"],
            },
            "discoverse": {
                "ok": config.discoverse_root.exists(),
                "path": str(config.discoverse_root),
            },
            "artifixer": {
                "ok": bool(config.artifixer_root and config.artifixer_root.exists()),
                "path": str(config.artifixer_root) if config.artifixer_root else None,
            },
            "cuda": {
                "ok": nvidia["ok"],
                "summary": nvidia["summary"],
            },
            "disk": {
                "ok": disk.free > 10 * 1024**3,
                "total_gb": round(disk.total / 1024**3, 2),
                "free_gb": round(disk.free / 1024**3, 2),
            },
        },
    }
    payload["ok"] = all(
        bool(payload["checks"][name]["ok"])
        for name in ["workspace_writable", "colmap", "real2sim", "disk"]
        if isinstance(payload["checks"].get(name), dict)
    )
    return payload


def print_doctor(config: ProductConfig) -> None:
    print(json.dumps(run_doctor(config), indent=2, ensure_ascii=False))


def _command_probe(command: list[str], *, extra_path: Path | None = None) -> dict[str, object]:
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return {"ok": False, "summary": f"missing:{command[0]}"}
    env = os.environ.copy()
    if extra_path is not None:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8, env=env)
    except Exception as exc:
        return {"ok": False, "summary": str(exc)}
    first = (completed.stdout or "").splitlines()[0:2]
    return {"ok": completed.returncode == 0, "summary": " | ".join(first)}


def _node_binary_for(config: ProductConfig) -> Path | str | None:
    bundled = config.app_root / "node" / "bin" / "node"
    if bundled.exists():
        return bundled
    found = shutil.which("node")
    return found if found else None


def _node_major_ok(summary: str) -> bool:
    token = summary.strip().split()[0] if summary.strip() else ""
    if token.startswith("v"):
        token = token[1:]
    try:
        return int(token.split(".", 1)[0]) >= 22
    except ValueError:
        return False


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
