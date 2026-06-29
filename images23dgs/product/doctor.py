from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import ProductConfig, ensure_workspace


def run_doctor(config: ProductConfig) -> dict[str, Any]:
    ensure_workspace(config)
    colmap = _command_probe([str(config.colmap_binary), "-h"])
    gsplat = _command_probe([str(config.gsplat_python), "-c", "import torch; from gsplat import rasterization; print('torch+gsplat rasterization ok')"])
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


def _command_probe(command: list[str]) -> dict[str, object]:
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return {"ok": False, "summary": f"missing:{command[0]}"}
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8)
    except Exception as exc:
        return {"ok": False, "summary": str(exc)}
    first = (completed.stdout or "").splitlines()[0:2]
    return {"ok": completed.returncode == 0, "summary": " | ".join(first)}


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
