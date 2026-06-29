from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from images23dgs.pipeline import PipelineConfig, run_pipeline
from images23dgs.rgbd import RGBDOptimizeConfig, run_rgbd_optimized

from .config import ProductConfig
from .store import ProductStore


TEMPLATES = {
    "quick_preview": {"label": "快速预览", "dry_run": True, "max_image_size": 960},
    "standard": {"label": "标准重建", "dry_run": False, "max_image_size": 1600},
    "rgbd_optimized": {"label": "RGBD优化", "dry_run": False, "max_image_size": 1280},
    "high_quality": {"label": "高质量训练", "dry_run": False, "max_image_size": 1600},
}


class JobWorker:
    def __init__(self, store: ProductStore, config: ProductConfig) -> None:
        self.store = store
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="images23dgs-product-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued_job()
            if job is None:
                time.sleep(0.5)
                continue
            self._run_job(job)

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        run_dir = Path(job["run_dir"])
        log_path = run_dir / "logs" / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store.get_job(job_id)["cancel_requested"]:
            self.store.update_job(job_id, status="canceled", error="任务已取消")
            return
        self.store.update_job(job_id, status="running")
        try:
            result = self._execute(job, log_path)
        except Exception as exc:
            log_path.write_text(log_path.read_text(encoding="utf-8") + "\n" + traceback.format_exc(), encoding="utf-8")
            self.store.update_job(job_id, status="failed", error=str(exc))
            return
        if self.store.get_job(job_id)["cancel_requested"]:
            _append_log(log_path, "任务已收到取消请求，当前阶段结束后标记为 canceled")
            self.store.update_job(job_id, status="canceled", error="任务已取消", result=result)
            return
        self.store.update_job(job_id, status="succeeded", result=result)

    def _execute(self, job: dict[str, Any], log_path: Path) -> dict[str, Any]:
        dataset = self.store.get_dataset(job["dataset_id"])
        dataset_path = Path(dataset["path"])
        source_path = _resolve_dataset_source(dataset_path)
        template = TEMPLATES.get(job["template"], TEMPLATES["quick_preview"])
        params = dict(job.get("parameters") or {})
        dry_run = bool(params.get("dry_run", template["dry_run"]))
        run_dir = Path(job["run_dir"])
        log_path.write_text(
            f"任务 {job['id']} 开始\n模板: {template['label']}\n数据集: {source_path}\ndry_run: {dry_run}\n",
            encoding="utf-8",
        )
        if job["template"] == "rgbd_optimized":
            result = run_rgbd_optimized(
                RGBDOptimizeConfig(
                    source=source_path,
                    output=run_dir,
                    scene_name=str(params.get("scene_name", dataset["name"])),
                    max_point_count=int(params.get("max_point_count", 850_000)),
                    point_stride=int(params.get("point_stride", 4)),
                    point_keep_every=int(params.get("point_keep_every", 2)),
                    trained_ply=_optional_path(params.get("trained_ply")),
                    training_metrics=_optional_path(params.get("training_metrics")),
                    training_preview=_optional_path(params.get("training_preview")),
                    training_contact_sheet=_optional_path(params.get("training_contact_sheet")),
                    train_gsplat=bool(params.get("train_gsplat", False)),
                    gsplat_python=_optional_path(params.get("gsplat_python")) or self.config.gsplat_python,
                    gsplat_train_script=_optional_path(params.get("gsplat_train_script")) or self.config.gsplat_train_script,
                    gsplat_max_steps=int(params.get("gsplat_max_steps", 200)),
                    gsplat_max_frames=int(params.get("gsplat_max_frames", 16)),
                    gsplat_image_max_size=int(params.get("gsplat_image_max_size", 448)),
                    gsplat_max_points=int(params.get("gsplat_max_points", 80_000)),
                    gsplat_target_gaussians=int(params.get("gsplat_target_gaussians", 50_000)),
                    gsplat_dense_image_points_per_frame=int(params.get("gsplat_dense_image_points_per_frame", 0)),
                    gsplat_initial_scale=float(params.get("gsplat_initial_scale", 0.0)),
                    gsplat_device=str(params.get("gsplat_device", "cuda")),
                    metadata_pose_convention=str(params.get("metadata_pose_convention", "auto")),
                    dry_run=dry_run,
                ),
                log=lambda message: _append_log(log_path, message),
            )
        else:
            manifest = run_pipeline(
                PipelineConfig(
                    images=source_path,
                    output=run_dir,
                    prompt=str(params.get("prompt", "static indoor scene")),
                    scene_name=str(params.get("scene_name", dataset["name"])),
                    discoverse_root=self.config.discoverse_root,
                    real2sim_root=self.config.real2sim_root,
                    colmap_binary=str(self.config.colmap_binary),
                    dry_run=dry_run,
                    skip_artifixer=bool(params.get("skip_artifixer", dry_run)),
                    force_artifixer=bool(params.get("force_artifixer", False)),
                    artifixer_anchor_count=_optional_int(params.get("artifixer_anchor_count")),
                    artifixer_reconstruction_steps=int(params.get("artifixer_reconstruction_steps", 10000)),
                    colmap_max_image_size=int(params.get("colmap_max_image_size", template["max_image_size"])),
                    colmap_matcher=str(params.get("colmap_matcher", "sequential")),
                    backend=str(params.get("backend", "auto")),
                )
            )
            result = {
                "manifest": str(Path(manifest["output"]) / "reports" / "run_manifest.json"),
                "viewer": manifest.get("viewer", {}),
                "dry_run": dry_run,
            }
        _write_artifact_index(run_dir)
        log_path.write_text(log_path.read_text(encoding="utf-8") + "\n任务完成\n", encoding="utf-8")
        return result


def _resolve_dataset_source(dataset_path: Path) -> Path:
    marker = dataset_path / "source_path.txt"
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    return dataset_path


def _append_log(path: Path, message: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + message.rstrip() + "\n", encoding="utf-8")


def _optional_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _write_artifact_index(run_dir: Path) -> None:
    artifacts = []
    for path in [run_dir / "reports" / "run_manifest.json", run_dir / "viewer" / "index.html", run_dir / "source_view_qa.html"]:
        if path.is_file():
            artifacts.append({"name": path.name, "path": str(path), "size": path.stat().st_size})
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "index.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
