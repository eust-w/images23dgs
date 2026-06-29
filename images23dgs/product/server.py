import shutil
import zipfile
from pathlib import Path
from typing import Any

from .config import ProductConfig, ensure_workspace, load_config
from .datasets import import_path_dataset, ingest_upload
from .doctor import run_doctor
from .store import ProductStore
from .worker import JobWorker, TEMPLATES


def create_app(config: ProductConfig | None = None):
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import FileResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install web dependencies with: pip install 'images23dgs[web]'") from exc

    cfg = config or load_config()
    ensure_workspace(cfg)
    store = ProductStore(cfg.db_path)
    worker = JobWorker(store, cfg)

    class ImportPathRequest(BaseModel):
        path: str
        name: str | None = None

    class CreateJobRequest(BaseModel):
        dataset_id: str
        template: str = "quick_preview"
        parameters: dict[str, Any] = {}

    app = FastAPI(title="images23dgs 产品化工作台", version="0.1.0")
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.on_event("startup")
    def _startup() -> None:
        worker.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        worker.stop()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "workspace_dir": str(cfg.workspace_dir), "templates": TEMPLATES}

    @app.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        return run_doctor(cfg)

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return store.list_datasets()

    @app.post("/api/datasets/import-path")
    def import_path(payload: ImportPathRequest) -> dict[str, Any]:
        try:
            dataset_dir, scan = import_path_dataset(Path(payload.path), cfg.datasets_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return store.create_dataset(name=payload.name or dataset_dir.name, path=dataset_dir, scan=scan.to_jsonable())

    @app.post("/api/datasets/upload")
    def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        upload_path = cfg.uploads_dir / Path(file.filename or "upload.bin").name
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            dataset_dir, scan = ingest_upload(upload_path, cfg.datasets_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return store.create_dataset(name=dataset_dir.name, path=dataset_dir, scan=scan.to_jsonable())

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return store.list_jobs()

    @app.post("/api/jobs")
    def create_job(payload: CreateJobRequest) -> dict[str, Any]:
        try:
            store.get_dataset(payload.dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        if payload.template not in TEMPLATES:
            raise HTTPException(status_code=400, detail=f"unknown template: {payload.template}")
        run_dir = cfg.runs_dir / payload.dataset_id / payload.template
        index = 1
        while run_dir.exists():
            index += 1
            run_dir = cfg.runs_dir / payload.dataset_id / f"{payload.template}_{index}"
        return store.create_job(dataset_id=payload.dataset_id, template=payload.template, parameters=payload.parameters, run_dir=run_dir)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str):
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        path = Path(job["run_dir"]) / "logs" / "job.log"
        return PlainTextResponse(path.read_text(encoding="utf-8") if path.is_file() else "")

    @app.get("/api/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> list[dict[str, Any]]:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        run_dir = Path(job["run_dir"])
        artifacts = []
        for name, path, url in [
            ("run_manifest.json", run_dir / "reports" / "run_manifest.json", f"/runs/{job_id}/reports/run_manifest.json"),
            ("index.html", run_dir / "viewer" / "index.html", f"/runs/{job_id}/viewer/index.html"),
            ("source_view_qa.html", run_dir / "source_view_qa.html", f"/runs/{job_id}/source_view_qa.html"),
        ]:
            if path.is_file():
                artifacts.append({"name": name, "path": str(path), "url": url, "size": path.stat().st_size})
        if run_dir.is_dir():
            artifacts.append({"name": "run_package.zip", "path": str(run_dir), "url": f"/api/jobs/{job_id}/download", "size": _dir_size(run_dir)})
        return artifacts

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return store.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict[str, Any]:
        try:
            return store.retry_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/jobs/{job_id}/download")
    def download_job(job_id: str):
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        run_dir = Path(job["run_dir"])
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run directory not found")
        zip_path = cfg.runs_dir / f"{job_id}.zip"
        _zip_dir(run_dir, zip_path)
        return FileResponse(zip_path, filename=f"{job_id}.zip", media_type="application/zip")

    @app.get("/runs/{job_id}/viewer/index.html")
    def job_viewer(job_id: str):
        return FileResponse(_job_file(store, job_id, "viewer/index.html"))

    @app.get("/runs/{job_id}/viewer/{path:path}")
    def job_viewer_asset(job_id: str, path: str):
        return FileResponse(_job_file(store, job_id, f"viewer/{path}"))

    @app.get("/runs/{job_id}/source_view_qa.html")
    def job_source_view_qa(job_id: str):
        return FileResponse(_job_file(store, job_id, "source_view_qa.html"))

    @app.get("/runs/{job_id}/reports/run_manifest.json")
    def job_manifest(job_id: str):
        return FileResponse(_job_file(store, job_id, "reports/run_manifest.json"))

    app.mount("/runs", StaticFiles(directory=cfg.runs_dir), name="runs")
    return app


def main() -> int:
    import uvicorn

    cfg = load_config()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
    return 0


def _job_file(store: ProductStore, job_id: str, relative_path: str) -> Path:
    from fastapi import HTTPException

    try:
        job = store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    path = (Path(job["run_dir"]) / relative_path).resolve()
    run_dir = Path(job["run_dir"]).resolve()
    if run_dir not in path.parents and path != run_dir:
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


def _zip_dir(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
