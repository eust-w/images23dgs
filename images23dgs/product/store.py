from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class ProductStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def create_dataset(self, *, name: str, path: Path, scan: dict[str, Any]) -> dict[str, Any]:
        dataset_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into datasets(id,name,path,scan_json,created_at) values(?,?,?,?,?)",
                (dataset_id, name, str(path), json.dumps(scan, ensure_ascii=False), now),
            )
        return self.get_dataset(dataset_id)

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from datasets order by created_at desc").fetchall()
        return [self._dataset_row(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("select * from datasets where id=?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return self._dataset_row(row)

    def create_job(self, *, dataset_id: str, template: str, parameters: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "insert into jobs(id,dataset_id,template,status,parameters_json,run_dir,created_at,updated_at) values(?,?,?,?,?,?,?,?)",
                (job_id, dataset_id, template, "queued", json.dumps(parameters, ensure_ascii=False), str(run_dir), now, now),
            )
        return self.get_job(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("select * from jobs order by created_at desc").fetchall()
        return [self._job_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("select * from jobs where id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_row(row)

    def next_queued_job(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("select * from jobs where status='queued' order by created_at limit 1").fetchone()
        return self._job_row(row) if row else None

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._connect() as db:
            if job["status"] == "queued":
                db.execute("update jobs set status='canceled', cancel_requested=1, updated_at=? where id=?", (_now(), job_id))
            elif job["status"] == "running":
                db.execute("update jobs set cancel_requested=1, updated_at=? where id=?", (_now(), job_id))
        return self.get_job(job_id)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        run_dir = Path(job["run_dir"])
        retry_index = 1
        retry_dir = run_dir.with_name(f"{run_dir.name}_retry{retry_index}")
        while retry_dir.exists():
            retry_index += 1
            retry_dir = run_dir.with_name(f"{run_dir.name}_retry{retry_index}")
        return self.create_job(dataset_id=job["dataset_id"], template=job["template"], parameters=job["parameters"], run_dir=retry_dir)

    def update_job(self, job_id: str, *, status: str | None = None, error: str | None = None, result: dict[str, Any] | None = None) -> None:
        job = self.get_job(job_id)
        result_json = json.dumps(result if result is not None else job.get("result", {}), ensure_ascii=False)
        with self._connect() as db:
            db.execute(
                "update jobs set status=?, error=?, result_json=?, updated_at=? where id=?",
                (status or job["status"], error, result_json, _now(), job_id),
            )

    def _init(self) -> None:
        with self._connect() as db:
            db.execute(
                "create table if not exists datasets("
                "id text primary key, name text not null, path text not null, scan_json text not null, created_at real not null)"
            )
            db.execute(
                "create table if not exists jobs("
                "id text primary key, dataset_id text not null, template text not null, status text not null,"
                "parameters_json text not null, run_dir text not null, error text, result_json text not null default '{}',"
                "cancel_requested integer not null default 0, created_at real not null, updated_at real not null)"
            )
            columns = {row["name"] for row in db.execute("pragma table_info(jobs)").fetchall()}
            if "cancel_requested" not in columns:
                db.execute("alter table jobs add column cancel_requested integer not null default 0")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def _dataset_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "scan": json.loads(row["scan_json"]),
            "created_at": row["created_at"],
        }

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "template": row["template"],
            "status": row["status"],
            "parameters": json.loads(row["parameters_json"]),
            "run_dir": row["run_dir"],
            "error": row["error"],
            "result": json.loads(row["result_json"] or "{}"),
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _now() -> float:
    return time.time()
