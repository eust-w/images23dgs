from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class CommandRecord:
    stage: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    returncode: int | None = None

    def shell_line(self) -> str:
        prefix = ""
        if self.cwd:
            prefix += f"cd {shlex.quote(self.cwd)} && "
        if self.env:
            assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items()))
            prefix += assignments + " "
        return prefix + " ".join(shlex.quote(part) for part in self.command)


class CommandRunner:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.records: list[CommandRecord] = []

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str] | None:
        env_delta = dict(env or {})
        record = CommandRecord(
            stage=stage,
            command=[str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            env=env_delta,
            dry_run=self.dry_run,
        )
        self.records.append(record)
        if self.dry_run:
            return None

        full_env = os.environ.copy()
        full_env.update(env_delta)
        completed = subprocess.run(
            record.command,
            cwd=record.cwd,
            env=full_env,
            check=check,
            text=True,
        )
        record.returncode = completed.returncode
        return completed

    def write_script(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
        ]
        last_stage = None
        for record in self.records:
            if record.stage != last_stage:
                lines.append(f"# {record.stage}")
                last_stage = record.stage
            lines.append(record.shell_line())
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        path.chmod(0o755)

    def to_jsonable(self) -> list[dict[str, object]]:
        return [
            {
                "stage": record.stage,
                "command": record.command,
                "cwd": record.cwd,
                "env": record.env,
                "dry_run": record.dry_run,
                "returncode": record.returncode,
                "shell": record.shell_line(),
            }
            for record in self.records
        ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepend_pythonpath(*paths: Path) -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH", "")
    entries = [str(path.resolve()) for path in paths if path]
    if existing:
        entries.append(existing)
    return {"PYTHONPATH": os.pathsep.join(entries)}
