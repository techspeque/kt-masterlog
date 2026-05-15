"""
Run registry — per-run manifest files for cross-tool discovery.

When kt-masterlog starts a tuning run, it writes a small JSON manifest
to the registry directory describing the run: project name, master CSV
path, writing PID, start time, status. The manifest is updated when the
run completes or fails.

This lets sister tools (e.g. kt-masterviz) discover and pick from
active or recent runs without the user typing CSV paths.

Schema (the cross-package contract — version 1):

    {
        "schema_version": 1,
        "run_id":         "<uuid4 hex>",
        "project_name":   "<config.project_name>",
        "csv_path":       "<absolute path to master CSV>",
        "pid":            <int>,
        "hostname":       "<machine hostname>",
        "started_at":     "<ISO 8601 UTC>",
        "ended_at":       null | "<ISO 8601 UTC>",
        "status":         "running" | "completed" | "failed"
    }

Readers should treat ``status == "running"`` with a dead ``pid`` as
"crashed" — this module never writes that status directly.

Registry directory resolution:
    $KT_MASTERLOG_REGISTRY_DIR if set, else ~/.kt-masterlog/runs/

Failures in this module are non-fatal — a tuning run that can't write
its manifest still proceeds; only auto-discovery by viewers is lost.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


def registry_dir() -> Path:
    """Resolve the registry directory at call time.

    Checks ``$KT_MASTERLOG_REGISTRY_DIR`` first (intended for tests and
    advanced users who want a non-default location), falls back to the
    standard ``~/.kt-masterlog/runs/``.
    """
    env = os.environ.get("KT_MASTERLOG_REGISTRY_DIR")
    if env:
        return Path(env)
    return Path.home() / ".kt-masterlog" / "runs"


@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    project_name: str
    csv_path: str
    pid: int
    hostname: str
    started_at: str
    ended_at: str | None
    status: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def register_run(project_name: str, csv_path: str | os.PathLike) -> str | None:
    """Write a ``running`` manifest for a starting tuning run.

    Returns the run_id on success, ``None`` on failure (logged but
    non-fatal — the run continues without registry support).
    """
    try:
        dir_ = registry_dir()
        dir_.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        manifest = RunManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            project_name=project_name,
            csv_path=str(Path(csv_path).resolve()),
            pid=os.getpid(),
            hostname=socket.gethostname(),
            started_at=_now_iso(),
            ended_at=None,
            status="running",
        )
        _atomic_write_json(dir_ / f"{run_id}.json", asdict(manifest))
        logger.debug("Registered run %s -> %s", run_id, dir_)
        return run_id
    except Exception:
        logger.warning("Failed to register run", exc_info=True)
        return None


def update_run_status(run_id: str | None, status: str) -> None:
    """Update an existing manifest's ``status`` and ``ended_at``.

    No-op if ``run_id`` is None or the manifest file is missing.
    Allowed status values are ``"completed"`` and ``"failed"``.
    """
    if run_id is None:
        return
    try:
        path = registry_dir() / f"{run_id}.json"
        if not path.exists():
            return
        data = json.loads(path.read_text())
        data["status"] = status
        data["ended_at"] = _now_iso()
        _atomic_write_json(path, data)
    except Exception:
        logger.warning("Failed to update run %s status", run_id, exc_info=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via temp-file + rename — atomic on POSIX."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
