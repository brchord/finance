"""
job_files.py

The on-disk protocol between monte_carlo.py (run with --job-dir) and the
UI: file names inside a job folder, atomic JSON writes, and status.json
helpers.

Deliberately free of engine imports (numpy, numba, the simulators) so the
UI can use it without pulling simulation code into the Streamlit process.
"""

import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

CONFIG_FILE = "config.json"
STATUS_FILE = "status.json"
RESULTS_FILE = "results.json"
META_FILE = "meta.json"
LOG_FILE = "log.txt"

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"


def now_iso() -> str:
    "Local time, second resolution, ISO 8601."
    return dt.datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any, indent: Optional[int] = 2):
    """
    Writes data as JSON to a temp file in path's folder, then renames it
    over path, so a concurrent reader sees either the old file or the new
    one, never a partial write.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        # mkstemp creates the file 0600; give it a regular file's mode.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def read_json(path: Path) -> Optional[Any]:
    "Parsed JSON at path, or None if the file doesn't exist."
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def git_commit(repo_dir: Path) -> Optional[str]:
    """
    HEAD's commit hash for the repo containing repo_dir, suffixed with
    "-dirty" when tracked files have uncommitted changes. None if git or
    the repo isn't available.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=repo_dir,
            capture_output=True).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


class StatusWriter:
    """
    Maintains a job folder's status.json. Every update rewrites the whole
    file atomically.
    """

    def __init__(self, job_dir: Path, pid: int):
        self.path = Path(job_dir) / STATUS_FILE
        self.status = {
            "state": RUNNING,
            "pid": pid,
            "cells_done": 0,
            "cells_total": None,
            "started_at": now_iso(),
            "updated_at": None,
            "error": None,
        }
        self._write()

    def _write(self):
        self.status["updated_at"] = now_iso()
        write_json_atomic(self.path, self.status)

    def progress(self, cells_done: int, cells_total: int):
        "Records that cells_done of cells_total cells have finished."
        self.status["cells_done"] = cells_done
        self.status["cells_total"] = cells_total
        self._write()

    def succeeded(self):
        self.status["state"] = SUCCEEDED
        self._write()

    def failed(self, error: str):
        self.status["state"] = FAILED
        self.status["error"] = error
        self._write()
