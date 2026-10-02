import json
import os

import pytest

import job_files


class TestWriteJsonAtomic:
    def test_writes_and_replaces(self, tmp_path):
        path = tmp_path / "x.json"
        job_files.write_json_atomic(path, {"a": 1})
        job_files.write_json_atomic(path, {"a": 2})
        assert json.loads(path.read_text()) == {"a": 2}
        assert os.listdir(tmp_path) == ["x.json"]
        assert path.stat().st_mode & 0o777 == 0o644

    def test_failed_write_keeps_old_file_and_no_temp(self, tmp_path):
        path = tmp_path / "x.json"
        job_files.write_json_atomic(path, {"a": 1})
        with pytest.raises(TypeError):
            job_files.write_json_atomic(path, {"a": object()})
        assert json.loads(path.read_text()) == {"a": 1}
        assert os.listdir(tmp_path) == ["x.json"]


def test_read_json_missing_is_none(tmp_path):
    assert job_files.read_json(tmp_path / "nope.json") is None


class TestStatusWriter:
    def test_lifecycle(self, tmp_path):
        writer = job_files.StatusWriter(tmp_path, pid=42)
        status = job_files.read_json(tmp_path / job_files.STATUS_FILE)
        assert status["state"] == job_files.RUNNING
        assert status["pid"] == 42
        assert status["cells_done"] == 0 and status["cells_total"] is None

        writer.progress(3, 10)
        status = job_files.read_json(tmp_path / job_files.STATUS_FILE)
        assert (status["cells_done"], status["cells_total"]) == (3, 10)

        writer.succeeded()
        status = job_files.read_json(tmp_path / job_files.STATUS_FILE)
        assert status["state"] == job_files.SUCCEEDED

    def test_failed_records_error(self, tmp_path):
        writer = job_files.StatusWriter(tmp_path, pid=1)
        writer.failed("ValueError: boom")
        status = job_files.read_json(tmp_path / job_files.STATUS_FILE)
        assert status["state"] == job_files.FAILED
        assert status["error"] == "ValueError: boom"


def test_git_commit_outside_repo_is_none(tmp_path):
    assert job_files.git_commit(tmp_path) is None
