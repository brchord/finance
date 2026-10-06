import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
import time

import pytest

import job_files
from planner import decision
from planner import store

PROFILE = store.Profile(initial_nav=1_000_000, retirement_age=60,
                        years_to_simulate=10)
SWEEP = store.Sweep(spending_floor=80_000, spending_ceil=90_000,
                    spending_step=10_000, equity_floor=0.5, equity_ceil=0.6,
                    equity_step=0.1, total_paths=8)


@pytest.fixture
def review(tmp_path):
    return store.create_review("Q4 check-in", dt.date(2026, 10, 2), PROFILE,
                               root=tmp_path)


def wait_for(predicate, seconds=120.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.2)
    pytest.fail(f"condition not met within {seconds}s")


class TestReviews:
    def test_create_and_load(self, review, tmp_path):
        assert review.id == "2026-10-02-q4-check-in"
        loaded = store.load_review(review.path)
        assert loaded == review
        assert loaded.profile.terminal_age == 70
        assert loaded.profile.tax_regime == decision.DEFAULT_TAX_REGIME

    def test_ids_are_unique(self, review, tmp_path):
        again = store.create_review("Q4 check-in", dt.date(2026, 10, 2),
                                    PROFILE, root=tmp_path)
        assert again.id == "2026-10-02-q4-check-in-2"

    def test_listed_most_recent_first(self, tmp_path):
        old = store.create_review("a", dt.date(2026, 1, 1), PROFILE,
                                  root=tmp_path)
        new = store.create_review("b", dt.date(2026, 7, 1), PROFILE,
                                  root=tmp_path)
        assert [r.id for r in store.list_reviews(tmp_path)] == [
            new.id, old.id]

    def test_missing_root_lists_nothing(self, tmp_path):
        assert store.list_reviews(tmp_path / "nope") == []


class TestConfig:
    def test_build_config_matches_cli_format(self):
        config = store.build_config(PROFILE, SWEEP, workers=3,
                                    master_seed=7)
        assert config["tax_regimes"] == [decision.DEFAULT_TAX_REGIME]
        assert config["models"] == [decision.DECISION_MODEL,
                                    decision.REFERENCE_MODELS[0]]
        assert (config["workers"], config["master_seed"]) == (3, 7)
        assert store.Sweep.from_config(config) == SWEEP

    def test_hash_ignores_seed_only(self):
        a = store.build_config(PROFILE, SWEEP, workers=3, master_seed=1)
        b = store.build_config(PROFILE, SWEEP, workers=3, master_seed=2)
        assert store.config_hash(a) == store.config_hash(b)
        c = store.build_config(PROFILE, SWEEP, workers=4, master_seed=1)
        assert store.config_hash(a) != store.config_hash(c)


class TestAssumptions:
    def test_defaults_are_the_engines(self):
        # The UI must not import simulation code, so Profile restates the
        # simulators' defaults; old reviews rely on them matching.
        import inspect
        from market_modelling.path_simulation import (
            RegimeSwitchingValuationVARSimulator)
        accepted = inspect.signature(
            RegimeSwitchingValuationVARSimulator.__init__).parameters
        for key, value in PROFILE.simulator_params.items():
            assert accepted[key].default == value, key
        assert PROFILE.dividend_yield == 0.01

    def test_config_carries_them(self):
        profile = dataclasses.replace(PROFILE, initial_cape=39.5,
                                      annual_buyback_yield=0.015,
                                      dividend_yield=0.012)
        config = store.build_config(profile, SWEEP, workers=2)
        assert config["simulator_params"] == {
            "initial_cape": 39.5, "target_cape": 22.0,
            "annual_earnings_growth": 0.02, "annual_buyback_yield": 0.015}
        assert config["dividend_yield"] == 0.012

    def test_they_are_part_of_the_hash(self):
        changed = dataclasses.replace(PROFILE, target_cape=26.0)
        assert (store.config_hash(store.build_config(PROFILE, SWEEP,
                                                     workers=2))
                != store.config_hash(store.build_config(changed, SWEEP,
                                                        workers=2)))

    def test_old_review_loads_with_engine_defaults(self, tmp_path):
        # review.json written before the assumption fields existed.
        path = tmp_path / "old"
        (path / "runs").mkdir(parents=True)
        job_files.write_json_atomic(path / store.REVIEW_FILE, {
            "label": "old", "date": "2026-04-01",
            "created_at": "2026-04-01T00:00:00",
            "profile": {"initial_nav": 1e6, "retirement_age": 60,
                        "years_to_simulate": 10,
                        "tax_regime": "pre_tcja_reversion"}})
        profile = store.load_review(path).profile
        assert profile.simulator_params == PROFILE.simulator_params
        assert profile.dividend_yield == 0.01

    def test_round_trip(self, tmp_path):
        profile = dataclasses.replace(PROFILE, initial_cape=39.5,
                                      dividend_yield=0.012)
        review = store.create_review("x", dt.date(2026, 10, 6), profile,
                                     root=tmp_path)
        assert store.load_review(review.path).profile == profile


class TestHousehold:
    COUPLE = decision.Household(
        person=decision.Life(90.0, 11.0),
        partner=decision.Life(93.0, 9.0), partner_age_offset=-4.0)

    def test_round_trip(self, tmp_path):
        review = store.create_review("x", dt.date(2026, 10, 6), PROFILE,
                                     root=tmp_path, household=self.COUPLE)
        assert store.load_review(review.path).household == self.COUPLE

    def test_old_review_gets_the_default(self, tmp_path):
        path = tmp_path / "old"
        (path / "runs").mkdir(parents=True)
        job_files.write_json_atomic(path / store.REVIEW_FILE, {
            "label": "old", "date": "2026-04-01",
            "created_at": "2026-04-01T00:00:00",
            "profile": {"initial_nav": 1e6, "retirement_age": 60,
                        "years_to_simulate": 10}})
        assert store.load_review(path).household == decision.Household()

    def test_update_keeps_runs_and_the_rest(self, review):
        run = write_run(review, status(job_files.SUCCEEDED, 1))
        updated = store.update_household(review, self.COUPLE)
        loaded = store.load_review(review.path)
        assert loaded == updated
        assert (loaded.label, loaded.profile) == (review.label,
                                                  review.profile)
        assert [r.id for r in store.list_runs(loaded)] == [run.id]

    def test_not_part_of_the_run_config(self):
        # Mortality weighs results; it doesn't change what is simulated.
        config = store.build_config(PROFILE, SWEEP, workers=2)
        assert not {"household", "person", "partner"} & set(config)


def write_run(review, status=None, age_seconds=0.0):
    run_dir = review.runs_dir / "abc"
    run_dir.mkdir(parents=True)
    config = store.build_config(PROFILE, SWEEP, workers=2, master_seed=1)
    job_files.write_json_atomic(run_dir / job_files.CONFIG_FILE, config)
    if age_seconds:
        t = time.time() - age_seconds
        os.utime(run_dir / job_files.CONFIG_FILE, (t, t))
    if status is not None:
        job_files.write_json_atomic(run_dir / job_files.STATUS_FILE, status)
    return store.load_run(run_dir)


def status(state, pid):
    return {"state": state, "pid": pid, "cells_done": 1, "cells_total": 4,
            "started_at": "2026-10-02T10:00:00",
            "updated_at": "2026-10-02T10:00:05", "error": None}


class TestRunState:
    def test_starting_then_failed_without_status(self, review):
        assert write_run(review).state == store.STARTING

    def test_never_started(self, review):
        run = write_run(review, age_seconds=store.STARTUP_GRACE_SECONDS + 5)
        assert run.state == job_files.FAILED
        assert "never started" in run.error

    def test_running_with_live_pid(self, review):
        run = write_run(review, status(job_files.RUNNING, os.getpid()))
        assert run.state == job_files.RUNNING
        assert run.active
        assert run.progress == 0.25

    def test_running_with_dead_pid_is_failed(self, review):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        run = write_run(review, status(job_files.RUNNING, proc.pid))
        assert run.state == job_files.FAILED
        assert "exited without reporting" in run.error

    def test_queued_with_dead_pid_is_failed(self, review):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        run = write_run(review, status(job_files.QUEUED, proc.pid))
        assert run.state == job_files.FAILED
        assert "exited without reporting" in run.error

    def test_queued_with_live_pid_is_active(self, review):
        run = write_run(review, status(job_files.QUEUED, os.getpid()))
        assert run.state == job_files.QUEUED
        assert run.active

    def test_exited_child_is_not_alive(self, review):
        # An exited child we haven't waited on is a zombie, which still
        # answers kill(pid, 0).
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        time.sleep(1.0)
        run = write_run(review, status(job_files.RUNNING, proc.pid))
        assert run.state == job_files.FAILED


SLEEPER = """
import json, os, sys, time
job_dir = sys.argv[sys.argv.index("--job-dir") + 1]
status = {"state": "running", "pid": os.getpid(), "cells_done": 0,
          "cells_total": 4, "started_at": "2026-10-02T10:00:00",
          "updated_at": "2026-10-02T10:00:00", "error": None}
with open(os.path.join(job_dir, "status.json"), "w") as f:
    json.dump(status, f)
time.sleep(600)
"""


class TestLaunch:
    def test_runs_the_cli_and_dedups(self, review, market_cache):
        run = store.launch(review, SWEEP, market_cache=market_cache,
                           workers=2)
        assert run.config["master_seed"] is not None
        done = wait_for(lambda: (r := store.load_run(run.path)).state
                        not in (store.STARTING, job_files.RUNNING) and r)
        assert done.state == job_files.SUCCEEDED, (
            done.log_path.read_text())
        assert done.progress == 1.0

        cells = store.review_cells(review)
        # 2 models x 2 spending x 2 equity, one tax regime
        assert len(cells) == 8
        assert {c.run_id for c in cells} == {run.id}
        assert all(c.finished_at == done.finished_at for c in cells)

        again = store.launch(review, SWEEP, market_cache=market_cache,
                             workers=2)
        assert again.path == run.path
        assert again.config["master_seed"] == run.config["master_seed"]
        assert len(store.list_runs(review)) == 1

    def test_cancel_kills_and_relaunch_replaces(
            self, review, market_cache, tmp_path):
        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text(SLEEPER)
        fake_python = tmp_path / "fake_python"
        fake_python.write_text(
            f"#!/bin/sh\nexec {sys.executable} {sleeper} \"$@\"\n")
        fake_python.chmod(0o755)

        run = store.launch(review, SWEEP, market_cache=market_cache,
                           workers=2, python=str(fake_python))
        running = wait_for(lambda: (r := store.load_run(run.path)).state
                           == job_files.RUNNING and r, seconds=30)
        pid = running.status["pid"]

        store.cancel(running)
        cancelled = store.load_run(run.path)
        assert cancelled.state == job_files.CANCELLED
        assert not cancelled.active
        wait_for(lambda: not store._pid_alive(pid), seconds=10)

        # A cancelled sweep is re-run from scratch with a new seed.
        rerun = store.launch(review, SWEEP, market_cache=market_cache,
                             workers=2, python=str(fake_python))
        assert rerun.path == run.path
        assert rerun.active
        assert rerun.config["master_seed"] != run.config["master_seed"]
        wait_for(lambda: store.load_run(run.path).state
                 == job_files.RUNNING, seconds=30)
        store.cancel(store.load_run(run.path))


class TestQueue:
    def test_launch_waits_for_the_queue_and_can_be_cancelled(
            self, review, market_cache):
        lock = review.path.parent / store.QUEUE_LOCK_FILE
        with job_files.run_lock(lock):  # a run in progress
            run = store.launch(review, SWEEP, market_cache=market_cache,
                               workers=2)
            queued = wait_for(lambda: (r := store.load_run(run.path)).state
                              == job_files.QUEUED and r, seconds=60)
            assert queued.active and queued.progress == 0.0
            assert [r.id for r in store.active_runs(review.path.parent)] \
                == [run.id]
            store.cancel(queued)
            assert store.load_run(run.path).state == job_files.CANCELLED
            pid = queued.status["pid"]
            wait_for(lambda: not store._pid_alive(pid), seconds=10)

    def test_queued_run_starts_when_the_queue_frees(self, review,
                                                    market_cache):
        lock = review.path.parent / store.QUEUE_LOCK_FILE
        with job_files.run_lock(lock):
            run = store.launch(review, SWEEP, market_cache=market_cache,
                               workers=2)
            wait_for(lambda: store.load_run(run.path).state
                     == job_files.QUEUED, seconds=60)
        done = wait_for(lambda: (r := store.load_run(run.path)).state
                        == job_files.SUCCEEDED and r)
        assert done.started_at is not None
