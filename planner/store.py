"""
planner/store.py

Reviews and runs on disk, and launching/cancelling runs. Independent of
Streamlit. See doc/plans/UI Design.md, "Data model: review -> runs -> cells".

    reviews/<review_id>/review.json
    reviews/<review_id>/runs/<config_hash>/{config,status,results,meta}.json

A run is launched as a detached `monte_carlo.py --job-dir` subprocess; the
UI only ever reads the files it writes (job_files.py).
"""

import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import job_files
from planner import decision

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR_ENV = "PLANNER_REVIEWS_DIR"
REVIEW_FILE = "review.json"

# A run whose config.json exists but whose CLI hasn't written status.json
# yet is "starting" for this long, then considered failed.
STARTUP_GRACE_SECONDS = 60

# Config fields that don't change which cells are simulated, excluded from
# the hash so re-running the same sweep reuses the existing run.
_UNHASHED_FIELDS = ("master_seed",)


@dataclass(frozen=True)
class Profile:
    """
    What every run of a review shares; see Review. The market assumptions
    belong here rather than on a run because cells merge across a review's
    runs: two runs under different assumptions must not be mixed. Their
    defaults are the engine's (doc/assumptions.md), so reviews created
    before the fields existed load with the assumptions they ran under.
    """
    initial_nav: float
    retirement_age: float
    years_to_simulate: float
    tax_regime: str = decision.DEFAULT_TAX_REGIME
    initial_cape: float = 34.0
    target_cape: float = 22.0
    annual_earnings_growth: float = 0.02
    annual_buyback_yield: float = 0.0
    dividend_yield: float = 0.01

    @property
    def terminal_age(self) -> float:
        return self.retirement_age + self.years_to_simulate

    @property
    def simulator_params(self) -> dict:
        "The CLI's simulator_params (constructor kwargs of the models)."
        return {"initial_cape": self.initial_cape,
                "target_cape": self.target_cape,
                "annual_earnings_growth": self.annual_earnings_growth,
                "annual_buyback_yield": self.annual_buyback_yield}


@dataclass(frozen=True)
class Sweep:
    "The ranges one run simulates."
    spending_floor: float
    spending_ceil: float
    spending_step: float
    equity_floor: float
    equity_ceil: float
    equity_step: float
    total_paths: int
    models: tuple = (decision.DECISION_MODEL, decision.REFERENCE_MODELS[0])

    @classmethod
    def from_config(cls, config: dict) -> "Sweep":
        return cls(
            spending_floor=config["yearly_spending_floor"],
            spending_ceil=config["yearly_spending_ceil"],
            spending_step=config["spend_increments"],
            equity_floor=config["equity_floor"],
            equity_ceil=config["equity_ceil"],
            equity_step=config["weight_increments"],
            total_paths=config["total_paths"],
            models=tuple(config["models"]))


@dataclass(frozen=True)
class Review:
    """
    A NAV snapshot at a point in time: label, date and the profile shared
    by all of its runs, so that their cells can be merged.
    """
    path: Path
    label: str
    date: str          # ISO date the review is for
    created_at: str
    profile: Profile
    # Who the money must last for; weights ruin by the chance someone is
    # alive (decision.Criteria). Unlike the profile it doesn't change what
    # is simulated, so it can be edited after runs exist.
    household: decision.Household = decision.Household()

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"


def reviews_root() -> Path:
    "Where reviews live: $PLANNER_REVIEWS_DIR, else reviews/ in the repo."
    return Path(os.environ.get(REVIEWS_DIR_ENV, REPO_ROOT / "reviews"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "review"


def household_to_json(household: decision.Household) -> dict:
    return {"person": asdict(household.person),
            "partner": (asdict(household.partner)
                        if household.partner is not None else None),
            "partner_age_offset": household.partner_age_offset}


def household_from_json(data: Optional[dict]) -> decision.Household:
    "None (a review from before households existed) is the default."
    if data is None:
        return decision.Household()
    partner = data.get("partner")
    return decision.Household(
        person=decision.Life(**data["person"]),
        partner=decision.Life(**partner) if partner is not None else None,
        partner_age_offset=data.get("partner_age_offset", 0.0))


def _write_review(review: Review):
    job_files.write_json_atomic(review.path / REVIEW_FILE, {
        "label": review.label, "date": review.date,
        "created_at": review.created_at, "profile": asdict(review.profile),
        "household": household_to_json(review.household)})


def load_review(path: Path) -> Review:
    data = job_files.read_json(path / REVIEW_FILE)
    if data is None:
        raise FileNotFoundError(path / REVIEW_FILE)
    return Review(path=path, label=data["label"], date=data["date"],
                  created_at=data["created_at"],
                  profile=Profile(**data["profile"]),
                  household=household_from_json(data.get("household")))


def update_household(review: Review,
                     household: decision.Household) -> Review:
    "Saves a review's household; its runs stay valid."
    updated = dataclasses.replace(review, household=household)
    _write_review(updated)
    return updated


def list_reviews(root: Optional[Path] = None) -> List[Review]:
    "All reviews, most recent first."
    root = root or reviews_root()
    if not root.is_dir():
        return []
    reviews = [load_review(p) for p in root.iterdir()
               if (p / REVIEW_FILE).is_file()]
    return sorted(reviews, key=lambda r: (r.date, r.created_at),
                  reverse=True)


def create_review(label: str, date: dt.date, profile: Profile,
                  root: Optional[Path] = None,
                  household: decision.Household = decision.Household()
                  ) -> Review:
    root = root or reviews_root()
    root.mkdir(parents=True, exist_ok=True)
    base = f"{date.isoformat()}-{_slug(label)}"
    path, n = root / base, 2
    while path.exists():
        path, n = root / f"{base}-{n}", n + 1
    (path / "runs").mkdir(parents=True)
    review = Review(path=path, label=label, date=date.isoformat(),
                    created_at=job_files.now_iso(), profile=profile,
                    household=household)
    _write_review(review)
    return review


def build_config(profile: Profile, sweep: Sweep,
                 workers: Optional[int] = None,
                 master_seed: Optional[int] = None) -> dict:
    "The CLI config (monte_carlo.py -c / config.json) for a run."
    return {
        "initial_nav": profile.initial_nav,
        "retirement_age": profile.retirement_age,
        "years_to_simulate": profile.years_to_simulate,
        "tax_regimes": [profile.tax_regime],
        "yearly_spending_floor": sweep.spending_floor,
        "yearly_spending_ceil": sweep.spending_ceil,
        "spend_increments": sweep.spending_step,
        "equity_floor": sweep.equity_floor,
        "equity_ceil": sweep.equity_ceil,
        "weight_increments": sweep.equity_step,
        "total_paths": sweep.total_paths,
        "models": list(sweep.models),
        "simulator_params": profile.simulator_params,
        "dividend_yield": profile.dividend_yield,
        # Chunking (and so every chunk's seed) depends on the worker count,
        # so it is part of what makes a run reproducible.
        "workers": workers if workers is not None else os.cpu_count(),
        "master_seed": master_seed,
    }


def config_hash(config: dict) -> str:
    "Hash of a config's canonical JSON, ignoring _UNHASHED_FIELDS."
    hashed = {k: v for k, v in config.items() if k not in _UNHASHED_FIELDS}
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _pid_alive(pid: int) -> bool:
    try:
        # Reaps the process if it is our own exited child (a zombie still
        # answers kill(pid, 0)).
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


STARTING = "starting"


@dataclass
class Run:
    path: Path
    config: dict
    status: Optional[dict]
    meta: Optional[dict]
    state: str = field(init=False)

    def __post_init__(self):
        self.state = self._derive_state()

    def _derive_state(self) -> str:
        if self.status is None:
            started = (self.path / job_files.CONFIG_FILE).stat().st_mtime
            if time.time() - started < STARTUP_GRACE_SECONDS:
                return STARTING
            return job_files.FAILED
        state = self.status["state"]
        if state == job_files.RUNNING and not _pid_alive(self.status["pid"]):
            return job_files.FAILED
        return state

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def active(self) -> bool:
        return self.state in (STARTING, job_files.RUNNING)

    @property
    def sweep(self) -> Sweep:
        return Sweep.from_config(self.config)

    @property
    def error(self) -> Optional[str]:
        if self.status is None:
            return (None if self.state == STARTING
                    else "the run never started; see log.txt")
        if (self.status["state"] == job_files.RUNNING
                and self.state == job_files.FAILED):
            return "the process exited without reporting; see log.txt"
        return self.status.get("error")

    @property
    def progress(self) -> float:
        if self.status is None or not self.status.get("cells_total"):
            return 0.0
        return self.status["cells_done"] / self.status["cells_total"]

    @property
    def started_at(self) -> Optional[str]:
        return self.status["started_at"] if self.status else None

    @property
    def finished_at(self) -> Optional[str]:
        if self.state != job_files.SUCCEEDED or self.status is None:
            return None
        return self.status["updated_at"]

    @property
    def results_path(self) -> Path:
        return self.path / job_files.RESULTS_FILE

    @property
    def log_path(self) -> Path:
        return self.path / job_files.LOG_FILE

    def results(self) -> Optional[dict]:
        if self.state != job_files.SUCCEEDED:
            return None
        return job_files.read_json(self.results_path)


def load_run(path: Path) -> Run:
    config = job_files.read_json(path / job_files.CONFIG_FILE)
    if config is None:
        raise FileNotFoundError(path / job_files.CONFIG_FILE)
    return Run(path=path, config=config,
               status=job_files.read_json(path / job_files.STATUS_FILE),
               meta=job_files.read_json(path / job_files.META_FILE))


def list_runs(review: Review) -> List[Run]:
    "A review's runs, most recently started first."
    if not review.runs_dir.is_dir():
        return []
    runs = [load_run(p) for p in review.runs_dir.iterdir()
            if (p / job_files.CONFIG_FILE).is_file()]
    return sorted(runs, key=lambda r: r.started_at or "9999", reverse=True)


def review_cells(review: Review) -> List[decision.Cell]:
    "Cells of every succeeded run in the review (unmerged)."
    cells: List[decision.Cell] = []
    for run in list_runs(review):
        results = run.results()
        if results is not None:
            cells.extend(decision.cells_from_results(
                results, run_id=run.id, finished_at=run.finished_at or ""))
    return cells


def launch(review: Review, sweep: Sweep, *,
           market_cache: Path = REPO_ROOT / "market_data.parquet",
           workers: Optional[int] = None,
           python: str = sys.executable) -> Run:
    """
    Starts a run of sweep in review, unless the same sweep already
    succeeded or is in progress there, in which case that run is returned.
    A failed or cancelled run of the same sweep is replaced.
    """
    config = build_config(review.profile, sweep, workers=workers)
    run_dir = review.runs_dir / config_hash(config)
    if (run_dir / job_files.CONFIG_FILE).is_file():
        existing = load_run(run_dir)
        if existing.active or existing.state == job_files.SUCCEEDED:
            return existing
        for name in (job_files.STATUS_FILE, job_files.RESULTS_FILE,
                     job_files.META_FILE, job_files.LOG_FILE):
            (run_dir / name).unlink(missing_ok=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    config["master_seed"] = random.randrange(1 << 32)
    job_files.write_json_atomic(run_dir / job_files.CONFIG_FILE, config)

    with open(run_dir / job_files.LOG_FILE, "w", encoding="utf-8") as log:
        subprocess.Popen(
            [python, str(REPO_ROOT / "monte_carlo.py"),
             "--job-dir", str(run_dir), "--backend", "numba",
             "-m", str(market_cache)],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    return load_run(run_dir)


def cancel(run: Run):
    """
    Kills the run's process group (the CLI and its worker processes) and
    marks it cancelled.
    """
    if run.status is None or not run.active:
        return
    try:
        os.killpg(run.status["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    status = dict(run.status, state=job_files.CANCELLED,
                  updated_at=job_files.now_iso())
    job_files.write_json_atomic(run.path / job_files.STATUS_FILE, status)
