"""
Headless smoke tests of the Streamlit pages (streamlit.testing): each page
renders without raising, against an empty reviews folder and against a
review with a real run of the CLI on the synthetic test market.
"""
import datetime as dt
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import job_files  # noqa: E402
from planner import store  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")
PROFILE = store.Profile(initial_nav=1_000_000, retirement_age=60,
                        years_to_simulate=10)
SWEEP = store.Sweep(spending_floor=60_000, spending_ceil=180_000,
                    spending_step=40_000, equity_floor=0.4, equity_ceil=0.6,
                    equity_step=0.2, total_paths=200)


@pytest.fixture(scope="module")
def reviews_with_run(tmp_path_factory, market_cache):
    root = tmp_path_factory.mktemp("reviews")
    review = store.create_review("Test review", dt.date(2026, 10, 2),
                                 PROFILE, root=root)
    run = store.launch(review, SWEEP, market_cache=market_cache, workers=2)
    deadline = time.monotonic() + 120
    while store.load_run(run.path).active:
        assert time.monotonic() < deadline, "run did not finish"
        time.sleep(0.2)
    assert store.load_run(run.path).state == job_files.SUCCEEDED
    # A second, older review so History has something to compare.
    older = store.create_review("Older", dt.date(2026, 4, 1), PROFILE,
                                root=root)
    store.launch(older, SWEEP, market_cache=market_cache, workers=2)
    while store.list_runs(older)[0].active:
        assert time.monotonic() < deadline, "run did not finish"
        time.sleep(0.2)
    return root


def run_page(monkeypatch, root, page=None):
    monkeypatch.setenv(store.REVIEWS_DIR_ENV, str(root))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page is not None:
        at.switch_page(page)
        at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", [None, "views/cell.py",
                                  "views/history.py"])
def test_pages_render_without_reviews(monkeypatch, tmp_path, page):
    at = run_page(monkeypatch, tmp_path, page)
    assert at.info  # each page points the user at creating a review


def test_explorer_shows_headline(monkeypatch, reviews_with_run):
    at = run_page(monkeypatch, reviews_with_run)
    labels = [m.label for m in at.metric]
    assert "Max sustainable spending" in labels
    assert "Reference model" in labels
    assert at.dataframe  # the ranked table


def test_explorer_ceiling_changes_headline(monkeypatch, reviews_with_run):
    at = run_page(monkeypatch, reviews_with_run)
    at.sidebar.number_input(key="ceiling_pct").set_value(50.0).run()
    assert not at.exception
    loose = next(m.value for m in at.metric
                 if m.label == "Max sustainable spending")
    at.sidebar.number_input(key="ceiling_pct").set_value(0.1).run()
    strict = next(m.value for m in at.metric
                  if m.label == "Max sustainable spending")
    assert loose != strict


def test_cell_page(monkeypatch, reviews_with_run):
    at = run_page(monkeypatch, reviews_with_run, "views/cell.py")
    labels = [m.label for m in at.metric]
    # Decision and reference model KPI rows.
    assert labels.count("P(ruin)") == 2
    assert "ES10 ruin age" in labels
    # Survival, fan chart (decision + reference tabs) and histogram.
    assert "NAV percentiles by age" in [h.value for h in at.subheader]
    assert len(at.tabs) == 2


def test_history_page(monkeypatch, reviews_with_run):
    at = run_page(monkeypatch, reviews_with_run, "views/history.py")
    assert len(at.dataframe) == 2  # timeline table and review diff
