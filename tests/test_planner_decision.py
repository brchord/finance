import json
from pathlib import Path

import pytest

from planner import decision as d

GOLDEN = Path(__file__).parent / "golden" / "monte_carlo_agg.json"


def cell(spending=100_000.0, equity=0.6, ruin=0, paths=10_000, es10=None,
         p10=1.0, p50=2.0, model=d.DECISION_MODEL, run_id="r",
         finished_at="2026-01-01T00:00:00", histogram=None):
    return d.Cell(
        model=model, spending=spending, equity=equity,
        tax_regime=d.DEFAULT_TAX_REGIME, total_paths=paths, ruin_count=ruin,
        ruin_month_min=None, ruin_month_es5=None, ruin_month_es10=es10,
        p5_return=0.0, p10_return=p10, p25_return=0.0, p50_return=p50,
        ruin_histogram=tuple(histogram or ()), run_id=run_id,
        finished_at=finished_at)


class TestCell:
    def test_ruin_rate_and_se(self):
        c = cell(ruin=500, paths=10_000)
        assert c.ruin_rate == 0.05
        assert c.ruin_rate_se == pytest.approx((0.05 * 0.95 / 10_000) ** 0.5)

    def test_allocation(self):
        assert cell(equity=0.6).allocation == "60/40"
        assert cell(equity=0.35).allocation == "35/65"

    def test_survival(self):
        c = cell(paths=10, histogram=[0, 1, 0, 2])
        assert c.survival() == pytest.approx([1.0, 0.9, 0.9, 0.7])

    def test_month_to_age(self):
        assert d.month_to_age(18, 42) == 43.5
        assert d.month_to_age(None, 42) is None


def test_cells_from_results_reads_cli_output():
    results = json.loads(GOLDEN.read_text())
    cells = d.cells_from_results(results, run_id="x", finished_at="t")
    n = sum(len(v) for v in results["results"].values())
    assert len(cells) == n
    first = results["results"]["HybridValuationVARSimulator"][0]
    c = next(c for c in cells if c.model == "HybridValuationVARSimulator")
    assert (c.spending, c.equity, c.tax_regime) == (
        first["spending"], first["equity"], first["tax_regime"])
    assert c.ruin_count == first["ruin_path_count"]
    assert c.total_paths == results["total_paths"]
    assert sum(c.ruin_histogram) == c.ruin_count
    assert (c.run_id, c.finished_at) == ("x", "t")
    assert c.nav_bands == first["nav_bands"]
    assert c.nav_bands["years"][0] == 0


def test_cells_without_nav_bands():
    results = json.loads(GOLDEN.read_text())
    for entries in results["results"].values():
        for e in entries:
            del e["nav_bands"]
    assert all(c.nav_bands is None for c in d.cells_from_results(results))


class TestMerge:
    def test_more_paths_wins(self):
        small = cell(paths=10_000, run_id="small", finished_at="2")
        big = cell(paths=50_000, run_id="big", finished_at="1")
        assert d.merge_cells([big, small])[big.key].run_id == "big"
        assert d.merge_cells([small, big])[big.key].run_id == "big"

    def test_newest_wins_among_equal_paths(self):
        old = cell(run_id="old", finished_at="2026-01-01")
        new = cell(run_id="new", finished_at="2026-02-01")
        assert d.merge_cells([new, old])[old.key].run_id == "new"

    def test_distinct_cells_kept(self):
        merged = d.merge_cells([cell(equity=0.5), cell(equity=0.6),
                                cell(spending=90_000.0)])
        assert len(merged) == 3


class TestCriteria:
    def test_gate_is_inclusive(self):
        c = d.Criteria(ruin_ceiling=0.05)
        assert c.passes(cell(ruin=500, paths=10_000))
        assert not c.passes(cell(ruin=501, paths=10_000))

    def test_near_ceiling(self):
        c = d.Criteria(ruin_ceiling=0.05)
        # SE at 5% with 10k paths is ~0.22pp.
        assert c.near_ceiling(cell(ruin=480, paths=10_000))
        assert not c.near_ceiling(cell(ruin=400, paths=10_000))


class TestRank:
    criteria = d.Criteria()

    def ranked(self, *cells):
        return d.rank(list(cells), self.criteria)

    def test_large_ruin_difference_beats_es10(self):
        # 4.9% ruin with a later ES10 must not beat 1.0% ruin.
        risky = cell(equity=0.8, ruin=490, es10=79 * 12)
        safe = cell(equity=0.4, ruin=100, es10=78 * 12)
        top = self.ranked(risky, safe)[0]
        assert top.cell is safe
        assert top.reason == "won on ruin rate"

    def test_es10_decides_within_ruin_tolerance(self):
        a = cell(equity=0.5, ruin=200, es10=70 * 12)
        b = cell(equity=0.6, ruin=250, es10=72 * 12)
        top = self.ranked(a, b)[0]
        assert top.cell is b
        assert top.reason == "tied on ruin rate; won on ES10 age"

    def test_no_ruin_is_best_es10(self):
        a = cell(equity=0.5, ruin=0, es10=None)
        b = cell(equity=0.6, ruin=50, es10=80 * 12)
        assert self.ranked(b, a)[0].cell is a

    def test_p10_then_p50(self):
        a = cell(equity=0.5, ruin=200, es10=70 * 12, p10=1.0, p50=3.0)
        b = cell(equity=0.6, ruin=200, es10=70.5 * 12, p10=1.1, p50=5.0)
        # Wealth multiples 2.0 vs 2.1: within 10%, so P50 decides.
        top = self.ranked(a, b)[0]
        assert top.cell is b
        assert top.reason == (
            "tied on ruin rate, ES10 age, P10 return; won on P50 return")

        c = cell(equity=0.7, ruin=200, es10=70 * 12, p10=2.0, p50=1.0)
        top = self.ranked(a, b, c)[0]
        assert top.cell is c
        assert top.reason == "tied on ruin rate, ES10 age; won on P10 return"

    def test_p10_tolerance_works_near_zero_return(self):
        # Returns of -0.02 and +0.02 are 4% apart in wealth terms: a tie.
        a = cell(equity=0.5, ruin=200, es10=70 * 12, p10=-0.02, p50=9.0)
        b = cell(equity=0.6, ruin=200, es10=70 * 12, p10=0.02, p50=1.0)
        assert self.ranked(a, b)[0].cell is a

    def test_tolerances_are_anchored_to_the_best(self):
        # a~b and b~c on ruin (within 1pp) but a and c are 1.6pp apart: c
        # is dropped at step 0 even though it has the best ES10.
        a = cell(equity=0.4, ruin=100, es10=60 * 12)
        b = cell(equity=0.5, ruin=180, es10=65 * 12)
        c = cell(equity=0.6, ruin=260, es10=90 * 12)
        order = [r.cell for r in self.ranked(a, b, c)]
        assert order == [b, a, c]

    def test_failing_cells_last_by_ruin(self):
        good = cell(equity=0.5, ruin=100)
        bad = cell(equity=0.6, ruin=900)
        worse = cell(equity=0.7, ruin=1500)
        ranked = self.ranked(worse, bad, good)
        assert [r.cell for r in ranked] == [good, bad, worse]
        assert [r.passes for r in ranked] == [True, False, False]
        assert [r.rank for r in ranked] == [1, 2, 3]


def grid(ruin_by_spending, equities=(0.5, 0.6), paths=10_000):
    """Cells whose min ruin per spending level is given (in paths)."""
    return [cell(spending=s, equity=e, ruin=r + i * 50, paths=paths,
                 es10=600.0)
            for s, r in ruin_by_spending.items()
            for i, e in enumerate(equities)]


class TestFrontier:
    criteria = d.Criteria(ruin_ceiling=0.05)

    def test_bracket_and_interpolation(self):
        f = d.frontier(grid({90_000: 200, 100_000: 400, 110_000: 600}),
                       self.criteria)
        assert f.best_spending == 100_000
        assert f.next_failing_spending == 110_000
        # 4% -> 6% crosses 5% halfway.
        assert f.estimate == pytest.approx(105_000)
        assert f.best_cell.cell.equity == 0.5
        assert not f.non_monotonic

    def test_every_level_passes(self):
        f = d.frontier(grid({90_000: 100, 100_000: 200}), self.criteria)
        assert f.best_spending == 100_000
        assert f.next_failing_spending is None
        assert f.estimate is None

    def test_no_level_passes(self):
        f = d.frontier(grid({90_000: 800, 100_000: 900}), self.criteria)
        assert f.best_spending is None
        assert f.next_failing_spending == 90_000

    def test_non_monotonic_flagged(self):
        f = d.frontier(grid({90_000: 520, 100_000: 480, 110_000: 700}),
                       self.criteria)
        assert f.best_spending == 100_000
        assert f.non_monotonic
        assert f.near_ceiling


class TestRefinement:
    criteria = d.Criteria(ruin_ceiling=0.05)

    def test_zooms_into_bracket(self):
        cells = grid({90_000: 200, 100_000: 400, 110_000: 600},
                     equities=(0.4, 0.5, 0.6))
        p = d.propose_refinement(cells, self.criteria, 10_000, 0.1)
        assert (p.spending_floor, p.spending_ceil) == (100_000, 110_000)
        assert p.spending_step == 2_500  # a quarter of 10k
        # Top two allocations at 100k are 40% and 50%.
        assert (p.equity_floor, p.equity_ceil, p.equity_step) == (
            0.35, 0.55, 0.05)

    def test_extends_upward_when_all_pass(self):
        p = d.propose_refinement(grid({90_000: 100, 100_000: 200}),
                                 self.criteria, 10_000, 0.1)
        assert (p.spending_floor, p.spending_ceil, p.spending_step) == (
            110_000, 150_000, 10_000)

    def test_extends_downward_when_none_pass(self):
        p = d.propose_refinement(grid({90_000: 800, 100_000: 900}),
                                 self.criteria, 10_000, 0.1)
        assert (p.spending_floor, p.spending_ceil, p.spending_step) == (
            40_000, 80_000, 10_000)

    @pytest.mark.parametrize("bracket,step", [
        (10_000, 2_500), (5_000, 1_000), (2_000, 500), (1_000, 500)])
    def test_spending_step_divides_bracket(self, bracket, step):
        cells = grid({100_000: 400, 100_000 + bracket: 600})
        p = d.propose_refinement(cells, self.criteria, bracket, 0.1)
        assert p.spending_step == step

    def test_nothing_finer_than_resolution(self):
        cells = grid({100_000: 400, 100_500: 600})
        assert d.propose_refinement(cells, self.criteria, 500, 0.1) is None

    def test_equity_steps_are_round(self):
        # Top two at 50% and 60%, current step 5% -> 2% grid.
        cells = grid({100_000: 400, 110_000: 600})
        p = d.propose_refinement(cells, self.criteria, 10_000, 0.05)
        assert (p.equity_floor, p.equity_ceil, p.equity_step) == (
            0.46, 0.64, 0.02)

    def test_equity_clamped_to_unit_interval(self):
        cells = grid({100_000: 400, 110_000: 600}, equities=(0.95, 1.0))
        p = d.propose_refinement(cells, self.criteria, 10_000, 0.1)
        assert p.equity_ceil == 1.0

    def test_empty(self):
        assert d.propose_refinement([], self.criteria, 10_000, 0.1) is None
