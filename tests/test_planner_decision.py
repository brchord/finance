import dataclasses
import json
from pathlib import Path

import pytest

from planner import decision as d

GOLDEN = Path(__file__).parent / "golden" / "monte_carlo_agg.json"


def cell(spending=100_000.0, equity=0.6, ruin=0, paths=10_000, es10=None,
         p10=1.0, p50=2.0, model=d.DECISION_MODEL, run_id="r",
         finished_at="2026-01-01T00:00:00", histogram=None):
    """
    Without a histogram, all `ruin` paths are ruined in month 0 of a
    10-year horizon, when everyone is alive: lifetime ruin then equals
    the ruin rate.
    """
    if histogram is None:
        histogram = [ruin] + [0] * 119
    ruin = int(sum(histogram))
    return d.Cell(
        model=model, spending=spending, equity=equity,
        tax_regime=d.DEFAULT_TAX_REGIME, total_paths=paths, ruin_count=ruin,
        ruin_month_min=None, ruin_month_es5=None, ruin_month_es10=es10,
        p5_return=0.0, p10_return=p10, p25_return=0.0, p50_return=p50,
        ruin_histogram=tuple(histogram), run_id=run_id,
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


REAL_KEYS = ("real_nav_bands", "p5_real_return", "p10_real_return",
             "p25_real_return", "p50_real_return", "ruin_prob_by_age")


def test_cells_read_real_metrics():
    results = json.loads(GOLDEN.read_text())
    for entries in results["results"].values():
        for i, e in enumerate(entries):
            e["real_nav_bands"] = {"years": [0], "p50": [float(i)]}
            for q in (5, 10, 25, 50):
                e[f"p{q}_real_return"] = q / 100.0 + i
            e["ruin_prob_by_age"] = {"85": i / 1000.0}
    for c in d.cells_from_results(results):
        entry = next(e for e in results["results"][c.model]
                     if (e["spending"], e["equity"], e["tax_regime"]) ==
                     (c.spending, c.equity, c.tax_regime))
        for key in REAL_KEYS:
            assert getattr(c, key) == entry[key], key


def test_cells_without_real_metrics():
    results = json.loads(GOLDEN.read_text())
    for entries in results["results"].values():
        for e in entries:
            for key in REAL_KEYS:
                e.pop(key, None)
    for c in d.cells_from_results(results):
        assert all(getattr(c, key) is None for key in REAL_KEYS)


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


def ruined_at(month, count, months=480):
    "A histogram with `count` ruins in `month` (age 65 + month / 12)."
    histogram = [0] * months
    histogram[month] = count
    return histogram


class TestRank:
    criteria = d.Criteria()  # retirement at 65, default mortality

    def ranked(self, *cells):
        return d.rank(list(cells), self.criteria)

    def test_lower_lifetime_ruin_wins(self):
        risky = cell(equity=0.8, ruin=300)
        safe = cell(equity=0.4, ruin=100)
        top = self.ranked(risky, safe)[0]
        assert top.cell is safe
        assert top.reason == "won on lifetime ruin"
        assert top.decided_by == "lifetime ruin"

    def test_ruin_you_likely_wont_live_to_counts_less(self):
        # 9% of paths ruined at 100 fail a 5% ceiling on ruin by the
        # horizon, but few people live to 100: lifetime ruin is far lower,
        # and below 5% ruined at 65.
        late = cell(equity=0.8, histogram=ruined_at(420, 900))
        early = cell(equity=0.4, histogram=ruined_at(0, 500))
        assert late.ruin_rate == 0.09
        assert self.criteria.ruin(late) < 0.2 * late.ruin_rate
        assert self.ranked(early, late)[0].cell is late

    def test_years_in_ruin_decides_within_ruin_tolerance(self):
        # Same count; ruin at 85 is less likely to be lived through (and
        # shorter) than at 65. Lifetime ruin is within 1pp, so years in
        # ruin decides.
        at_65 = cell(equity=0.5, histogram=ruined_at(0, 150))
        at_85 = cell(equity=0.6, histogram=ruined_at(240, 150))
        assert (self.criteria.ruin(at_65) - self.criteria.ruin(at_85)
                < self.criteria.ruin_tolerance)
        top = self.ranked(at_65, at_85)[0]
        assert top.cell is at_85
        assert top.reason == "tied on lifetime ruin; won on years in ruin"

    def test_exact_tie_goes_to_the_safer_cell(self):
        # 0 vs 0.5% ruin tie on both ruin steps; identical returns.
        a = cell(equity=0.5, ruin=0)
        b = cell(equity=0.6, ruin=50)
        assert self.ranked(b, a)[0].cell is a
        assert self.ranked(a, b)[0].cell is a

    def test_p10_then_p50(self):
        a = cell(equity=0.5, ruin=20, p10=1.0, p50=3.0)
        b = cell(equity=0.6, ruin=20, p10=1.1, p50=5.0)
        # Wealth multiples 2.0 vs 2.1: within 10%, so P50 decides.
        top = self.ranked(a, b)[0]
        assert top.cell is b
        assert top.reason == ("tied on lifetime ruin, years in ruin, P10 "
                              "return; won on P50 return")
        assert top.decided_by == "P50 return"

        c = cell(equity=0.7, ruin=20, p10=2.0, p50=1.0)
        top = self.ranked(a, b, c)[0]
        assert top.cell is c
        assert top.reason == (
            "tied on lifetime ruin, years in ruin; won on P10 return")

    def test_real_returns_when_every_cell_has_them(self):
        # Nominal says a, real says b.
        a = dataclasses.replace(cell(equity=0.5, ruin=20, p10=3.0),
                                p10_real_return=0.1, p50_real_return=0.5)
        b = dataclasses.replace(cell(equity=0.6, ruin=20, p10=1.0),
                                p10_real_return=0.6, p50_real_return=0.5)
        assert self.ranked(a, b)[0].cell is b
        # One cell without real returns: everything compares nominal.
        old = cell(equity=0.7, ruin=20, p10=0.0)
        assert self.ranked(a, b, old)[0].cell is a

    def test_p10_tolerance_works_near_zero_return(self):
        # Returns of -0.02 and +0.02 are 4% apart in wealth terms: a tie.
        a = cell(equity=0.5, ruin=20, p10=-0.02, p50=9.0)
        b = cell(equity=0.6, ruin=20, p10=0.02, p50=1.0)
        assert self.ranked(a, b)[0].cell is a

    def test_tolerances_are_anchored_to_the_best(self):
        # a~b and b~c on lifetime ruin (within 1pp) but a and c are 1.6pp
        # apart: c is dropped at step 0 even though, ruined at 74, it has
        # the fewest years in ruin. a and b tie on years in ruin (within
        # 0.1 years over a 10-year horizon), so P10 decides.
        crit = d.Criteria(retirement_age=65)
        a = cell(equity=0.4, ruin=100)
        b = cell(equity=0.5, ruin=180, p10=1.5)
        c = cell(equity=0.6, histogram=[0] * 119 + [270])
        assert crit.years_in_ruin(c) < crit.years_in_ruin(a)
        assert crit.ruin(c) - crit.ruin(a) > crit.ruin_tolerance
        order = [r.cell for r in d.rank([a, b, c], crit)]
        assert order == [b, a, c]

    def test_failing_cells_last_by_lifetime_ruin(self):
        good = cell(equity=0.5, ruin=100)
        bad = cell(equity=0.6, ruin=900)
        worse = cell(equity=0.7, ruin=1500)
        ranked = self.ranked(worse, bad, good)
        assert [r.cell for r in ranked] == [good, bad, worse]
        assert [r.passes for r in ranked] == [True, False, False]
        assert [r.rank for r in ranked] == [1, 2, 3]
        assert [r.decided_by for r in ranked] == ["—", "ceiling", "ceiling"]


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


class TestRuinByAge:
    @pytest.mark.parametrize("retirement_age", [60, 60.5, 42.25])
    def test_matches_the_cli(self, retirement_age):
        # Same convention as monte_carlo.ruin_probability_by_age, which
        # writes ruin_prob_by_age into newer results.
        import numpy as np
        import monte_carlo
        rng = np.random.default_rng(1)
        histogram = rng.integers(0, 3, 240).astype(float)
        c = cell(histogram=list(histogram), paths=1_000)
        ages = [retirement_age - 1, retirement_age, retirement_age + 1 / 12,
                retirement_age + 7.5, 75, 85, 200]
        expected = monte_carlo.ruin_probability_by_age(
            histogram, 1_000, retirement_age, ages)
        for a in ages:
            assert d.ruin_prob_before(c, a, retirement_age) == (
                pytest.approx(expected[f"{a:g}"])), a

    def test_agrees_with_survival(self):
        c = cell(histogram=[0, 2, 0, 5, 1, 0], paths=20)
        survival = c.survival()
        for n in range(1, 7):
            assert d.ruin_prob_before(c, 60 + n / 12, 60) == (
                pytest.approx(1 - survival[n - 1]))

    def test_ages_inside_the_horizon(self):
        assert d.ruin_ages(42, 105) == [75, 85, 95]
        assert d.ruin_ages(68, 78) == [75]
        assert d.ruin_ages(75, 90) == [85]  # retirement age itself is out
        assert d.ruin_ages(60, 70) == []


class TestDollars:
    def test_real_or_nominal(self):
        c = dataclasses.replace(cell(p10=1.0), p10_real_return=0.4,
                                real_nav_bands={"p50": [2]},
                                nav_bands={"p50": [3]})
        assert (c.pct_return(10, False), c.pct_return(10, True)) == (1.0,
                                                                     0.4)
        assert c.bands(True) == {"p50": [2]}
        assert c.bands(False) == {"p50": [3]}

    def test_missing_real_is_none(self):
        c = cell()
        assert c.pct_return(50, True) is None
        assert c.bands(True) is None


IMMORTAL = d.Household(person=d.Life(modal_age=1_000.0, dispersion=10.0))


class TestLife:
    def test_default_fits_the_cdc_male_table(self):
        # Fitted to the CDC 2022 male table with 1%/year mortality
        # improvement, for someone alive at 42 (see Life's docstring).
        life = d.Life()
        assert life.life_expectancy(42) == pytest.approx(81.9, abs=0.3)
        assert life.survival(85, 42) == pytest.approx(0.46, abs=0.02)
        assert life.survival(95, 42) == pytest.approx(0.18, abs=0.02)

    def test_survival_is_a_survival_curve(self):
        life = d.Life()
        assert life.survival(42, 42) == 1.0
        assert life.survival(30, 42) == 1.0
        values = [life.survival(a, 42) for a in range(42, 121)]
        assert values == sorted(values, reverse=True)
        assert values[-1] < 1e-3

    def test_later_modal_age_lives_longer(self):
        assert (d.Life(modal_age=92).life_expectancy(42)
                > d.Life(modal_age=85).life_expectancy(42))


class TestHousehold:
    def test_single_is_the_persons_survival(self):
        h = d.Household()
        assert h.p_alive(85, 42) == h.person.survival(85, 42)

    def test_couple_is_last_survivor(self):
        person, partner = d.Life(85, 10), d.Life(90, 9)
        h = d.Household(person=person, partner=partner,
                        partner_age_offset=-3)
        a = person.survival(85, 60)
        b = partner.survival(82, 57)
        assert h.p_alive(85, 60) == pytest.approx(1 - (1 - a) * (1 - b))
        assert h.p_alive(85, 60) > d.Household(person=person).p_alive(85, 60)


class TestLifetimeRuin:
    def test_immortal_household_gives_the_ruin_rate(self):
        crit = d.Criteria(retirement_age=60, household=IMMORTAL)
        c = cell(histogram=ruined_at(100, 30) , paths=1_000)
        assert crit.ruin(c) == pytest.approx(c.ruin_rate)
        assert crit.ruin_se(c) == pytest.approx(c.ruin_rate_se)

    def test_weights_each_ruin_by_p_alive(self):
        crit = d.Criteria(retirement_age=60)
        histogram = [0] * 480
        histogram[60], histogram[300] = 10, 20   # ruin at 65 and at 85
        c = cell(histogram=histogram, paths=1_000)
        h = crit.household
        expected = (10 * h.p_alive(65, 60) + 20 * h.p_alive(85, 60)) / 1_000
        assert crit.ruin(c) == pytest.approx(expected)

    def test_se_is_that_of_the_weighted_paths(self):
        import numpy as np
        crit = d.Criteria(retirement_age=60)
        histogram = [0] * 480
        histogram[12], histogram[400] = 40, 70
        c = cell(histogram=histogram, paths=500)
        h = crit.household
        per_path = np.array([h.p_alive(61, 60)] * 40
                            + [h.p_alive(60 + 400 / 12, 60)] * 70
                            + [0.0] * 390)
        assert crit.ruin(c) == pytest.approx(per_path.mean())
        assert crit.ruin_se(c) == pytest.approx(
            per_path.std() / np.sqrt(500))

    def test_years_in_ruin_counts_time_left_in_the_horizon(self):
        # Immortal: a path ruined in month m spends the rest of the
        # 120-month horizon ruined.
        crit = d.Criteria(retirement_age=60, household=IMMORTAL)
        histogram = [0] * 120
        histogram[0], histogram[60] = 1, 3
        c = cell(histogram=histogram, paths=10)
        assert crit.years_in_ruin(c) == pytest.approx((10 + 3 * 5) / 10)

    def test_years_in_ruin_weighs_early_ruin_more(self):
        crit = d.Criteria(retirement_age=42)
        early = cell(histogram=ruined_at(12, 100, months=696))
        late = cell(histogram=ruined_at(500, 100, months=696))
        assert crit.years_in_ruin(early) > 5 * crit.years_in_ruin(late)
