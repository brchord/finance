import warnings

import numpy as np
import pytest

from portfolio_models.linear_models import LongSPYWithTreasuryLadders as LSTL
from tax_models.regimes import BASELINE_2026_SINGLE, build_tax_regime


def run(equity, ladder, spending, *, months=60, nav=1_000_000.0, spx=None,
        y3=0.0, y5=0.0, div=0.0, regime=None, full_book=False):
    """Runs one path. Defaults: flat SPX (SPY = $100), flat CPI, zero yields."""
    strategy = LSTL(equity, ladder, spending, div,
                    tax_regime=build_tax_regime(regime))
    path = strategy.run_simulation(
        spx=np.full(months, 1000.0) if spx is None else spx,
        cpi=np.full(months, 100.0),
        yield3m=np.full(months, y3),
        yield5y=np.full(months, y5),
        initial_nav=nav,
        months=months,
        full_book=full_book,
    )
    return strategy, path


def year_income(strategy, year):
    events = [(o, p) for m, o, p in strategy.income_events
              if year * 12 <= m < (year + 1) * 12]
    return sum(o for o, _ in events), sum(p for _, p in events)


class TestConstruction:
    def test_allocations_must_sum_to_one(self):
        with pytest.raises(ValueError, match="100%"):
            LSTL(0.5, 0.4, 1000.0)

    @pytest.mark.parametrize("spending", [0.0, -1000.0])
    def test_spending_must_be_positive(self, spending):
        with pytest.raises(ValueError, match="non-positive"):
            LSTL(0.6, 0.4, spending)

    def test_from_json_object(self):
        s = LSTL.from_json_object({
            "type": "LongSPYWithTreasuryLadders",
            "equity_allocation": 0.6, "ladder_allocation": 0.4,
            "yearly_spending": 40_000, "dividend_yield": 0.02,
            "tax_regime": "current_law_indexed"})
        assert s.equity_allocation == 0.6
        assert s.spy_div_yield == 0.02
        assert s.tax_regime.name == "current_law_indexed"

    def test_from_json_object_defaults_to_untaxed(self):
        s = LSTL.from_json_object({
            "type": "LongSPYWithTreasuryLadders",
            "equity_allocation": 0.6, "ladder_allocation": 0.4,
            "yearly_spending": 40_000})
        assert s.tax_regime is None
        assert s.spy_div_yield == 0.01

    def test_from_json_object_wrong_type(self):
        assert LSTL.from_json_object({"type": "Other"}) is None


class TestNeededLiquidity:
    def test_no_notes_reserves_twelve_months(self):
        assert LSTL._get_needed_liquidity(100.0, 5, {}) == 1200.0

    def test_reserves_until_nearest_maturity(self):
        notes = {0: (24, 1.0, 0.0), 1: (36, 1.0, 0.0)}
        assert LSTL._get_needed_liquidity(100.0, 20, notes) == 400.0

    def test_at_least_one_month(self):
        assert LSTL._get_needed_liquidity(
            100.0, 24, {0: (24, 1.0, 0.0)}) == 100.0


class TestConservation:
    """With no yields, no growth and no taxes, NAV only falls by spending."""

    def test_ladder_only_nav_falls_by_spending(self):
        _, path = run(0.0, 1.0, 12_000)
        np.testing.assert_allclose(
            path, 1_000_000 - 1_000 * np.arange(1, 61))

    def test_equity_only_nav_falls_by_spending(self):
        strategy, path = run(1.0, 0.0, 12_000)
        np.testing.assert_allclose(
            path, 1_000_000 - 1_000 * np.arange(1, 61))
        # Selling at the purchase price realizes no gain.
        assert all(st == 0 and lt == 0
                   for _, st, lt in strategy.realized_gains)

    def test_mixed_allocation_nav_falls_by_spending(self):
        _, path = run(0.6, 0.4, 12_000)
        np.testing.assert_allclose(
            path, 1_000_000 - 1_000 * np.arange(1, 61), atol=1e-6)

    def test_flat_market_with_interest_grows(self):
        _, path = run(0.0, 1.0, 12, y3=0.04, y5=0.04, months=24)
        assert path[-1] > 1_000_000


class TestRuin:
    def test_ruined_path_is_zero_filled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # e.g. 0/0 RuntimeWarning
            _, path = run(1.0, 0.0, 120_000, nav=100_000.0, months=24)
        # 10k/month withdrawals against 100k NAV: gone after month 10.
        assert path[8] == pytest.approx(10_000)
        assert path[9] == 0.0
        assert (path[9:] == 0.0).all()
        assert (path[:9] > 0).all()


class TestEquity:
    def test_dividends_are_quarterly_preferential_income(self):
        strategy, _ = run(1.0, 0.0, 12_000, div=0.04, months=24)
        div_events = [(m, o, p) for m, o, p in strategy.income_events
                      if p > 0]
        assert [m for m, _, _ in div_events] == [3, 6, 9, 12, 15, 18, 21]
        assert all(o == 0.0 for _, o, _ in div_events)

    def test_rising_market_realizes_long_term_gains(self):
        spx = 1000.0 * 1.01 ** np.arange(120)
        strategy, path = run(1.0, 0.0, 12_000, spx=spx, months=120)
        annual = strategy.annual_realized_gains()
        assert sum(lt for _, lt in annual.values()) > 0
        # Rolled-up totals match the per-sale ledger and the lot tracker.
        assert sum(st + lt for st, lt in annual.values()) == pytest.approx(
            sum(st + lt for _, st, lt in strategy.realized_gains))
        assert sum(lt for _, lt in annual.values()) == pytest.approx(
            strategy.spy_lots.realized_long_term_gain)
        assert (path > 0).all()

    def test_annual_realized_gains_keys_by_year(self):
        strategy = LSTL(0.5, 0.5, 1000.0)
        strategy.realized_gains = [(0, 1.0, 2.0), (11, 3.0, 4.0),
                                   (12, 5.0, 6.0)]
        assert strategy.annual_realized_gains() == {
            0: (4.0, 6.0), 1: (5.0, 6.0)}


class TestTaxes:
    def test_untaxed_records_no_payments(self):
        strategy, _ = run(0.0, 1.0, 12, y3=0.04, y5=0.04, nav=10_000_000.0,
                          months=36)
        assert strategy.tax_paid == []

    def test_income_below_standard_deduction_owes_nothing(self):
        strategy, _ = run(0.0, 1.0, 12, y3=0.015, y5=0.015, months=36,
                          regime="current_law_indexed")
        assert year_income(strategy, 0)[0] < 16_100
        assert strategy.tax_paid == []

    def test_tax_paid_matches_tax_law_on_recorded_income(self):
        strategy, _ = run(0.0, 1.0, 12, y3=0.04, y5=0.04, nav=10_000_000.0,
                          months=36, regime="current_law_indexed")
        # Flat CPI, so the resolved law is the untouched 2026 baseline.
        assert [m for m, _ in strategy.tax_paid] == [12, 24, 36]
        for year, (_, paid) in enumerate(strategy.tax_paid):
            expected = BASELINE_2026_SINGLE.compute_tax(
                *year_income(strategy, year))
            assert paid == pytest.approx(expected)

    def test_taxes_reduce_final_nav_by_at_least_the_tax_paid(self):
        kwargs = dict(y3=0.04, y5=0.04, nav=10_000_000.0, months=36)
        taxed, taxed_path = run(0.0, 1.0, 12, regime="current_law_indexed",
                                **kwargs)
        _, untaxed_path = run(0.0, 1.0, 12, **kwargs)
        paid = sum(p for _, p in taxed.tax_paid)
        assert untaxed_path[-1] - taxed_path[-1] >= paid > 0

    def test_equity_heavy_path_pays_tax_on_a_yearly_cadence(self):
        spx = 1000.0 * 1.01 ** np.arange(120)
        strategy, path = run(1.0, 0.0, 60_000, spx=spx, nav=2_000_000.0,
                             months=120, div=0.02,
                             regime="current_law_indexed")
        assert strategy.tax_paid
        assert all(m % 12 == 0 and p > 0 for m, p in strategy.tax_paid)
        assert (path > 0).all()


class TestBookkeeping:
    def test_full_book_does_not_change_results(self):
        kwargs = dict(y3=0.04, y5=0.04, nav=10_000_000.0, months=36,
                      regime="current_law_indexed")
        _, plain = run(0.0, 1.0, 12, **kwargs)
        strategy, booked = run(0.0, 1.0, 12, full_book=True, **kwargs)
        np.testing.assert_array_equal(plain, booked)
        assert strategy.transaction_book()
        assert strategy.book[0]["trade"] == "buy"

    def test_book_empty_without_full_book(self):
        strategy, _ = run(0.6, 0.4, 12_000, months=12)
        assert strategy.transaction_book() == []

    def test_run_simulation_resets_state_between_paths(self):
        strategy = LSTL(0.6, 0.4, 12_000.0, 0.01,
                        tax_regime=build_tax_regime("current_law_indexed"))
        kwargs = dict(spx=np.full(36, 1000.0), cpi=np.full(36, 100.0),
                      yield3m=np.full(36, 0.04), yield5y=np.full(36, 0.04),
                      initial_nav=10_000_000.0, months=36)
        first = strategy.run_simulation(**kwargs)
        n_events = len(strategy.income_events)
        second = strategy.run_simulation(**kwargs)
        np.testing.assert_array_equal(first, second)
        assert len(strategy.income_events) == n_events
