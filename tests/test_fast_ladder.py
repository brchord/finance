"""
Parity tests for portfolio_models/fast_ladder.py against the reference
LongSPYWithTreasuryLadders.run_simulation. A failure here must always mean
a porting mistake in fast_ladder.py, never an intentional behavior change
-- see doc/plans/GPU Optimization Plan.md, Stage 1.
"""
import numpy as np
import pytest

import market_modelling.path_simulation as ps
from portfolio_models.fast_ladder import (
    flatten_tax_regime, run_simulation_fast, run_simulation_fast_batch)
from portfolio_models.linear_models import LongSPYWithTreasuryLadders as LSTL
from tax_models.regimes import build_tax_regime

MODELS = [ps.HybridValuationVARSimulator, ps.RegimeSwitchingValuationVARSimulator]
TAX_REGIMES = ["none", "current_law_indexed", "historical_average_drift",
               "pre_tcja_reversion"]
MONTHS = 240  # 20 years: long enough to exercise every ladder maturity and
              # several tax years, short enough to keep the test suite fast.
PATHS = 40


@pytest.fixture(params=MODELS, ids=[m.name() for m in MODELS])
def fitted(request, aligned_market):
    levels, returns = aligned_market
    simulator = request.param()
    simulator.fit(returns, levels)
    return simulator


def run_reference(spx, cpi, yield3m, yield5y, initial_nav, months,
                   equity, ladder, spending, div, tax_regime_name):
    strategy = LSTL(equity, ladder, spending, div,
                    tax_regime=build_tax_regime(tax_regime_name))
    nav_path = strategy.run_simulation(
        spx=spx, cpi=cpi, yield3m=yield3m, yield5y=yield5y,
        initial_nav=initial_nav, months=months, full_book=False,
    )
    ruin_month = -1
    if nav_path[-1] == 0.0:
        ruin_month = int(np.argmax(nav_path == 0.0))
    return nav_path, ruin_month


def run_fast(spx, cpi, yield3m, yield5y, initial_nav, months,
             equity, ladder, spending, div, tax_regime_name):
    tax_regime = build_tax_regime(tax_regime_name)
    max_years = (months + 11) // 12 + 1
    flat = flatten_tax_regime(tax_regime, max_years)
    nav_path, ruin_month = run_simulation_fast(
        spx, cpi, yield3m, yield5y, initial_nav, months,
        equity, ladder, spending, div, *flat)
    return np.asarray(nav_path), int(ruin_month)


class TestFastLadderParity:
    @pytest.mark.parametrize("tax_regime_name", TAX_REGIMES)
    def test_matches_reference_across_many_paths(
            self, fitted, tax_regime_name):
        spx_paths, cpi_paths, tbill_paths, tnote_paths = fitted.simulate_paths(
            MONTHS, PATHS, seed=42)

        for equity, ladder, spending, nav in [
            (0.6, 0.4, 60_000.0, 1_000_000.0),
            (0.8, 0.2, 90_000.0, 1_500_000.0),
            (0.3, 0.7, 40_000.0, 500_000.0),
        ]:
            for i in range(PATHS):
                ref_path, ref_ruin = run_reference(
                    spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                    nav, MONTHS, equity, ladder, spending, 0.01,
                    tax_regime_name)
                fast_path, fast_ruin = run_fast(
                    spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                    nav, MONTHS, equity, ladder, spending, 0.01,
                    tax_regime_name)

                assert fast_ruin == ref_ruin, (
                    f"ruin month mismatch path={i} regime={tax_regime_name} "
                    f"equity={equity}: ref={ref_ruin} fast={fast_ruin}")
                np.testing.assert_allclose(
                    fast_path, ref_path, rtol=1e-9, atol=1e-6,
                    err_msg=(f"nav path mismatch path={i} "
                             f"regime={tax_regime_name} equity={equity}"))

    def test_terminal_nav_matches_reference(self, fitted):
        """Same as above but isolates just the terminal-NAV comparison the
        MC engine actually consumes (monte_carlo.py's
        _execute_strategy_batch), at a longer horizon matching the
        large-run shape."""
        months = 756
        spx_paths, cpi_paths, tbill_paths, tnote_paths = fitted.simulate_paths(
            months, 20, seed=7)

        for i in range(20):
            ref_path, ref_ruin = run_reference(
                spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                1_000_000.0, months, 0.6, 0.4, 60_000.0, 0.01,
                "current_law_indexed")
            fast_path, fast_ruin = run_fast(
                spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                1_000_000.0, months, 0.6, 0.4, 60_000.0, 0.01,
                "current_law_indexed")

            assert fast_ruin == ref_ruin
            assert fast_path[-1] == pytest.approx(ref_path[-1], rel=1e-9,
                                                   abs=1e-6)


class TestFastLadderBatch:
    def test_batch_matches_single_path_calls(self, fitted):
        months = 120
        spx_paths, cpi_paths, tbill_paths, tnote_paths = fitted.simulate_paths(
            months, 25, seed=3)

        tax_regime = build_tax_regime("current_law_indexed")
        max_years = (months + 11) // 12 + 1
        flat = flatten_tax_regime(tax_regime, max_years)

        batch_navs, batch_ruins = run_simulation_fast_batch(
            spx_paths, cpi_paths, tbill_paths, tnote_paths,
            1_000_000.0, months, 0.6, 0.4, 60_000.0, 0.01, *flat)

        for i in range(25):
            nav_path, ruin_month = run_simulation_fast(
                spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                1_000_000.0, months, 0.6, 0.4, 60_000.0, 0.01, *flat)
            assert batch_ruins[i] == ruin_month
            assert batch_navs[i] == pytest.approx(nav_path[-1])


class TestFlattenTaxRegime:
    @pytest.mark.parametrize("tax_regime_name", TAX_REGIMES)
    def test_recovers_resolve_output(self, tax_regime_name):
        """flatten_tax_regime's arrays, re-scaled, must reproduce whatever
        tax_regime.resolve() itself would compute -- since the whole point
        is to replace calling resolve() at runtime."""
        tax_regime = build_tax_regime(tax_regime_name)
        if tax_regime is None:
            pytest.skip("'none' has no regime to flatten")

        max_years = 60
        (has_tax, ord_rates, ord_floors, ord_deduction,
         ltcg_rates, ltcg_floors, niit_rate, niit_threshold) = (
            flatten_tax_regime(tax_regime, max_years))
        assert has_tax

        for year in [0, 1, 9, 10, 15, 30, 59]:
            for cpi_relative in [1.0, 1.4, 2.7]:
                law = tax_regime.resolve(year, cpi_relative)

                expected_ord_floors = np.array(
                    [f for _, f in law.ordinary.brackets])
                expected_ord_rates = np.array(
                    [r for r, _ in law.ordinary.brackets])
                np.testing.assert_allclose(
                    ord_floors[year] * cpi_relative, expected_ord_floors)
                np.testing.assert_allclose(ord_rates[year], expected_ord_rates)
                assert (ord_deduction[year] * cpi_relative
                        == pytest.approx(law.ordinary.standard_deduction))

                expected_ltcg_floors = np.array(
                    [f for _, f in law.ltcg.brackets])
                expected_ltcg_rates = np.array(
                    [r for r, _ in law.ltcg.brackets])
                np.testing.assert_allclose(
                    ltcg_floors[year] * cpi_relative, expected_ltcg_floors)
                np.testing.assert_allclose(
                    ltcg_rates[year], expected_ltcg_rates)

                assert niit_rate == law.niit.rate
                assert niit_threshold == law.niit.threshold_single
