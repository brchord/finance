"""
The configurable valuation assumptions (initial_cape, annual_buyback_yield)
on the CAPE-drag simulators, and the real-terms / unconditional-ruin
helpers in monte_carlo.py.
"""
import numpy as np
import pytest

import monte_carlo as mc
from market_modelling.fast_hybrid_path_simulation import (
    simulate_hybrid_paths_fast)
from market_modelling.fast_regime_switching_path_simulation import (
    simulate_regime_switching_paths_fast)
from market_modelling.path_simulation import (
    HybridValuationVARSimulator, RegimeSwitchingValuationVARSimulator,
    ValuationAdjustedVARSimulator)

CAPE_MODELS = [HybridValuationVARSimulator,
               RegimeSwitchingValuationVARSimulator,
               ValuationAdjustedVARSimulator]
BUYBACK_MODELS = [HybridValuationVARSimulator,
                  RegimeSwitchingValuationVARSimulator]


def fitted(cls, aligned_market, **kwargs):
    levels, returns = aligned_market
    simulator = cls(**kwargs)
    simulator.fit(returns, levels)
    return simulator


class TestInitialCape:
    @pytest.mark.parametrize("cls", CAPE_MODELS)
    def test_default_is_unchanged(self, cls, aligned_market):
        # 34.0 is what every run effectively used before: fit()'s default
        # always overwrote the constructor's value.
        assert fitted(cls, aligned_market).initial_cape == 34.0

    @pytest.mark.parametrize("cls", CAPE_MODELS)
    def test_constructor_value_survives_fit(self, cls, aligned_market):
        assert fitted(cls, aligned_market,
                      initial_cape=40.0).initial_cape == 40.0

    @pytest.mark.parametrize("cls", CAPE_MODELS)
    def test_fit_argument_still_takes_precedence(self, cls, aligned_market):
        levels, returns = aligned_market
        simulator = cls(initial_cape=40.0)
        simulator.fit(returns, levels, initial_cape=30.0)
        assert simulator.initial_cape == 30.0

    @pytest.mark.parametrize("cls", CAPE_MODELS)
    def test_higher_cape_lowers_equity_paths(self, cls, aligned_market):
        low = fitted(cls, aligned_market, initial_cape=30.0)
        high = fitted(cls, aligned_market, initial_cape=40.0)
        spx_low = low.simulate_paths(120, 50, seed=5)[0]
        spx_high = high.simulate_paths(120, 50, seed=5)[0]
        assert np.median(spx_high[:, -1]) < np.median(spx_low[:, -1])


class TestBuybackYield:
    @pytest.mark.parametrize("cls", BUYBACK_MODELS)
    def test_zero_reproduces_original_paths(self, cls, aligned_market):
        default = fitted(cls, aligned_market)
        explicit = fitted(cls, aligned_market, annual_buyback_yield=0.0)
        for a, b in zip(default.simulate_paths(36, 10, seed=2),
                        explicit.simulate_paths(36, 10, seed=2)):
            np.testing.assert_array_equal(a, b)

    def test_regime_switching_adds_pure_drift(self, aligned_market):
        # Regime-switching shocks don't depend on the equity drift, so a
        # buyback yield scales SPX by exactly exp(b * t) and leaves CPI and
        # yields untouched.
        cls = RegimeSwitchingValuationVARSimulator
        base = fitted(cls, aligned_market).simulate_paths(60, 20, seed=3)
        bought = fitted(cls, aligned_market,
                        annual_buyback_yield=0.012).simulate_paths(
                            60, 20, seed=3)
        expected = np.exp(0.012 / 12.0 * np.arange(1, 61))
        np.testing.assert_allclose(bought[0] / base[0],
                                   np.broadcast_to(expected, base[0].shape),
                                   rtol=1e-12)
        for i in (1, 2, 3):
            np.testing.assert_array_equal(base[i], bought[i])

    def test_hybrid_raises_equity_paths(self, aligned_market):
        # Hybrid feeds the realized equity return back into the VAR
        # history, so the effect isn't a pure multiplier there.
        cls = HybridValuationVARSimulator
        base = fitted(cls, aligned_market).simulate_paths(120, 50, seed=3)
        bought = fitted(cls, aligned_market,
                        annual_buyback_yield=0.015).simulate_paths(
                            120, 50, seed=3)
        assert np.median(bought[0][:, -1]) > np.median(base[0][:, -1])

    @pytest.mark.parametrize("cls,fast,rtol", [
        (HybridValuationVARSimulator, simulate_hybrid_paths_fast, 1e-9),
        # Bit-identical by design; the tolerance only absorbs last-ulp
        # differences between CPUs (see CLAUDE.md, "Known flaky CI").
        (RegimeSwitchingValuationVARSimulator,
         simulate_regime_switching_paths_fast, 1e-12),
    ])
    def test_fast_path_matches_reference(self, cls, fast, rtol,
                                         aligned_market):
        simulator = fitted(cls, aligned_market, initial_cape=40.0,
                           annual_buyback_yield=0.015)
        reference = simulator.simulate_paths(48, 12, seed=9)
        accelerated = fast(simulator, 48, 12, seed=9)
        for ref, acc in zip(reference, accelerated):
            np.testing.assert_allclose(acc, ref, rtol=rtol)


class TestPriceLevels:
    def test_constant_inflation(self):
        months = 30
        cpi = 100.0 * 1.01 ** np.arange(months)
        cpi_paths = np.vstack([cpi, 2.0 * cpi])  # level-free: same ratios
        annual, terminal = mc.price_levels(cpi_paths, months)
        assert annual.shape == (2, 3)  # start + 2 whole years
        np.testing.assert_allclose(annual[:, 0], 1.0)
        np.testing.assert_allclose(annual[:, 1], 1.01 ** 11)
        np.testing.assert_allclose(annual[:, 2], 1.01 ** 23)
        np.testing.assert_allclose(terminal, 1.01 ** 29)

    def test_matches_annual_snapshot_months(self):
        months = 37
        rng = np.random.default_rng(0)
        cpi_paths = 100.0 * np.exp(np.cumsum(
            rng.normal(0.002, 0.001, (4, months)), axis=1))
        annual, terminal = mc.price_levels(cpi_paths, months)
        snaps = mc.fast_ladder.annual_snapshot_months(months)
        np.testing.assert_array_equal(
            annual[:, 1:], cpi_paths[:, snaps] / cpi_paths[:, :1])
        np.testing.assert_array_equal(
            terminal, cpi_paths[:, -1] / cpi_paths[:, 0])


class TestRuinProbabilityByAge:
    # 10 paths, retirement at 60: ruins in months 0, 11, 12 and 299.
    HISTOGRAM = np.zeros(360)
    HISTOGRAM[[0, 11, 12, 299]] = 1

    def probs(self, ages):
        return mc.ruin_probability_by_age(self.HISTOGRAM, 10, 60, ages)

    def test_counts_only_ruins_strictly_before_the_age(self):
        # Age 61 is month 12: months 0 and 11 are before it, 12 is not.
        assert self.probs([61]) == {"61": 0.2}
        assert self.probs([61 + 1 / 12]) == {"61.0833": 0.3}

    def test_is_unconditional_and_cumulative(self):
        probs = self.probs([60, 70, 85, 95])
        assert probs == {"60": 0.0, "70": 0.3, "85": 0.4, "95": 0.4}

    def test_ages_past_the_horizon_count_every_ruin(self):
        assert self.probs([200]) == {"200": 0.4}

    def test_float_noise_in_months_is_ignored(self):
        # (85 - 60) * 12 is exactly 300; 0.1-style float noise must not
        # push the cutoff to 301 and pull in a ruin at month 300.
        histogram = np.zeros(400)
        histogram[300] = 1
        assert mc.ruin_probability_by_age(
            histogram, 1, 60.1, [85.1]) == {"85.1": 0.0}
