import numpy as np
import pandas as pd
import pytest

import market_modelling.path_simulation as ps
from monte_carlo import MonteCarloCLI

MODELS = MonteCarloCLI.SUPPORTED_MODELS
MONTHS, PATHS = 24, 50


@pytest.fixture(params=MODELS, ids=[m.name() for m in MODELS])
def fitted(request, aligned_market):
    levels, returns = aligned_market
    simulator = request.param()
    simulator.fit(returns, levels)
    return simulator


class TestPathSimulators:
    def test_unfitted_simulator_refuses_to_simulate(self):
        for model in MODELS:
            with pytest.raises(RuntimeError, match="not fitted"):
                model().simulate_paths(12, 5, seed=1)

    def test_output_shapes(self, fitted):
        paths = fitted.simulate_paths(MONTHS, PATHS, seed=1)
        assert len(paths) == 4
        assert all(p.shape == (PATHS, MONTHS) for p in paths)

    def test_same_seed_gives_identical_paths(self, fitted):
        a = fitted.simulate_paths(MONTHS, PATHS, seed=5)
        b = fitted.simulate_paths(MONTHS, PATHS, seed=5)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_different_seed_gives_different_paths(self, fitted):
        a = fitted.simulate_paths(MONTHS, PATHS, seed=5)
        b = fitted.simulate_paths(MONTHS, PATHS, seed=6)
        assert not np.array_equal(a[0], b[0])

    def test_simulating_does_not_mutate_the_fitted_model(self, fitted):
        first = fitted.simulate_paths(MONTHS, PATHS, seed=9)
        fitted.simulate_paths(MONTHS, PATHS, seed=10)
        again = fitted.simulate_paths(MONTHS, PATHS, seed=9)
        for x, y in zip(first, again):
            np.testing.assert_array_equal(x, y)

    def test_paths_are_finite_and_economically_sane(self, fitted):
        spx, cpi, y3m, y5y = fitted.simulate_paths(MONTHS, PATHS, seed=2)
        for series in (spx, cpi, y3m, y5y):
            assert np.isfinite(series).all()
        assert (spx > 0).all()
        assert (cpi > 0).all()
        assert (y3m >= 0).all()
        assert (y5y >= 0).all()
        assert y3m.max() < 0.5 and y5y.max() < 0.5

    def test_paths_start_near_the_last_observed_levels(
            self, fitted, aligned_market):
        levels, _ = aligned_market
        spx, cpi, y3m, y5y = fitted.simulate_paths(MONTHS, PATHS, seed=3)
        last = levels.iloc[-1]
        # One monthly step away from the last fitted level, never a restart.
        assert np.abs(np.log(spx[:, 0] / last["spx_close"])).max() < 0.5
        assert np.abs(np.log(cpi[:, 0] / last["cpi"])).max() < 0.1
        assert np.abs(y3m[:, 0] - last["yield_3m"]).max() < 0.03
        assert np.abs(y5y[:, 0] - last["yield_5y"]).max() < 0.03

    def test_paths_within_a_batch_are_mostly_distinct(self, fitted):
        spx = fitted.simulate_paths(MONTHS, PATHS, seed=4)[0]
        # Bootstrap models can legitimately redraw the same historical block
        # for two paths, so allow a few collisions -- but not a shared path.
        assert len({tuple(row) for row in spx}) >= 0.8 * PATHS


def test_model_names_are_unique_and_stable():
    names = [m.name() for m in MODELS]
    assert len(set(names)) == len(names)
    assert all(m.name() == m.__name__ for m in MODELS)


def test_path_simulator_base_is_abstract():
    with pytest.raises(TypeError):
        ps.PathSimulator()


class TestRegimeSwitching:
    def test_too_few_contraction_months_is_rejected(self, aligned_market):
        levels, returns = aligned_market
        # The 1990s window holds no NBER-dated contraction months.
        with pytest.raises(ValueError, match="Regime 1 has only"):
            ps.RegimeSwitchingBootstrapSimulator().fit(
                returns.loc["1993":"1999"], levels)

    def test_custom_regime_labels_are_respected(self, aligned_market):
        levels, returns = aligned_market
        labels = pd.Series(0, index=returns.index)
        labels.iloc[100:160] = 1
        sim = ps.RegimeSwitchingBootstrapSimulator()
        sim.fit(returns, levels, regime_labels=labels)
        assert len(sim.pool[1]) == 60
        assert len(sim.pool[0]) == len(returns) - 60


class TestLabelRegimes:
    # NBER: peak 2007-12, trough 2009-06. A contraction runs from the month
    # AFTER the peak through the trough month, inclusive.
    index = pd.date_range("2007-10-01", "2009-09-01", freq="MS")

    def label(self, month):
        return int(ps.label_regimes(self.index)[pd.Timestamp(month)])

    def test_peak_month_is_still_expansion(self):
        assert self.label("2007-12-01") == 0

    def test_first_contraction_month(self):
        assert self.label("2008-01-01") == 1

    def test_trough_month_is_contraction(self):
        assert self.label("2009-06-01") == 1

    def test_month_after_trough_is_expansion(self):
        assert self.label("2009-07-01") == 0

    def test_custom_cycles(self):
        labels = ps.label_regimes(
            pd.date_range("2000-01-01", periods=6, freq="MS"),
            nber_cycles=[("2000-02-01", "2000-04-01")])
        assert labels.tolist() == [0, 0, 1, 1, 0, 0]

    def test_no_contractions_in_window(self):
        labels = ps.label_regimes(
            pd.date_range("1995-01-01", periods=12, freq="MS"))
        assert (labels == 0).all()
