"""
Parity tests for market_modelling/fast_regime_switching_path_simulation.py
against the reference RegimeSwitchingValuationVARSimulator.simulate_paths
and RegimeSwitchingBootstrapSimulator.simulate_paths. A failure here must
always mean a porting mistake, never an intentional behavior change.
"""
import numpy as np
import pytest

from market_modelling.path_simulation import (
    RegimeSwitchingBootstrapSimulator, RegimeSwitchingValuationVARSimulator)
from market_modelling.fast_regime_switching_path_simulation import (
    simulate_regime_switching_bootstrap_paths_fast,
    simulate_regime_switching_paths_fast)


@pytest.fixture
def fitted(aligned_market):
    levels, returns = aligned_market
    simulator = RegimeSwitchingValuationVARSimulator()
    simulator.fit(returns, levels)
    return simulator


class TestFastRegimeSwitchingPathSimulationParity:
    @pytest.mark.parametrize("months,paths,seed", [
        (24, 50, 1),
        (60, 30, 5),
        (756, 20, 42),   # the large-run horizon
        (13, 7, 0),      # not a multiple of either regime's block size
        (7, 100, 11),    # shorter than the contraction block size (6)
    ])
    def test_matches_reference_bit_for_bit(self, fitted, months, paths, seed):
        ref = fitted.simulate_paths(months, paths, seed=seed)
        fast = simulate_regime_switching_paths_fast(
            fitted, months, paths, seed=seed)

        for name, r, f in zip(
                ["spx", "cpi", "yield3m", "yield5y"], ref, fast):
            np.testing.assert_array_equal(
                f, r,
                err_msg=f"{name} mismatch at months={months} paths={paths} "
                        f"seed={seed}")

    def test_different_seeds_differ(self, fitted):
        a = simulate_regime_switching_paths_fast(fitted, 24, 20, seed=1)
        b = simulate_regime_switching_paths_fast(fitted, 24, 20, seed=2)
        assert not np.allclose(a[0], b[0])

    def test_same_seed_reproducible(self, fitted):
        a = simulate_regime_switching_paths_fast(fitted, 24, 20, seed=9)
        b = simulate_regime_switching_paths_fast(fitted, 24, 20, seed=9)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_unfitted_simulator_refuses_to_simulate(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            simulate_regime_switching_paths_fast(
                RegimeSwitchingValuationVARSimulator(), 12, 5, seed=1)

    def test_custom_block_sizes_match_reference(self, aligned_market):
        """Non-default block sizes change how many blocks each run splits
        into (and therefore how many rng.integers draws happen) -- worth
        checking directly rather than only exercising the {0: 48, 1: 6}
        default."""
        levels, returns = aligned_market
        simulator = RegimeSwitchingValuationVARSimulator(
            block_sizes={0: 24, 1: 3})
        simulator.fit(returns, levels)

        ref = simulator.simulate_paths(48, 25, seed=3)
        fast = simulate_regime_switching_paths_fast(simulator, 48, 25, seed=3)
        for r, f in zip(ref, fast):
            np.testing.assert_array_equal(f, r)


@pytest.fixture
def fitted_bootstrap(aligned_market):
    levels, returns = aligned_market
    simulator = RegimeSwitchingBootstrapSimulator()
    simulator.fit(returns, levels)
    return simulator


class TestFastRegimeSwitchingBootstrapPathSimulationParity:
    @pytest.mark.parametrize("months,paths,seed", [
        (24, 50, 1),
        (60, 30, 5),
        (756, 20, 42),   # the large-run horizon
        (13, 7, 0),      # not a multiple of either regime's block size
        (7, 100, 11),    # shorter than the contraction block size (6)
        (24, 1, 3),      # a single path, as in 1-path chunks
    ])
    def test_matches_reference_bit_for_bit(
            self, fitted_bootstrap, months, paths, seed):
        ref = fitted_bootstrap.simulate_paths(months, paths, seed=seed)
        fast = simulate_regime_switching_bootstrap_paths_fast(
            fitted_bootstrap, months, paths, seed=seed)

        for name, r, f in zip(
                ["spx", "cpi", "yield3m", "yield5y"], ref, fast):
            np.testing.assert_array_equal(
                f, r,
                err_msg=f"{name} mismatch at months={months} paths={paths} "
                        f"seed={seed}")

    def test_same_seed_reproducible_and_seeds_differ(self, fitted_bootstrap):
        a = simulate_regime_switching_bootstrap_paths_fast(
            fitted_bootstrap, 24, 20, seed=9)
        b = simulate_regime_switching_bootstrap_paths_fast(
            fitted_bootstrap, 24, 20, seed=9)
        c = simulate_regime_switching_bootstrap_paths_fast(
            fitted_bootstrap, 24, 20, seed=10)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        assert not np.allclose(a[0], c[0])

    def test_unfitted_simulator_refuses_to_simulate(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            simulate_regime_switching_bootstrap_paths_fast(
                RegimeSwitchingBootstrapSimulator(), 12, 5, seed=1)

    def test_custom_block_sizes_match_reference(self, aligned_market):
        levels, returns = aligned_market
        simulator = RegimeSwitchingBootstrapSimulator(
            block_sizes={0: 24, 1: 3})
        simulator.fit(returns, levels)

        ref = simulator.simulate_paths(48, 25, seed=3)
        fast = simulate_regime_switching_bootstrap_paths_fast(
            simulator, 48, 25, seed=3)
        for r, f in zip(ref, fast):
            np.testing.assert_array_equal(f, r)
