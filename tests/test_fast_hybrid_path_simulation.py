"""
Parity tests for market_modelling/fast_hybrid_path_simulation.py against
the reference HybridValuationVARSimulator.simulate_paths. A failure here
must always mean a porting mistake, never an intentional behavior change
-- see doc/plans/GPU Optimization Plan.md.
"""
import numpy as np
import pytest

from market_modelling.path_simulation import HybridValuationVARSimulator
from market_modelling.fast_hybrid_path_simulation import (
    simulate_hybrid_paths_fast)


@pytest.fixture
def fitted(aligned_market):
    levels, returns = aligned_market
    simulator = HybridValuationVARSimulator()
    simulator.fit(returns, levels)
    return simulator


class TestFastHybridPathSimulationParity:
    @pytest.mark.parametrize("months,paths,seed", [
        (24, 50, 1),
        (60, 30, 5),
        (756, 20, 42),   # the large-run horizon
        (13, 7, 0),      # not a multiple of the block size
    ])
    def test_matches_reference_bit_for_bit(self, fitted, months, paths, seed):
        ref = fitted.simulate_paths(months, paths, seed=seed)
        fast = simulate_hybrid_paths_fast(fitted, months, paths, seed=seed)

        for name, r, f in zip(
                ["spx", "cpi", "yield3m", "yield5y"], ref, fast):
            np.testing.assert_allclose(
                f, r, rtol=1e-9, atol=1e-9,
                err_msg=f"{name} mismatch at months={months} paths={paths} "
                        f"seed={seed}")

    def test_different_seeds_differ(self, fitted):
        a = simulate_hybrid_paths_fast(fitted, 24, 20, seed=1)
        b = simulate_hybrid_paths_fast(fitted, 24, 20, seed=2)
        assert not np.allclose(a[0], b[0])

    def test_same_seed_reproducible(self, fitted):
        a = simulate_hybrid_paths_fast(fitted, 24, 20, seed=9)
        b = simulate_hybrid_paths_fast(fitted, 24, 20, seed=9)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_unfitted_simulator_refuses_to_simulate(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            simulate_hybrid_paths_fast(
                HybridValuationVARSimulator(), 12, 5, seed=1)

    def test_lag_order_2_matches_reference(self, aligned_market):
        """lag_order != 1 is the one structural parameter the port treats
        generically (variable design-row width); worth checking directly
        rather than assuming the default-only tests exercise it."""
        levels, returns = aligned_market
        simulator = HybridValuationVARSimulator(lag_order=2)
        simulator.fit(returns, levels)

        ref = simulator.simulate_paths(36, 15, seed=3)
        fast = simulate_hybrid_paths_fast(simulator, 36, 15, seed=3)
        for r, f in zip(ref, fast):
            np.testing.assert_allclose(f, r, rtol=1e-9, atol=1e-9)
