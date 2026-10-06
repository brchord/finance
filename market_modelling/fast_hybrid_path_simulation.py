"""
fast_hybrid_path_simulation.py

Numba port of HybridValuationVARSimulator.simulate_paths's per-timestep
recurrence (market_modelling/path_simulation.py). That recurrence is
sequential over the 756-step time axis (each step depends on the previous
one's state), so it is NOT fixable by the same across-paths vectorization
used for the bootstrap block assembly -- see path_simulation.py's own
comment on the fancy-indexed gather: both models are sequential in time,
and parallelism exists only across paths. Profiling showed this loop, not the bootstrap
assembly, is HybridValuationVARSimulator's actual dominant path-generation
cost: per-step Python/numpy dispatch overhead across 756 iterations, not
the arithmetic itself.

This module does not replace or modify HybridValuationVARSimulator, which
remains the reference implementation -- see
tests/test_fast_hybrid_path_simulation.py for the parity tests that pin
this port to it.

The RNG draws and bootstrap block assembly (random_block_starts,
bootstrapped_residuals) are duplicated here rather than shared with the
reference, matching the precedent of portfolio_models/fast_ladder.py
of keeping the reference class untouched; the duplication is pinned by
parity tests that compare the final path arrays, not just the
intermediate residuals.

Hardcodes num_variables == 4 (spx log-return, cpi log-return, 3M yield
diff, 5Y yield diff), matching HybridValuationVARSimulator's fixed VAR(p)
variable set -- the predicted-increment overwrite at indices 0/2/3 is
specific to this model, not a general num_variables-sized loop.
"""

import numpy as np
from numba import njit, prange

from market_modelling.path_simulation import HybridValuationVARSimulator

N_VARIABLES = 4


@njit(parallel=True, cache=True)
def _simulate_hybrid_core(
    bootstrapped_residuals,   # (num_paths, months, 4) float64
    historical_seed_matrix,   # (p, 4) float64
    coefficient_matrix,       # (1 + p*4, 4) float64
    historical_spx_mean,      # float64 scalar
    initial_spx_level, initial_cpi_level,
    initial_yield_3m, initial_yield_5y,
    initial_cape, target_cape, phi_cape, gamma_cape,
    equilibrium_equity_drift, phi_rate,
    target_yield_3m, target_yield_5y,
    p, months,
):
    """
    Numba port of the per-step loop in HybridValuationVARSimulator.
    simulate_paths (path_simulation.py, lines ~749-799). Comments reference
    the original loop's structure so a diff against the reference is easy
    to audit.
    """
    num_paths = bootstrapped_residuals.shape[0]
    dt = 1.0 / 12.0
    design_width = 1 + p * N_VARIABLES
    log_target_cape = np.log(target_cape)
    log_initial_cape = np.log(initial_cape)

    spx_paths = np.zeros((num_paths, months))
    cpi_paths = np.zeros((num_paths, months))
    yield_3m_paths = np.zeros((num_paths, months))
    yield_5y_paths = np.zeros((num_paths, months))

    # prange is the OUTER loop here (over paths, each independent within a
    # step), not the step loop -- paths are fully sequential over time, so
    # each thread runs one path through all `months` steps start to finish.
    # This launches one parallel region total instead of one per step,
    # which matters: launching/joining the thread pool `months` times (756
    # at the large-run horizon) for a few dozen FLOPs of work per path per
    # step was pure overhead, and measured ~2x slower than this layout.
    for i in prange(num_paths):
        curr_spx = initial_spx_level
        curr_cpi = initial_cpi_level
        curr_3m = initial_yield_3m
        curr_5y = initial_yield_5y
        log_cape = log_initial_cape

        # This path's history window, seeded from historical_seed_matrix --
        # same as the reference's
        # np.tile(self.historical_seed_matrix, (num_paths, 1, 1))[i].
        hist = np.empty((p, N_VARIABLES))
        for j in range(p):
            for k in range(N_VARIABLES):
                hist[j, k] = historical_seed_matrix[j, k]

        design = np.empty(design_width)

        for step in range(months):
            # Design row: [1, history[p-1], history[p-2], ..., history[0]]
            # -- same order as the reference's
            # [ones] + [path_histories[:, p-lag, :] for lag in 1..p].
            design[0] = 1.0
            idx = 1
            for lag in range(1, p + 1):
                src_j = p - lag
                for k in range(N_VARIABLES):
                    design[idx] = hist[src_j, k]
                    idx += 1

            raw0 = 0.0
            raw1 = 0.0
            raw2 = 0.0
            raw3 = 0.0
            for d in range(design_width):
                dv = design[d]
                raw0 += dv * coefficient_matrix[d, 0]
                raw1 += dv * coefficient_matrix[d, 1]
                raw2 += dv * coefficient_matrix[d, 2]
                raw3 += dv * coefficient_matrix[d, 3]
            raw0 += bootstrapped_residuals[i, step, 0]
            raw1 += bootstrapped_residuals[i, step, 1]
            raw2 += bootstrapped_residuals[i, step, 2]
            raw3 += bootstrapped_residuals[i, step, 3]

            spx_shock = raw0 - historical_spx_mean
            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -gamma_cape * valuation_gap * dt

            spx_log_ret = (
                equilibrium_equity_drift + spx_shock + valuation_penalty)
            cpi_log_ret = raw1
            diff_3m = raw2 - phi_rate * (curr_3m - target_yield_3m) * dt
            diff_5y = raw3 - phi_rate * (curr_5y - target_yield_5y) * dt

            curr_spx *= np.exp(spx_log_ret)
            curr_cpi *= np.exp(cpi_log_ret)
            curr_3m = max(0.0, curr_3m + diff_3m)
            curr_5y = max(0.0, curr_5y + diff_5y)

            log_cape += (
                spx_shock - (phi_cape + gamma_cape) * valuation_gap * dt)

            spx_paths[i, step] = curr_spx
            cpi_paths[i, step] = curr_cpi
            yield_3m_paths[i, step] = curr_3m
            yield_5y_paths[i, step] = curr_5y

            # Shift history left by one step, append this step's predicted
            # increments (not the raw ones -- same substitution the
            # reference makes at indices 0/2/3 before pushing into history).
            for j in range(p - 1):
                for k in range(N_VARIABLES):
                    hist[j, k] = hist[j + 1, k]
            hist[p - 1, 0] = spx_log_ret
            hist[p - 1, 1] = cpi_log_ret
            hist[p - 1, 2] = diff_3m
            hist[p - 1, 3] = diff_5y

    return spx_paths, cpi_paths, yield_3m_paths, yield_5y_paths


def simulate_hybrid_paths_fast(
    simulator: HybridValuationVARSimulator,
    simulation_months: int = 360,
    num_paths: int = 10000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated equivalent of
    HybridValuationVARSimulator.simulate_paths. Identical RNG draws
    (same rng.integers call, same shape/order), identical bootstrap block
    assembly, and the jitted per-step recurrence above in place of the
    reference's Python loop. See tests/test_fast_hybrid_path_simulation.py
    for the parity tests pinning this to the reference.
    """
    if (simulator.coefficient_matrix is None
            or simulator.residual_matrix is None):
        raise RuntimeError("Model is not fitted. Call fit() first.")
    assert simulator.residual_matrix.shape[1] == N_VARIABLES, (
        "fast_hybrid_path_simulation hardcodes num_variables == 4")
    assert simulator.historical_seed_matrix is not None
    assert simulator.historical_mean_returns is not None
    assert simulator.expected_inflation is not None
    assert simulator.initial_spx_level is not None
    assert simulator.initial_cpi_level is not None
    assert simulator.initial_yield_3m is not None
    assert simulator.initial_yield_5y is not None
    assert simulator.target_yield_3m is not None
    assert simulator.target_yield_5y is not None

    rng = np.random.default_rng(seed)
    effective_sample_size, num_variables = simulator.residual_matrix.shape
    p = simulator.lag_order
    block_size = simulator.residual_block_size

    num_blocks = int(np.ceil(simulation_months / block_size))
    max_start_index = effective_sample_size - block_size
    random_block_starts = rng.integers(
        0, max_start_index + 1, size=(num_paths, num_blocks))

    block_offsets = np.arange(block_size)
    block_row_idx = (
        random_block_starts[:, :, None] + block_offsets[None, None, :])
    bootstrapped_residuals = simulator.residual_matrix[block_row_idx]
    bootstrapped_residuals = bootstrapped_residuals.reshape(
        num_paths, num_blocks * block_size, num_variables)
    bootstrapped_residuals = np.ascontiguousarray(
        bootstrapped_residuals[:, :simulation_months, :])

    dt = 1.0 / 12.0
    equilibrium_equity_drift = (
        simulator.earnings_growth + simulator.buyback_yield +
        simulator.expected_inflation) * dt
    historical_spx_mean = float(simulator.historical_mean_returns[0])

    return _simulate_hybrid_core(
        bootstrapped_residuals,
        np.ascontiguousarray(simulator.historical_seed_matrix),
        np.ascontiguousarray(simulator.coefficient_matrix),
        historical_spx_mean,
        simulator.initial_spx_level, simulator.initial_cpi_level,
        simulator.initial_yield_3m, simulator.initial_yield_5y,
        simulator.initial_cape, simulator.target_cape, simulator.phi_cape,
        simulator.gamma_cape, equilibrium_equity_drift, simulator.phi_rate,
        simulator.target_yield_3m, simulator.target_yield_5y,
        p, simulation_months,
    )
