"""
fast_regime_switching_path_simulation.py

Numba port of RegimeSwitchingValuationVARSimulator.simulate_paths
(market_modelling/path_simulation.py). Profiling showed its three steps
split roughly 90% / 9% / 1% of total time:

1. Regime path generation (per-step, vectorized-over-paths already, but
   sequential in time): minor cost, ported for completeness.
2. Regime-conditional block bootstrap: by far the dominant cost. The
   reference does this as a per-path nested `while` loop that calls
   `rng.integers(0, max_start + 1)` once per block -- a data-dependent
   number of times per path, since block count depends on how long each
   regime "run" turns out to be for that path's own simulated regime
   sequence.
3. Equilibrium drift + CAPE valuation drag recurrence: same shape as
   HybridValuationVARSimulator's recurrence (see
   fast_hybrid_path_simulation.py), minus the VAR matmul.

Step 2's RNG calls cannot simply move into a Numba kernel: Numba's
nopython mode does not support numpy's Generator API
(`np.random.default_rng`), only the legacy global `np.random` functions,
which use a different underlying bit generator (Mersenne Twister vs.
PCG64) -- "the same seed" would not mean the same draws. Instead, this
port keeps every RNG call in plain numpy, in the same order the reference
would make them, and does everything data-dependent-but-RNG-free (finding
each path's regime "runs" and how they split into blocks) in a Numba
kernel that only counts and labels blocks -- no randomness involved.
The actual `rng.integers()` call for block start positions is then issued
ONCE, batched, with one bound per block (`rng.integers(0, bounds_array)`),
which is confirmed equivalent -- bit for bit -- to calling it once per
block with that block's own bound, in the same order:

    >>> import numpy as np
    >>> highs = [100, 50, 100, 30]
    >>> rng1 = np.random.default_rng(7)
    >>> [rng1.integers(0, h) for h in highs]
    >>> rng2 = np.random.default_rng(7)
    >>> list(rng2.integers(0, np.array(highs)))
    # identical

This module does not replace or modify RegimeSwitchingValuationVARSimulator,
which remains the reference implementation -- see
tests/test_fast_regime_switching_path_simulation.py for the parity tests
that pin this port to it. Hardcodes num_variables == 4 and a 2-state
regime model (0=Expansion, 1=Contraction), matching the reference's own
fixed design (`self.pool`/`self.block_size` keyed on {0, 1} throughout
path_simulation.py).
"""

from typing import Optional

import numpy as np
from numba import njit, prange

from market_modelling.path_simulation import RegimeSwitchingValuationVARSimulator

N_VARIABLES = 4


@njit(parallel=True, cache=True)
def _simulate_regime_path_fast(initial_regimes, rand_draws, p01, p10, months):
    """
    Numba port of the regime-path per-step loop (path_simulation.py,
    lines ~1292-1299). `rand_draws` and `initial_regimes` are the exact
    same RNG draws the reference makes (rng.choice for the initial state,
    rng.random for the switch draws) -- only the sequential application of
    those draws moves into this kernel. prange over paths (outer loop):
    each path's regime sequence only depends on its own draws, not on
    other paths, and is itself a strict recurrence over time.
    """
    num_paths = initial_regimes.shape[0]
    regime_path = np.empty((num_paths, months), dtype=np.int64)
    for i in prange(num_paths):
        prev = initial_regimes[i]
        regime_path[i, 0] = prev
        for t in range(1, months):
            p_switch = p01 if prev == 0 else p10
            if rand_draws[i, t - 1] < p_switch:
                prev = 1 - prev
            regime_path[i, t] = prev
    return regime_path


@njit(cache=True)
def _count_total_blocks(regime_path, block_size0, block_size1):
    """First pass of the Step 2 port: counts how many blocks the
    reference's nested while loop would draw in total, across every path,
    without touching any RNG -- purely a function of the (already
    generated) regime_path and the two regimes' block sizes."""
    num_paths, months = regime_path.shape
    total = 0
    for path_idx in range(num_paths):
        t = 0
        while t < months:
            regime = regime_path[path_idx, t]
            run_end = t
            while run_end < months and regime_path[path_idx, run_end] == regime:
                run_end += 1
            run_length = run_end - t
            bsize = block_size0 if regime == 0 else block_size1
            filled = 0
            while filled < run_length:
                take = min(bsize, run_length - filled)
                filled += take
                total += 1
            t = run_end
    return total


@njit(cache=True)
def _build_block_metadata(regime_path, block_size0, block_size1, total_blocks):
    """
    Second pass: re-walks the identical path/run/block structure
    `_count_total_blocks` just measured, this time recording each block's
    (path, destination offset, length, regime) -- in exactly the order
    the reference's while loop would visit them, which is the order its
    rng.integers() calls would be issued in. That order is what lets the
    one batched rng.integers(0, bounds) call in
    simulate_regime_switching_paths_fast reproduce the reference's
    sequence of draws bit for bit.
    """
    num_paths, months = regime_path.shape
    path_of = np.empty(total_blocks, dtype=np.int64)
    lo_of = np.empty(total_blocks, dtype=np.int64)
    take_of = np.empty(total_blocks, dtype=np.int64)
    regime_of = np.empty(total_blocks, dtype=np.int64)

    bi = 0
    for path_idx in range(num_paths):
        t = 0
        while t < months:
            regime = regime_path[path_idx, t]
            run_end = t
            while run_end < months and regime_path[path_idx, run_end] == regime:
                run_end += 1
            run_length = run_end - t
            bsize = block_size0 if regime == 0 else block_size1
            filled = 0
            while filled < run_length:
                take = min(bsize, run_length - filled)
                path_of[bi] = path_idx
                lo_of[bi] = t + filled
                take_of[bi] = take
                regime_of[bi] = regime
                bi += 1
                filled += take
            t = run_end
    return path_of, lo_of, take_of, regime_of


@njit(parallel=True, cache=True)
def _scatter_blocks(
    increments, pool0, pool1, path_of, lo_of, take_of, regime_of, starts,
):
    """
    Writes each block's sampled rows into `increments`. Every block
    occupies a disjoint (path, time) slice -- the (path_of, lo_of,
    take_of) triples partition each path's full time axis exactly once,
    by construction of _build_block_metadata -- so this is safe to run
    with prange over blocks.
    """
    total_blocks = path_of.shape[0]
    for bi in prange(total_blocks):
        regime = regime_of[bi]
        start = starts[bi]
        take = take_of[bi]
        p_idx = path_of[bi]
        lo = lo_of[bi]
        if regime == 0:
            for r in range(take):
                for k in range(N_VARIABLES):
                    increments[p_idx, lo + r, k] = pool0[start + r, k]
        else:
            for r in range(take):
                for k in range(N_VARIABLES):
                    increments[p_idx, lo + r, k] = pool1[start + r, k]


@njit(parallel=True, cache=True)
def _simulate_regime_recurrence_fast(
    increments,                 # (num_paths, months, 4)
    initial_spx, initial_cpi, initial_3m, initial_5y,
    initial_cape, target_cape, phi_cape, gamma_cape,
    equilibrium_equity_drift, phi_rate,
    target_yield_3m, target_yield_5y,
    historical_spx_mean,
    months,
):
    """
    Numba port of Step 3 (path_simulation.py, lines ~1505-1538) -- the
    same equilibrium-drift-plus-CAPE-drag recurrence as
    HybridValuationVARSimulator's (see fast_hybrid_path_simulation.py),
    minus the VAR matmul since this model consumes `increments` directly.
    Same outer-prange-over-paths layout, for the same reason: one
    parallel region total, each thread runs one path through every step.
    """
    num_paths = increments.shape[0]
    dt = 1.0 / 12.0
    log_target_cape = np.log(target_cape)
    log_initial_cape = np.log(initial_cape)

    spx_paths = np.zeros((num_paths, months))
    cpi_paths = np.zeros((num_paths, months))
    yield_3m_paths = np.zeros((num_paths, months))
    yield_5y_paths = np.zeros((num_paths, months))

    for i in prange(num_paths):
        curr_spx = initial_spx
        curr_cpi = initial_cpi
        curr_3m = initial_3m
        curr_5y = initial_5y
        log_cape = log_initial_cape

        for step in range(months):
            raw0 = increments[i, step, 0]
            raw1 = increments[i, step, 1]
            raw2 = increments[i, step, 2]
            raw3 = increments[i, step, 3]

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

    return spx_paths, cpi_paths, yield_3m_paths, yield_5y_paths


def simulate_regime_switching_paths_fast(
    simulator: RegimeSwitchingValuationVARSimulator,
    simulation_months: int = 360,
    num_paths: int = 10000,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated equivalent of
    RegimeSwitchingValuationVARSimulator.simulate_paths. Every RNG call
    is made in the same order, with the same arguments (batched where
    confirmed equivalent -- see module docstring), as the reference; only
    the RNG-free bookkeeping and the array math move into Numba. See
    tests/test_fast_regime_switching_path_simulation.py for the parity
    tests pinning this to the reference.
    """
    if simulator.transition_matrix is None:
        raise RuntimeError("Model is not fitted. Call fit() first.")
    assert simulator.stationary_dist is not None
    assert simulator.historical_spx_mean is not None
    assert simulator.expected_inflation is not None
    assert simulator.initial_spx_level is not None
    assert simulator.initial_cpi_level is not None
    assert simulator.initial_yield_3m is not None
    assert simulator.initial_yield_5y is not None
    assert simulator.target_yield_3m is not None
    assert simulator.target_yield_5y is not None
    assert 0 in simulator.pool and 1 in simulator.pool, (
        "fast_regime_switching_path_simulation hardcodes a 2-state "
        "regime model (0, 1), matching the reference's own design")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0

    # Step 1: identical RNG draws to the reference, same order.
    initial_regimes = rng.choice(
        [0, 1], size=num_paths, p=simulator.stationary_dist)
    rand_draws = rng.random((num_paths, simulation_months - 1))
    regime_path = _simulate_regime_path_fast(
        initial_regimes, rand_draws,
        float(simulator.transition_matrix[0, 1]),
        float(simulator.transition_matrix[1, 0]),
        simulation_months,
    )

    # Step 2: RNG-free bookkeeping in Numba, then one batched rng.integers
    # call reproducing the reference's per-block draw sequence.
    pool0 = np.ascontiguousarray(simulator.pool[0])
    pool1 = np.ascontiguousarray(simulator.pool[1])
    bsize0 = simulator.block_size[0]
    bsize1 = simulator.block_size[1]
    max_start0 = pool0.shape[0] - bsize0
    max_start1 = pool1.shape[0] - bsize1

    total_blocks = _count_total_blocks(regime_path, bsize0, bsize1)
    path_of, lo_of, take_of, regime_of = _build_block_metadata(
        regime_path, bsize0, bsize1, total_blocks)

    bounds = np.where(regime_of == 0, max_start0 + 1, max_start1 + 1)
    starts = rng.integers(0, bounds)

    increments = np.zeros((num_paths, simulation_months, N_VARIABLES))
    _scatter_blocks(
        increments, pool0, pool1, path_of, lo_of, take_of, regime_of, starts)

    # Step 3: equilibrium drift + CAPE valuation drag recurrence.
    equilibrium_equity_drift = (
        simulator.earnings_growth + simulator.expected_inflation) * dt

    return _simulate_regime_recurrence_fast(
        increments,
        simulator.initial_spx_level, simulator.initial_cpi_level,
        simulator.initial_yield_3m, simulator.initial_yield_5y,
        simulator.initial_cape, simulator.target_cape, simulator.phi_cape,
        simulator.gamma_cape, equilibrium_equity_drift, simulator.phi_rate,
        simulator.target_yield_3m, simulator.target_yield_5y,
        simulator.historical_spx_mean,
        simulation_months,
    )
