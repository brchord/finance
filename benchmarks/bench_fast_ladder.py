"""
Benchmark: portfolio_models.fast_ladder vs. the reference
LongSPYWithTreasuryLadders.run_simulation, at a large real-world run shape
(50,000 paths, 756 months).

Not a test -- run manually:
    python benchmarks/bench_fast_ladder.py
"""
import os
import tempfile
import time

import numpy as np
import pandas as pd

import market_modelling.path_simulation as ps
from market_data.yf_fred_market_data import MarketDataManager
from portfolio_models.fast_ladder import (
    flatten_tax_regime, run_simulation_fast, run_simulation_fast_batch)
from portfolio_models.linear_models import LongSPYWithTreasuryLadders as LSTL
from tax_models.regimes import build_tax_regime

MONTHS = 756
LARGE_PATHS = 50_000
PROFILE_PATHS = 500
MODELS: list[type[ps.PathSimulator]] = [
    ps.HybridValuationVARSimulator, ps.RegimeSwitchingValuationVARSimulator]


def make_market_levels(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1982-01-01", "2024-12-31")
    n = len(days)
    spx = 120.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, n)))
    yield_3m = np.clip(
        0.05 + np.cumsum(rng.normal(0.0, 0.0008, n)) * 0.3, 0.001, 0.12)
    yield_5y = np.clip(yield_3m + 0.01 + rng.normal(0.0, 0.0005, n),
                        0.005, 0.14)
    cpi = 97.0 * np.exp(np.cumsum(np.full(n, 0.03 / 252) +
                                   rng.normal(0.0, 0.0003, n)))
    return pd.DataFrame(
        {"spx_close": spx, "yield_3m": yield_3m,
         "yield_5y": yield_5y, "cpi": cpi}, index=days)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "market.parquet")
        make_market_levels().to_parquet(cache)
        levels, returns = MarketDataManager(
            cache_filepath=cache).get_aligned_real_returns()

    tax_regime = build_tax_regime("current_law_indexed")
    max_years = (MONTHS + 11) // 12 + 1
    flat = flatten_tax_regime(tax_regime, max_years)

    for model_cls in MODELS:
        sim = model_cls()
        sim.fit(returns, levels)
        spx_paths, cpi_paths, tbill_paths, tnote_paths = sim.simulate_paths(
            MONTHS, LARGE_PATHS, seed=1)

        print(f"\n{'='*70}\n{model_cls.name()}\n{'='*70}")

        # --- Reference (Python loop over paths, scalar run_simulation) ---
        strategy = LSTL(0.6, 0.4, 60_000.0, tax_regime=tax_regime)
        t0 = time.perf_counter()
        for i in range(PROFILE_PATHS):
            strategy.run_simulation(
                spx=spx_paths[i, :], cpi=cpi_paths[i, :],
                yield3m=tbill_paths[i, :], yield5y=tnote_paths[i, :],
                initial_nav=1_000_000.0, months=MONTHS, full_book=False)
        t_ref = (time.perf_counter() - t0) / PROFILE_PATHS
        print(f"reference run_simulation:  {t_ref*1000:.4f} ms/path "
              f"(extrapolated {t_ref*LARGE_PATHS:.1f}s @ {LARGE_PATHS} paths)")

        # --- Numba single-path (compile once, then time) ---
        run_simulation_fast(
            spx_paths[0], cpi_paths[0], tbill_paths[0], tnote_paths[0],
            1_000_000.0, MONTHS, 0.6, 0.4, 60_000.0, 0.01, *flat)  # warmup/JIT
        t0 = time.perf_counter()
        for i in range(PROFILE_PATHS):
            run_simulation_fast(
                spx_paths[i], cpi_paths[i], tbill_paths[i], tnote_paths[i],
                1_000_000.0, MONTHS, 0.6, 0.4, 60_000.0, 0.01, *flat)
        t_fast = (time.perf_counter() - t0) / PROFILE_PATHS
        print(f"fast_ladder (1 thread):    {t_fast*1000:.4f} ms/path "
              f"(extrapolated {t_fast*LARGE_PATHS:.1f}s @ {LARGE_PATHS} paths) "
              f"-- {t_ref/t_fast:.1f}x reference")

        # --- Numba parallel batch, full large-run shape ---
        run_simulation_fast_batch(
            spx_paths[:2], cpi_paths[:2], tbill_paths[:2], tnote_paths[:2],
            1_000_000.0, MONTHS, 0.6, 0.4, 60_000.0, 0.01, *flat)  # warmup
        t0 = time.perf_counter()
        run_simulation_fast_batch(
            spx_paths, cpi_paths, tbill_paths, tnote_paths,
            1_000_000.0, MONTHS, 0.6, 0.4, 60_000.0, 0.01, *flat)
        t_batch = time.perf_counter() - t0
        print(f"fast_ladder_batch (parallel): {t_batch:.3f}s total "
              f"@ {LARGE_PATHS} paths ({t_batch/LARGE_PATHS*1000:.4f} ms/path) "
              f"-- {(t_ref*LARGE_PATHS)/t_batch:.1f}x reference")


if __name__ == "__main__":
    main()
