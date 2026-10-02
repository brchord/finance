"""
Stage 2 benchmark: MonteCarloCLI.run(backend="process") vs.
backend="numba") wall-clock time, at (a reduced slice of) the large-run
shape from doc/plans/GPU Optimization Plan.md: 2 models, 6 equity
allocations, 6 spending levels, 1 tax regime, 50,000 paths, 756 months.

Not a test -- run manually:
    python benchmarks/bench_cli_backends.py [total_paths]
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import monte_carlo as mc


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
    total_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    years_to_simulate = 63

    config = dict(
        yearly_spending_floor=60_000, yearly_spending_ceil=110_000,
        spend_increments=10_000, equity_floor=0.4, equity_ceil=0.9,
        weight_increments=0.1, initial_nav=1_000_000,
        years_to_simulate=years_to_simulate,
        retirement_age=60, total_paths=total_paths, workers=os.cpu_count(),
        tax_regimes=["current_law_indexed"], master_seed=123,
        models=["HybridValuationVARSimulator",
                "RegimeSwitchingValuationVARSimulator"],
    )
    n_portfolios = 6 * 6 * 2 * 1
    print(f"{n_portfolios} portfolios x {total_paths} paths x "
          f"{years_to_simulate*12:.0f} months, "
          f"{os.cpu_count()} cores")

    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "market.parquet")
        make_market_levels().to_parquet(cache)
        config_path = os.path.join(d, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        for backend in ["numba", "process"]:
            cli = mc.MonteCarloCLI(config_path, cache)
            t0 = time.perf_counter()
            cli.run(backend=backend)
            elapsed = time.perf_counter() - t0
            print(f"backend={backend}: {elapsed:.1f}s total "
                  f"({elapsed/n_portfolios:.2f}s/portfolio)")


if __name__ == "__main__":
    main()
