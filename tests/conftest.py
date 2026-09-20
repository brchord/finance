import multiprocessing

import numpy as np
import pandas as pd
import pytest

# Each MonteCarloCLI.run() / MonteCarloEngine.run() starts a fresh process
# pool. Under pytest the forkserver's workers would otherwise re-import
# monte_carlo (pandas, yfinance, ...) every time, which dominates the runtime
# of the MC tests.
multiprocessing.set_forkserver_preload(["monte_carlo"])


def make_market_levels(seed: int = 7) -> pd.DataFrame:
    """
    Synthetic daily market levels in the same schema as the cached parquet
    (spx_close, yield_3m, yield_5y, cpi), so tests never touch the real
    market_data.parquet or the network. Fully determined by `seed`.
    """
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


@pytest.fixture(scope="session")
def market_cache(tmp_path_factory):
    """Path to a parquet cache of synthetic market levels."""
    path = tmp_path_factory.mktemp("market") / "market.parquet"
    make_market_levels().to_parquet(path)
    return path


@pytest.fixture(scope="session")
def aligned_market(market_cache):
    """(levels, returns) exactly as the CLI builds them from a cache file."""
    from market_data.yf_fred_market_data import MarketDataManager
    return MarketDataManager(
        cache_filepath=str(market_cache)).get_aligned_real_returns()
