"""
mc_kickoff.py

Simple utility to run a sequence of MC simulations leveraging
Residual Boostrap Path simulation and using a portfolio comprising
of Long equity with T-Bill/T-Note ladders.
"""

import time

import numpy as np
import pandas as pd

from market_modelling.residual_boostrap import VARResidualBootstrapSimulator
from monte_carlo import MonteCarloEngine
from market_data.yf_fred_market_data import MarketDataManager
from portfolio_models.linear_models import LongSPYWithTreasuryLadders

portfolio_config = {
    "type": "LongSPYWithTreasuryLadders",
    "equity_allocation": None,
    "ladder_allocation": None,
    "yearly_spending":  None,
    "dividend_yield": 0.01,
    "average_inflation": 0.034
}

allocations = [
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1)
]

spendings = [
    150_000,
    200_000,
    250_000,
    300_000,
    350_000
]

def main():
    "Main entrypoint"
    mdm = MarketDataManager(cache_filepath="market_data.parquet")
    levels, returns = mdm.get_aligned_data()
    perf_counters = {}
    perf_counters["setup"] = []
    perf_counters["simulation"] = []
    perf_counters["data_storage"] = []

    rng = np.random.default_rng(250722)

    for a in allocations:
        for s in spendings:
            portfolio_config["equity_allocation"] = a[0]
            portfolio_config["ladder_allocation"] = a[1]
            portfolio_config["yearly_spending"] = s

            setup_start = time.perf_counter()

            simulator = VARResidualBootstrapSimulator()
            simulator.fit(returns, levels)

            strategy = LongSPYWithTreasuryLadders.from_json_object(portfolio_config)
            mc = MonteCarloEngine(strategy, simulator, 15876, 8_500_000, rng.integers(1 << 32))

            setup_end = time.perf_counter()

            perf_counters["setup"].append(setup_end - setup_start)

            sim_start = time.perf_counter()
            spx, nav, rets, pct, dds = mc.run(total_paths=30000, n_workers=18)
            sim_end = time.perf_counter()

            perf_counters["simulation"].append(sim_end - sim_start)

            data_start = time.perf_counter()
            df = pd.DataFrame({
                    "Terminal SPX": spx,
                    "Terminal NAV": nav,
                    "Total Return": rets,
                    "Pct of Initial NAV": pct,
                    "Max Drawdowns": dds
                })
            filename = f"spy_ladder_{int(s/1000)}k_{int(a[0]*100.0)}_" + \
                        f"{int(a[1]*100.0)}.csv"
            df.to_csv(filename)
            data_end = time.perf_counter()
            perf_counters["data_storage"].append(data_end - data_start)

    perf_df = pd.DataFrame(perf_counters)
    perf_df.to_csv("mc_perf_stats.csv")

if __name__ == '__main__':
    main()
