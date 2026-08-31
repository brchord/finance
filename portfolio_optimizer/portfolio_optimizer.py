"""
portfolio_simulation.py

This script provides a basic simulation for Stochastic Volatility
with Correlated Jumps (SVCJ) to be used to simulate equities
markets to later on be integrated into a Monte Carlo simulation
engine designed to stress test multiple trading strategies using
Conditional Value at Risk (CVaR).
"""

import time

import numpy as np

from market_modelling.dsvi import DynamicSVI
from market_modelling.svcj import SVCJSimulation
from portfolio_models.short_put_model import ShortSPXPutStrategy
from portfolio_models.put_credit_spreads_model import SPXPutCreditSpreadStrategy

# ============================================
# 5. Fitting current volatility curves with
#    real market data and start the
#    simulation.
# NOTE: Real options chain data from
#       Aug 15th 2026.
# ============================================
def main():
    """
    Simulation entry point.
    """

    spx_chain_data = """
    6900	.2283
    7200	.1907
    7375	.1698
    7475	.1586
    7550	.1506
    7625	.1433
    7700	.1367
    7750	.1328
    7825	.1280
    7875	.1256
    7925	.1238
    8000	.1213
    8050	.1210
    8100	.1214
    8300	.1330
    """
    spx_chain_dtes = 44.0
    spot_spx = 7786.00
    spot_vix = 0.1425
    strikes = []
    ivs = []

    # Start the timer
    start_time = time.perf_counter()
    for row in spx_chain_data.strip().split('\n'):
        strike, iv = row.split('\t')
        strikes.append(float(strike))
        ivs.append(float(iv))

    svi = DynamicSVI(np.array(strikes), np.array(ivs), spot_spx, spx_chain_dtes / 365)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"SVI Execution time: {execution_time:.3f}s")

    start_time = time.perf_counter()
    svcj = SVCJSimulation()
    spx, vix = svcj.generate_path(spot_spx, spot_vix)
    vix3m = svcj.derive_vix3m(vix)

    spx_put_trade_strategy = ShortSPXPutStrategy(distribution=12_500)
    spx_pcs_trade_strategy = SPXPutCreditSpreadStrategy(distribution=12_500)
    spx_put_trade_strategy.run_simulation(
        spot_spx=spx,
        spot_vix=vix,
        vix3m=vix3m,
        svi=svi,
        initial_nav=4_000_000,
        days=252)
    spx_pcs_trade_strategy.run_simulation(
        spot_spx=spx,
        spot_vix=vix,
        vix3m=vix3m,
        svi=svi,
        initial_nav=4_000_000,
        days=252)

    # Calculate elapsed time
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Path execution time: {execution_time:.3f}s")


if __name__ == "__main__":
    main()
