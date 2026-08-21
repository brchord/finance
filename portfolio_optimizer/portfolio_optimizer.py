"""
portfolio_simulation.py

This script provides a basic simulation for Stochastic Volatility
with Correlated Jumps (SVCJ) to be used to simulate equities
markets to later on be integrated into a Monte Carlo simulation
engine designed to stress test multiple trading strategies using
Conditional Value at Risk (CVaR).
"""

import math
import time

import numpy as np
import pandas as pd

import market_modelling.dsvi as dsvi
import market_modelling.svcj as svcj

from portfolio_models.short_spx_bond_overlay import ShortSPXPutStrategy

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

    svi = dsvi.DynamicSVI(np.array(strikes), np.array(ivs), spot_spx, spx_chain_dtes / 365)
    market_sim = svcj.SVCJSimulation()
    spx, vix = market_sim.generate_path(spot_spx, spot_vix)
    vix3m = market_sim.derive_vix3m(vix)
    trade_strategy = ShortSPXPutStrategy(spx, vix, vix3m, svi)
    trade_strategy.run_simulation(nav=8000000, monthly_distribution=12500, notional_leverage=0.5)

    # Calculate elapsed time
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.3f}s")


if __name__ == "__main__":
    main()


###############################################################################
# Further implementation roadmap:
#
#  1. Finish the rolling trade logic to properly simulate the delta 0.35
#     roll at a credit scenario. [DONE]
#  2. Spot check the IV of further expiration options and confirm
#     they make sense.
#  3. Implement a stopgap condition to avoid rolling indefinitely. [DONE]
#  4. Implement a stitchable segment simulation architecture
#     4.1 First, make sure every individual simulation records their terminal
#         NAV and IV and all the remaining positions are closed so the
#         portfolio is easy to carry forward with these 2 parameters as new
#         conditions for a subsequent simulation.
#  5. Compute the CVaR:
#     5.1 Sort all the terminal returns in ascending order, and compute the
#         average returns up to the P percentile.
#     5.2 We're interested in -15% 99-CVaR and -7% 90 CVaR (confirm this again
#         with the LLM)
#  6. Plot the return distributions.
#  7. Consider alternative portfolios.
#     7.1 15% SPY and 85% T-Bills + Short SPX puts
#     7.2 100% T-bills and 0.75x notional SPX 5-15 delta credit spreads
#     7.3 Pure SPY + T-Bills combinations as portfolio benchmarks.
#     7.3 Covered calls.
#     7.4 Implement a tool to find the efficient frontier varying a matrix
#         of portfolio parameters.
#  8. Figure out how to discount the inflation
#  9. Implement a more robust logging infrastructure.
# 10. Portfolio comparison using the exact trajectories.
#     10.1 Generate the trajectories first and then run each desired
#          portfolio configuration in parallel with the previously
#          generated trajectories.
# 11. Make more parametrizable choices for the SPX Short Put portfolio:
#     11.1 Delta rolling criteria.
#     11.2 Delta hard close criteria.
#     11.3 Tail expiration hard close criteria.
#     11.4 Notional leverage reduction during wade-in.
# 12. Get rid of the hard coded array of arguments on the options book and
#     use a dictionary instead to make the code self-documenting.
###############################################################################
