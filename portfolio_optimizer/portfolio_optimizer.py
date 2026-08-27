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
from portfolio_models.option_models import ShortSPXPutStrategy
from portfolio_models.option_models import SPXPutCreditSpreadStrategy

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

    spx_put_trade_strategy = ShortSPXPutStrategy(
        spot_spx=spx,
        spot_vix=vix,
        vix3m=vix3m,
        svi=svi,
        distribution=12_500)
    spx_pcs_trade_strategy = SPXPutCreditSpreadStrategy(
        spot_spx=spx,
        spot_vix=vix,
        vix3m=vix3m,
        svi=svi,
        distribution=12_500
    )
    spx_put_trade_strategy.run_simulation(4_000_000, 252)
    spx_pcs_trade_strategy.run_simulation(4_000_000, 252)

    # Calculate elapsed time
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Path execution time: {execution_time:.3f}s")


if __name__ == "__main__":
    main()


###############################################################################
# Further implementation roadmap:
#
#  1. [DONE] Finish the rolling trade logic to properly simulate the delta 0.35
#     roll at a credit scenario.
#  2. Spot check the IV of further expiration options and confirm
#     they make sense.
#  3. [DONE] Implement a stopgap condition to avoid rolling indefinitely.
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
#  7. Build modular composable portfolios
#     7.1 [DONE] Pure long equity portfolios based on a given market path
#         (i.e. long SPY).
#     7.2 [DONE] 100% T-Bills and adjustable position sized naked short
#         equity short put options (i.e. short SPX puts).
#     7.2 [DONE] 100% T-bills and adjustable position size equity
#         put credit spreads (i.e. short SPX Put-Credit-Spreads).
#     7.3 [DONE] 100% T-Bills.  Used as baseline benchmark.
#     7.3 Covered calls.
#     7.4 [DONE] A portfolio that can produce linear combinations
#         of the aforementioned fundamental portfolios.
#  8. Implement a tool to find the efficient frontier varying a matrix
#     of portfolio parameters.
#  9. Figure out how to discount the inflation.
# 10. Implement a more robust logging infrastructure.
# 11. Portfolio comparison using the exact trajectories.
#     11.1 Generate the trajectories first and then run each desired
#          portfolio configuration in parallel with the previously
#          generated trajectories.
# 12. Make more parametrizable choices for the SPX Short Put portfolio:
#     12.1 Delta rolling criteria.
#     12.2 Delta hard close criteria.
#     12.3 Tail expiration hard close criteria.
#     12.4 Notional leverage reduction during wade-in.
# 13. [DONE] Get rid of the hard coded array of arguments on the options
#     book and use a dictionary instead to make the code self-documenting.
###############################################################################
