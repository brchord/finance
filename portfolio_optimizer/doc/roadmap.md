
# Further implementation roadmap

  1. [x] Finish the rolling trade logic to properly simulate the delta -5 roll at a credit scenario.
  2. [ ] Spot check the IV of further expiration options and confirm they make sense.
  3. [x] Implement a stopgap condition to avoid rolling indefinitely.
  4. [ ] Implement a stitchable segment simulation architecture
     - First, make sure every individual simulation records their terminal NAV and IV and all the remaining positions are closed so the portfolio is easy to carry forward with these 2 parameters as new conditions for a subsequent simulation.
  5. [ ] Compute the CVaR:
     - Sort all the terminal returns in ascending order, and compute the average returns up to the P percentile.
     - We're interested in -15% 99-CVaR and -7% 90 CVaR (confirm this again with the LLM)
  6. [ ] Plot the return distributions.
  7. [ ] Build modular composable portfolios
     - [ ] Covered calls.
     - [x] Pure long equity portfolios based on a given market path (i.e. long SPY).
     - [x] 100% T-Bills and adjustable position sized naked short equity short put options (i.e. short SPX puts).
     - [x] 100% T-bills and adjustable position size equity put credit spreads (i.e. short SPX Put-Credit-Spreads).
     - [x] 100% T-Bills.  Used as baseline benchmark.
     - [x] A portfolio that can produce linear combinations of the aforementioned fundamental portfolios.
  8. [ ] Implement a tool to find the efficient frontier varying a matrix of portfolio parameters.
  9. [ ] Figure out how to discount the inflation.
 10. [ ] Implement a more robust logging infrastructure.
 11. [ ] Portfolio comparison using the exact trajectories.
     - [ ] Generate the trajectories first and then run each desired portfolio configuration in parallel with the previously generated trajectories.
 12. [ ] Make more parametrizable choices for the SPX Short Put portfolio:
     - [ ] Delta rolling criteria.
     - [ ] Delta hard close criteria.
     - [ ] Tail expiration hard close criteria.
     - [ ] Notional leverage reduction during wade-in.
 13. [x] Get rid of the hard coded array of arguments on the options book and use a dictionary instead to make the code self-documenting.
 14. [ ] Transaction tracking incomplete for all portfolios.
 15. [ ] Historical data tool needs to pull both SPX, VIX and VIX3M.