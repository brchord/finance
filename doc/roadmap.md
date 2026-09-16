
# Further implementation roadmap

- [x] Plot the return distributions.
- [x] Build modular composable portfolios
    - [x] Pure long equity portfolios based on a given market path (i.e. long SPY).
    - [x] 100% T-Bills.  Used as baseline benchmark.
    - [x] A portfolio that can produce linear combinations of the aforementioned fundamental portfolios.
    - [x] Retirement portfolio comprised of a T-Bill/T-Note ladder and long equity (SPX).
- [x] Implement a tool to find the efficient frontier varying a matrix of portfolio parameters.
- [x] Figure out how to discount the inflation.
- [x] Implement a more robust logging infrastructure.
- [x] Portfolio comparison using the exact trajectories.
- [x] Transaction tracking incomplete for all portfolios.
- [x] Unify the code for the MC driver and the notebook code responsible for aggreggating simulation statistics.