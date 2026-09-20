# Testing

## Running

```
.venv/bin/python -m pytest -q     # full suite (~5s)
.venv/bin/python -m mypy .        # type check (must stay clean)
```

Run a single module or test with the usual pytest selectors, e.g.
`.venv/bin/python -m pytest tests/test_strategy.py -k Ruin`.

## Test setup

- `tests/conftest.py` builds a **synthetic, seeded** daily market cache
  (parquet) so tests never touch `market_data.parquet` or the network.
  Fixtures: `market_cache`, `aligned_market`.
- `conftest.py` also calls `multiprocessing.set_forkserver_preload(["monte_carlo"])`.
  Without it, each per-portfolio process pool re-imports pandas/yfinance and
  the Monte Carlo tests take ~55s instead of ~2s.
- Live-fetch paths (`_fetch_remote_data`, IBKR) are not tested.

## Golden snapshot

`tests/golden/monte_carlo_agg.json` pins the aggregated Monte Carlo output for
a fixed seed and config (checked by `test_golden_aggregates`). A failure means
simulation output changed. If the change is intended (e.g. a model fix),
regenerate and review the diff before committing:

```
UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_monte_carlo.py
git diff tests/golden/monte_carlo_agg.json
```

Check that only the models you expected to change moved, and mention the
snapshot change in the commit message.

## What each module covers

| Module | Coverage |
|--------|----------|
| `tests/test_tax_regimes.py` | `tax_models/regimes.py`: bracket tables, `compute_tax`, the named scenarios, and `build_tax_regime`. |
| `tests/test_tax_lots.py` | `TaxLotTracker`: FIFO lot matching, short- vs long-term split (11 months is short-term), losses as negative gains, non-positive buys ignored, overselling raises. |
| `tests/test_strategy.py` | `LongSPYWithTreasuryLadders`: constructor validation (allocations sum to 100%, spending must be positive), JSON construction, needed-liquidity calculation, NAV conservation, ruin (zero-filled path, no warnings), quarterly dividends, realized-gain roll-ups, tax cadence and amounts, state reset, `full_book`. |
| `tests/test_path_simulation.py` | Invariants for all six path simulators, unique/stable model names, abstract base class, regime-switching fit checks, and `label_regimes` (contraction runs from the month after the peak through the trough month inclusive, on both month-start and month-end indexes). |
| `tests/test_monte_carlo.py` | Config sweeps, run shape, seed reproducibility, common random numbers across tax regimes, aggregation maths, and the golden snapshot. |
