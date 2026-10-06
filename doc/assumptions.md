# Market assumptions and robustness runs

The allocation the engine recommends is driven largely by a handful of
long-run assumptions. They are now config fields, recorded in every
results file under `assumptions`, so runs made under different
assumptions can't be confused with each other.

## Config fields

| Field | Default | Meaning |
|-------|---------|---------|
| `dividend_yield` | `0.01` | SPY cash dividend yield used by the strategy. |
| `simulator_params` | `{}` | Constructor keyword arguments for the path simulators. Each model receives only the keys its constructor accepts (logged at INFO); a key no selected model accepts is rejected as a likely typo. |
| `ruin_ages` | `[75, 85, 95]` | Ages for the unconditional `ruin_prob_by_age` metric. |

Useful `simulator_params` keys (CAPE-drag models:
`HybridValuationVARSimulator`, `RegimeSwitchingValuationVARSimulator`;
`ValuationAdjustedVARSimulator` takes the CAPE ones but not the drift
ones):

| Key | Default | Notes |
|-----|---------|-------|
| `initial_cape` | `34.0` | Starting Shiller CAPE. **Set it to the current value for every run.** Previously the constructors set 41 but `fit()` silently replaced it with 34, so every run so far used 34. |
| `target_cape` | `22.0` | Long-run CAPE the drag pulls toward. ~16 full-history median, ~20 post-1950, ~25-27 post-1990. |
| `annual_earnings_growth` | `0.02` | Real aggregate earnings growth. |
| `annual_buyback_yield` | `0.0` | Net share-repurchase yield, added to the per-share price drift. `0` reproduces the original model. |

With the defaults, the long-run real total return of equities is about
`annual_earnings_growth + dividend_yield` = 3%, before the CAPE drag. At
a CAPE of 22 the earnings yield is ~4.5%, so a steady state with
dividends plus buybacks of ~3-4% of price and ~2% growth would return
~5-6% real; the defaults sit well below that.

## Suggested robustness grid

Run the same sweep (allocation x spending, 50k paths, one master seed)
once per row and compare the passing cells:

| Run | `initial_cape` | `target_cape` | `annual_buyback_yield` | `dividend_yield` |
|-----|----------------|---------------|------------------------|------------------|
| original (as before) | 34 | 22 | 0.0 | 0.010 |
| current valuation only | current | 22 | 0.0 | 0.010 |
| + buybacks | current | 22 | 0.015 | 0.012 |
| + modern CAPE norm | current | 26 | 0.015 | 0.012 |
| pessimistic bound | current | 20 | 0.0 | 0.010 |

If the selected allocation is stable across rows, it is robust. If it
moves a lot between "current valuation only" and "+ buybacks", the
recommendation is being set by that one assumption, and the choice of
where to sit on that range should be made deliberately.

## Metrics for the selection

- `ruin_prob_by_age`: unconditional P(ruin before age) over **all**
  paths. Prefer it to `ruin_month_median` / `_es5` / `_es10`, which are
  computed over ruined paths only and can rank a rarely-ruining portfolio
  below an often-ruining one whose ruins happen later.
- `p*_return` are **nominal** (terminal NAV in future dollars);
  `p*_real_return` and `real_nav_bands` deflate each path by its own
  simulated price level (`monte_carlo.price_levels`), i.e. today's
  dollars.
