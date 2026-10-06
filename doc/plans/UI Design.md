# Monte Carlo Retirement Engine — UI Design Plan

This document captures the design decisions for adding a local web UI to the
Monte Carlo simulation engine (`monte_carlo.py`). It is the reference for
implementation work in Claude Code.

## Context

- The engine is pure Python with Numba-compiled hot paths, driven by a CLI
  (`monte_carlo.py`), and uses a `ProcessPoolExecutor` for parallelism.
- A run is a **sweep**: the config defines a spending range, an equity
  allocation range, a list of tax regimes and a list of models. The engine
  simulates every combination ("cell") with `total_paths` paths. Example:
  spending 80k–120k step 5k × equity 30–80% step 10% × 2 models = 54 cells.
- Config (JSON): `initial_nav`, `retirement_age`, `years_to_simulate`,
  `yearly_spending_floor/ceil`, `spend_increments`, `equity_floor/ceil`,
  `weight_increments`, `total_paths`, `models`, `tax_regimes`,
  `master_seed` (optional), `workers`, and the market assumptions
  `simulator_params` and `dividend_yield` (doc/assumptions.md).
- Outputs today: a raw JSON (per-path terminal NAVs, ~230 MB at 50k paths —
  the UI must never read it) and an aggregated JSON with one entry per cell.
- Ruin timing is tracked in **months since retirement**
  (`ruin_histogram`, one bin per month). The UI converts to ages:
  `age = retirement_age + month / 12`.
- With `--backend numba` (~47x faster than the original backend), a full
  run takes far less than the original 15–30 min. Measure it on the real
  config before sizing the progress UI.
- Current visualization is a Jupyter notebook.
- **Single user, local only.** No auth, no hosting.
- Use case: a recurring review every 6–12 months to recalibrate the plan
  from actual NAV.

## Goals

1. Answer the review's central question fast: **what is the highest yearly
   spending I can sustain, and with which equity allocation?**
2. Support the actual workflow: a wide sweep first, then progressively finer
   sweeps zooming in on the frontier.
3. Replace the notebook with a persistent dashboard.
4. Keep the engine a standalone program; the CLI keeps working as-is.

## Decision model

This is how the user chooses a cell. The UI encodes it rather than leaving
the user to scan tables.

### Models

- **Decision model:** `RegimeSwitchingValuationVARSimulator` (regime switching
  with VAR and CAPE discount — the most conservative). All gating and
  ranking uses this model only.
- **Reference model** (shown alongside, never used for decisions):
  `RegimeSwitchingBootstrapSimulator` by default (same family, no CAPE
  discounting). The user can switch the reference to
  `HybridValuationVARSimulator`. No other model options are needed.

### Tax regime

Fixed to `pre_tcja_reversion` (the biggest tax drag). The UI does not
compare tax regimes; the field is an advanced setting at most.

### Lifetime ruin and mortality

Ruin only matters if someone is alive to live through it. A cell's
**lifetime ruin probability** is P(ruined while someone in the household
is alive): each ruined path counts by P(alive) at its ruin age, from the
monthly `ruin_histogram`, so no engine change or re-run is needed. Ruin
after the horizon isn't simulated, so it doesn't count.

**Expected years in ruin** is the timing measure: the expected years lived
after the money runs out, within the horizon, averaged over *all* paths
(0 where it never does). It is the unconditional, retirement version of
expected shortfall. It replaced ES5/ES10 ruin ages, which were
conditional on ruin (averaged over ruined paths only): a cell that ruins
more often but later could rank above a safer one, and on the user's
profile they picked the riskier cell whenever they disagreed with years
in ruin. They were removed from the UI and from the CLI's aggregated
results; the UI derives every ruin-timing measure from
`ruin_histogram`.

Mortality is a Gompertz law per person (`decision.Life`: modal age at
death and dispersion), set per review in the sidebar ("Life expectancy")
and saved in `review.json`, editable after runs exist. One person by
default; a couple uses last-survivor P(alive) with independent lifetimes
and the partner's age difference.

Default calibration (`Life()`: modal age 88, dispersion 13): a
least-squares fit to survival from age 42 in the CDC's 2022 US life table
for males (NVSR vol. 74 no. 2, April 2025), after letting each age's death
rate fall 1% a year from 2022. The 1% is our assumption, standing in for
the SSA cohort tables (not machine-accessible): a period table
understates how long someone alive today will live, which for ruin is the
optimistic direction. Fitted to the table as published, the curve is
modal 83.7 / dispersion 11.7 and reproduces CDC's life expectancy at 42
(78.3 vs 78.0); with the improvement, life expectancy at 42 is 81.9 and
P(alive) is 46% at 85 and 18% at 95.

### Ruin ceiling

- Default **1%**, on **lifetime ruin**. Lifetime ruin runs well below
  ruin by the end of the horizon (about 3-4x for a 42-year-old with a
  horizon to 100): on the user's profile, the cells the earlier 5% ceiling
  on ruin by the horizon selected had ~1% lifetime ruin, so 1% keeps
  roughly the same risk level. 5% lifetime would allow ~14% of paths to
  be broke by 100.
- Adjustable in the UI. Changing it only re-evaluates results on disk.
- A cell **passes** if the decision model's lifetime ruin ≤ ceiling.

### Headline: maximum sustainable spending

The highest spending level at which at least one allocation passes. Report:

- **Bracket** (hard answer): last passing spending level and the first
  failing one, e.g. "95k ✓ / 100k ✗".
- **Interpolated estimate** (hint): take the minimum lifetime ruin across
  allocations at each spending level, and linearly interpolate where it
  crosses the ceiling. Label it as an estimate. A wide gap between the
  bracket and the estimate suggests running a refinement.
- Flag the result when the deciding cell's lifetime ruin is within ~2
  standard errors of the ceiling (the standard error of the mean of each
  path's P(alive at ruin), 0 for unruined paths).

### Ranking passing cells

Lexicographic with tie tolerances. Defaults (adjustable):

| Step | Criterion | Better | Tied if |
|---|---|---|---|
| 0 | Lifetime ruin | lower | within 20% of the ceiling (0.2pp at 1%) |
| 1 | Expected years in ruin | fewer | within that × 10 years (0.02 years at 1%) |
| 2 | P10 return (real) | higher | within 10% relative, on terminal wealth (1 + return) |
| 3 | P50 return (real) | higher | — (final decider; an exact tie goes to lower lifetime ruin, then fewer years in ruin) |

Notes:
- The risk tolerances **scale with the ceiling** (one setting, "ruin
  tie", as a share of it). Fixed tolerances sized for one ceiling break
  at another: at a 1% ceiling a 1pp tie made every passing cell tie on
  ruin, so ranking reduced to "pass, then highest P10" and picked the
  riskiest passing allocation (0.97% lifetime ruin over 0.53% in the test
  run). The ×10 converts a ruin tie into years: a path ruined while
  someone is alive is lived in ruin ~6-9 years on the user's profile, and
  it is the ratio of the original 1pp / 0.1-year defaults.
- A difference within **2 standard errors** of the two cells' estimates
  is always a tie, whatever the tolerance: below that it is simulation
  noise.
- Tested alternatives on 25k-path runs of the user's profile: noise-only
  ties made ranking "minimize risk at any cost" (20/80 with a -63% real
  median); fixed tolerances made it "maximize P10 once under the gate".
  Scaled tolerances picked 40/60-50/50 at every frontier level.
- Lifetime ruin still counts after the gate: a cell near the ceiling
  must not beat a much safer one on timing or returns alone.
- Returns are real (today's dollars) when every compared cell has them,
  else nominal for all (results that predate real returns).
- The P10 tolerance is measured on terminal wealth (1 + return), which is
  always ≥ 0. A relative tolerance on the return itself breaks down near
  zero, where the decision model's P10s often sit.
- Pairwise tolerances aren't transitive (A≈B, B≈C, A≉C), so a plain
  `sort` is ill-defined. Use **anchored selection**: at each step, keep the
  candidates within tolerance of the *best* value among the current
  candidates and drop the rest; the survivor after step 3 is the winner.
  For a full ranking, remove the winner and repeat.
- Show *why* each cell ranks where it does, e.g. "tied on lifetime
  ruin, years in ruin; won on P10 return".
- The main use is ranking the allocations at one spending level (in
  particular at the max sustainable spending), but the same function can
  rank any set of cells.

## Technology decision

**Streamlit**, run locally with `streamlit run app.py`. Plotly for charts.
UI dependencies go in a separate `requirements-ui.txt` so the engine's
pinned `requirements.txt` is unaffected.

Rationale: stays in Python, minimal code, strong fit for single-user data
dashboards and what-if exploration. Rejected alternatives: Dash (more
boilerplate, built for multi-user apps), Panel/Voilà (awkward as the UI
grows), NiceGUI (less data-focused), Textual (doesn't solve visualization).

## Core architectural rule

> **The UI never imports or calls the engine's simulation functions
> directly.** It only launches the CLI as a subprocess and reads the files
> the CLI writes.

Why:
- Streamlit reruns the script on every interaction in a per-session thread.
  A long blocking call there is fragile: refresh, closed tab or a stray
  widget click can lose the run.
- Process pools inside Streamlit break: with `spawn`, workers re-import
  Streamlit's runner rather than our module; with `fork`, forking a
  multithreaded server can deadlock.
- A detached subprocess running the CLI has a normal `__main__`, so
  multiprocessing behaves exactly as it does today, and jobs survive
  browser closes and UI restarts.

The UI may import pure helpers that don't touch simulation code (its own
schema/loading code, the ranking logic). It always launches the CLI with
`--backend numba`.

## Data model: review → runs → cells

```
reviews/                          # gitignored
  <review_id>/                    # e.g. 2026-10-02-q4
    review.json                   # label, date, the shared profile (below)
    runs/
      <config_hash>/
        config.json               # exact CLI input
        status.json               # progress + state, written by the CLI
        results.json              # aggregated results, written on success
        meta.json                 # timestamps, engine git commit, seed, backend
        log.txt                   # CLI stdout/stderr
```

- A **review** is a NAV snapshot at a point in time. It fixes the profile
  shared by all its runs: `initial_nav`, `retirement_age`,
  `years_to_simulate`, tax regime, and the market assumptions (starting
  and long-run CAPE, real earnings growth, net buyback yield, dividend
  yield). The UI enforces this; runs with a different profile belong in a
  different review. The assumptions are on the review, not the run,
  because cells merge across a review's runs; to compare assumptions,
  create one review per scenario and compare them in History. Reviews
  created before the assumption fields existed load with the engine's
  defaults, which are what they ran under.
- A **run** is one CLI invocation (one sweep). Its folder is keyed by a hash
  of the canonicalized `config.json`, excluding `master_seed`, so re-running
  the same sweep finds the existing run. (`workers` is included: chunking,
  and so every chunk's seed, depends on it.)
- The UI always sets `master_seed` (random per run, recorded in
  `config.json` and `meta.json`) so every run is reproducible.
- **Cells merge across runs within a review.** A cell's key is
  `(model, spending, equity, tax_regime)`. If several runs computed the same
  cell, the one with the most paths wins (ties: the newest). The explorer
  always shows the merged view, so the frontier fills in as you refine.
- **Dedup:** if a run folder with the same hash already succeeded, reuse
  it; if it is running, show its progress instead of relaunching. A failed
  or cancelled run of the same sweep is replaced (with a new seed).
- Reviews live in `reviews/` (gitignored), or in `$PLANNER_REVIEWS_DIR`.

### Job handling

Runs are much shorter now, so job tracking is deliberately light:

- **Launch:** UI writes `config.json`, then starts
  `monte_carlo.py --job-dir <run_dir> --backend numba` with
  `subprocess.Popen(..., start_new_session=True)`, stdout/stderr to
  `log.txt`.
- **Progress:** progress bar from `status.json`, polled with
  `st.fragment(run_every=2)` while a run is active. Shown inline on the
  explorer page; there is no separate monitor page.
- **Cancel:** kill the process group by PID (SIGTERM) and mark the run
  `cancelled`.
- **Stale detection:** `state == "running"` but the PID is gone → failed.
  A run with `config.json` but no `status.json` is "starting" for 60 s,
  then failed.

`status.json`:

```json
{
  "state": "running",
  "pid": 12345,
  "cells_done": 21,
  "cells_total": 54,
  "started_at": "2026-10-02T19:02:11",
  "updated_at": "2026-10-02T19:02:40",
  "error": null
}
```

`state` is one of `running`, `succeeded`, `failed` (with `error`), or
`cancelled` (written by the UI). `cells_done`/`cells_total` count
portfolios, i.e. aggregated result entries.

## Required CLI / engine changes

All implemented (`job_files.py` holds the shared file protocol).

1. **`-j/--job-dir <dir>`:** reads `<dir>/config.json`; writes `status.json`,
   `results.json` and `meta.json` into the folder. With `--job-dir`, the
   `-c/-r/-o` flags are not required. The raw output is skipped unless
   `-r` is given. Without `--job-dir`, behaviour is unchanged. All parsing
   stays inside `parse_args()` (no argv parameters).
2. **Status reporting:** write `status.json` at start, as cells complete,
   on success and on failure (with the error message).
3. **Atomic writes:** write to a temp file in the same folder, then
   `os.replace()`, so the UI never reads a half-written file.
4. **Ruin histogram in aggregated results:** already present in the raw
   output; copy it into each aggregated cell entry. It enables the survival
   curve.
5. **Per-year percentile bands:** both backends keep each path's NAV at
   every year-end (`fast_ladder.annual_snapshot_months`) and reduce them to
   per-year P5/P10/P25/P50 as each cell is collected
   (`monte_carlo.nav_bands`), so only those arrays reach the raw and
   aggregated output. No kernel change was needed: the batch wrappers
   already had every path's monthly NAV. The bands match exactly across
   backends (to rounding for HybridValuationVARSimulator), and the 50k-path
   runtime is unchanged.

## Results data contract (`results.json`)

Keep the existing aggregated shape (`initial_nav`, `years_to_simulate`,
`total_paths`, `retirement_age`, `results: {model: [cell, ...]}`). Per cell:

- `spending`, `allocation`, `tax_regime` (existing; add a numeric `equity`
  so the UI doesn't parse the `"60-40"` string)
- `ruin_path_count`. Ruin timing is only in `ruin_histogram`; the
  statistics over ruined paths (`ruin_month_min`, `ruin_month_median`,
  `ruin_month_es5`, `ruin_month_es10`) were removed. Older results still
  carry them and the UI ignores them.
- `p5_return`, `p10_return`, `p25_return`, `p50_return` (existing)
- `ruin_histogram`: monthly counts over the horizon (new)
- `nav_bands: {years: [0, 1, ...], p5: [], p10: [], p25: [], p50: []}`:
  **nominal** NAV per year since retirement, year 0 = initial NAV, ruined
  paths counted as 0 (`null` for results produced before it existed)
- `real_nav_bands` (same shape) and `p5/p10/p25/p50_real_return`: the
  same in today's dollars, each path deflated by its own simulated CPI.
  `p*_return` are nominal.
- `ruin_prob_by_age: {"75": p, ...}`: unconditional P(ruin before age).
  The UI computes the same from `ruin_histogram`
  (`decision.ruin_prob_before`), so it also works for older results.
- Top level: `assumptions: {dividend_yield, simulator_params}`.

Ruin rate = `ruin_path_count / total_paths`. Survival curve:
`P(solvent at month m) = 1 - cumsum(ruin_histogram)[m] / total_paths`.

## Design principles

- **Spending first.** Yearly spending is the most important dimension and
  the main axis of every primary view.
- **Downside first.** Show only the median and below; upper percentiles
  aren't displayed (good years are captured by recalibrating from actual
  NAV at the next review). Exception: P50 is used as the final tie-breaker.
- **Decision vs reference.** The decision model drives every headline and
  ranking; the reference model is shown for context, visually secondary.
- **Show the reasoning.** Rankings explain themselves; numbers near a
  threshold carry a noise flag.
- **De-emphasize minimum ruin age.** It is set by a single path. Show it as
  a detail only.

## Pages

### 1. Explorer (main page)

- **Review selector** (latest by default) and **sweep inputs**: spending
  floor/ceil/step, equity floor/ceil/step, paths. Profile fields come from
  the review. "Run" launches a run (or reuses a cached one); progress shows
  inline.
- **Display:** today's dollars (default) or nominal, for every return,
  the fan chart and History. Cells from runs that predate real figures
  fall back to nominal, with a note.
- **Controls:** lifetime ruin ceiling (default 1%), life expectancy
  (per review), reference model
  (`RegimeSwitchingBootstrapSimulator` | `HybridValuationVARSimulator`),
  tie tolerances (collapsed by default).
- **Headline:** maximum sustainable spending, as bracket + interpolated
  estimate, with the winning allocation at that level.
- **Lifetime ruin vs spending chart:** one line per allocation for the
  decision model, ceiling as a horizontal line, merged across all runs in the
  review. Reference model available as a toggle or a secondary panel.
- **Ranked table** at a selected spending level (defaults to the max
  sustainable spending): allocation, lifetime ruin (±95%), years in ruin,
  P(ruin before 75/85/95) for the ages inside the horizon, ruin by the
  end of the horizon, P10, P25, P50, paths, and the reference model's
  lifetime ruin for context; row shade explains the rank.
- **"Refine around frontier"**: proposes the next run, editable before
  launching (`planner.decision.propose_refinement`):
  - frontier bracketed → spending from the last passing to the first
    failing level, split into ~4 equal steps (multiples of $500);
  - every level passes → extend upward; none passes → extend downward;
  - equity spans the top two allocations at the frontier, padded by half
    the current step, at the next round step (10%, 5%, 2%, 1%).

### 2. Cell detail

For a selected (spending, allocation):
- KPI cards: lifetime ruin (headline), years in ruin, ruin by the end
  of the horizon, P10 and P50 return, P(ruin before 75/85/95) inside the
  horizon; minimum ruin age as a small detail. Decision model
  next to the reference model.
- Survival curves show P(alive) alongside, instead of a ceiling line.
- Survival curves: decision and reference on the same axes, the ceiling
  marked.
- Ruin-age histogram.
- Fan chart: per-year P5–P10, P10–P25, P25–P50 bands and the median,
  decision and reference model in tabs, optional log scale.

### 3. History

- Timeline across reviews: max sustainable spending, plus lifetime ruin
  and years in ruin of the chosen cell, each review under its own
  household settings.
- Side-by-side diff of two reviews: profile changes (including the
  market assumptions) and result changes.
- Survival curve overlay of the chosen cell, current vs previous review.
  An upward shift after a good stretch is the signal to consider raising
  spending.

## Streamlit implementation notes

- Multi-page layout: `app.py` with `st.navigation`, pages in `views/`. Not
  `pages/`: Streamlit's legacy auto-discovery of a `pages/` folder hijacks a
  first request straight to a page URL before `st.navigation` is registered.
- `.streamlit/config.toml` binds the server to localhost: no auth, so it
  must not listen on the network.
- Markdown text (captions, labels, buttons) must escape `$` (`ui.esc`,
  `ui.md_money`), or Streamlit renders `$...$` as LaTeX.
- Results are cached on disk in the run folders; use `st.cache_data` only
  for loading/parsing JSON, keyed on path + mtime.
- Keep the selected review, cell and controls in `st.session_state`.
- Keep the decision logic (merge, gate, max sustainable spending, ranking)
  in a plain Python module with no Streamlit imports, so it is unit-tested
  like the rest of the repo.
- Nothing in the UI blocks for more than a moment; long work always happens
  in the subprocess.

## Implementation order

All seven steps are implemented (October 2026).

1. **CLI:** `--job-dir`, `status.json`, atomic writes, `meta.json`,
   `ruin_histogram` and numeric `equity` in aggregated cells.
2. **Decision logic module + tests:** load and merge runs, gate, max
   sustainable spending (bracket + interpolation), anchored ranking.
3. **Minimal explorer:** create/load a review, launch a run, inline
   progress, headline, P(ruin)-vs-spending chart, ranked table.
4. **Cell detail:** KPI cards, survival curves, histogram, reference-model
   switch.
5. **Refine around frontier.**
6. **Per-year bands** in both backends, then the fan chart.
7. **History page.**

## Measured

On the user's machine with `--backend numba`, 50,000 paths × 12 cells
(`owning.json`) take ~7 s end to end, so a wide sweep of ~100 cells is
about a minute.
