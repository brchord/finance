"""
planner/decision.py

The UI's decision logic, independent of Streamlit: turning results.json
files into cells, merging cells across the runs of a review, gating them on
a ruin-rate ceiling, finding the maximum sustainable spending and ranking
cells. See doc/plans/UI Design.md, "Decision model".
"""

import math
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DECISION_MODEL = "RegimeSwitchingValuationVARSimulator"
REFERENCE_MODELS = ("RegimeSwitchingBootstrapSimulator",
                    "HybridValuationVARSimulator")
DEFAULT_TAX_REGIME = "pre_tcja_reversion"

CellKey = Tuple[str, float, float, str]


@dataclass(frozen=True)
class Cell:
    """
    One simulated (model, spending, equity, tax regime) combination.
    Ruin timings are in months since retirement; returns are total real
    returns over the horizon (terminal NAV / initial NAV - 1).
    """
    model: str
    spending: float
    equity: float
    tax_regime: str
    total_paths: int
    ruin_count: int
    ruin_month_min: Optional[float]
    ruin_month_es5: Optional[float]
    ruin_month_es10: Optional[float]
    p5_return: float
    p10_return: float
    p25_return: float
    p50_return: float
    ruin_histogram: Tuple[int, ...] = field(repr=False)
    # Per-year NAV percentiles ({"years", "p5", "p10", "p25", "p50"}), or
    # None for results produced before the CLI wrote them.
    nav_bands: Optional[dict] = field(default=None, repr=False,
                                      compare=False, hash=False)
    run_id: str = ""
    finished_at: str = ""

    @property
    def key(self) -> CellKey:
        return (self.model, self.spending, self.equity, self.tax_regime)

    @property
    def ruin_rate(self) -> float:
        return self.ruin_count / self.total_paths

    @property
    def ruin_rate_se(self) -> float:
        "Binomial standard error of ruin_rate."
        p = self.ruin_rate
        return math.sqrt(p * (1.0 - p) / self.total_paths)

    @property
    def allocation(self) -> str:
        "Equity/fixed-income split, e.g. '60/40'."
        equity_pct = round(self.equity * 100)
        return f"{equity_pct}/{100 - equity_pct}"

    def survival(self) -> List[float]:
        """
        P(still solvent at the end of month m) for each month m of the
        horizon.
        """
        return [1.0 - ruined / self.total_paths
                for ruined in accumulate(self.ruin_histogram)]


def month_to_age(month: Optional[float],
                 retirement_age: float) -> Optional[float]:
    "Age at a month index counted from retirement."
    return None if month is None else retirement_age + month / 12.0


def cells_from_results(results: dict, run_id: str = "",
                       finished_at: str = "") -> List[Cell]:
    "Cells of a results.json document (MonteCarloCLI.agg_results)."
    total_paths = results["total_paths"]
    return [
        Cell(model=model,
             spending=float(e["spending"]),
             equity=float(e["equity"]),
             tax_regime=e["tax_regime"],
             total_paths=total_paths,
             ruin_count=e["ruin_path_count"],
             ruin_month_min=e["ruin_month_min"],
             ruin_month_es5=e["ruin_month_es5"],
             ruin_month_es10=e["ruin_month_es10"],
             p5_return=e["p5_return"],
             p10_return=e["p10_return"],
             p25_return=e["p25_return"],
             p50_return=e["p50_return"],
             ruin_histogram=tuple(e["ruin_histogram"]),
             nav_bands=e.get("nav_bands"),
             run_id=run_id,
             finished_at=finished_at)
        for model, entries in results["results"].items()
        for e in entries
    ]


def merge_cells(cells: Iterable[Cell]) -> Dict[CellKey, Cell]:
    """
    One cell per key across runs: the one with the most paths, and among
    equals the most recently finished.
    """
    merged: Dict[CellKey, Cell] = {}
    for cell in cells:
        current = merged.get(cell.key)
        if current is None or ((cell.total_paths, cell.finished_at)
                               > (current.total_paths, current.finished_at)):
            merged[cell.key] = cell
    return merged


@dataclass(frozen=True)
class Criteria:
    """
    Ruin ceiling and ranking tie tolerances (doc/plans/UI Design.md,
    "Ranking passing cells").
    """
    ruin_ceiling: float = 0.05
    ruin_tolerance: float = 0.01      # absolute, as a fraction (1pp)
    es10_tolerance_years: float = 1.0
    p10_tolerance: float = 0.10       # relative, on terminal wealth

    def passes(self, cell: Cell) -> bool:
        return cell.ruin_rate <= self.ruin_ceiling

    def near_ceiling(self, cell: Cell) -> bool:
        "Ruin rate within 2 standard errors of the ceiling."
        return abs(cell.ruin_rate - self.ruin_ceiling) <= 2 * cell.ruin_rate_se


@dataclass(frozen=True)
class Ranked:
    cell: Cell
    rank: int            # 1-based
    passes: bool
    reason: str


def _es10_years(cell: Cell) -> float:
    # No ruined path at all is the best possible ES10.
    if cell.ruin_month_es10 is None:
        return math.inf
    return cell.ruin_month_es10 / 12.0


def _select_best(candidates: List[Cell],
                 criteria: Criteria) -> Tuple[Cell, str]:
    """
    Anchored lexicographic selection: at each step keep only the
    candidates within tolerance of the best value among those remaining.
    The first step that leaves a single candidate decides; if none does,
    the highest P50 return wins. Returns the winner and an explanation.
    """
    steps = [
        ("ruin rate",
         lambda c: -c.ruin_rate,
         lambda best: best - criteria.ruin_tolerance),
        ("ES10 age",
         _es10_years,
         lambda best: best - criteria.es10_tolerance_years),
        # Terminal wealth multiple (1 + return) is >= 0, so a relative
        # tolerance stays meaningful when the return itself is near zero.
        ("P10 return",
         lambda c: 1.0 + c.p10_return,
         lambda best: best * (1.0 - criteria.p10_tolerance)),
    ]
    tied_on: List[str] = []
    for name, value, threshold in steps:
        best = max(value(c) for c in candidates)
        cutoff = threshold(best)
        candidates = [c for c in candidates
                      if value(c) >= cutoff or value(c) == best]
        if len(candidates) == 1:
            return candidates[0], _explain(tied_on, name)
        tied_on.append(name)
    return (max(candidates, key=lambda c: c.p50_return),
            _explain(tied_on, "P50 return"))


def _explain(tied_on: List[str], decided_by: str) -> str:
    if tied_on:
        return f"tied on {', '.join(tied_on)}; won on {decided_by}"
    return f"won on {decided_by}"


def rank(cells: Sequence[Cell], criteria: Criteria) -> List[Ranked]:
    """
    Ranks cells: passing cells first, ordered by repeated anchored
    selection (pick the best, remove it, repeat), then failing cells by
    ascending ruin rate.
    """
    passing = [c for c in cells if criteria.passes(c)]
    failing = sorted((c for c in cells if not criteria.passes(c)),
                     key=lambda c: c.ruin_rate)
    ranked: List[Ranked] = []
    while passing:
        if len(passing) == 1:
            winner, reason = passing[0], "last passing cell"
        else:
            winner, reason = _select_best(passing, criteria)
        ranked.append(Ranked(winner, len(ranked) + 1, True, reason))
        passing.remove(winner)
    for cell in failing:
        ranked.append(Ranked(cell, len(ranked) + 1, False,
                             "ruin rate above ceiling"))
    return ranked


@dataclass(frozen=True)
class Frontier:
    """
    Maximum sustainable spending for one model. best_spending is the
    highest simulated spending level where some allocation passes;
    next_failing_spending is the next simulated level above it (None if
    every level above passes or none was simulated). estimate interpolates
    the ceiling crossing between the two.
    """
    best_spending: Optional[float]
    best_cell: Optional[Ranked]
    next_failing_spending: Optional[float]
    estimate: Optional[float]
    near_ceiling: bool
    # A failing level below best_spending: the frontier isn't monotonic,
    # which at these path counts means noise around the ceiling.
    non_monotonic: bool
    # Min ruin rate across allocations, per spending level.
    min_ruin_by_spending: Dict[float, float]


def frontier(cells: Sequence[Cell], criteria: Criteria) -> Frontier:
    "Maximum sustainable spending among cells (of one model)."
    by_spending: Dict[float, List[Cell]] = {}
    for cell in cells:
        by_spending.setdefault(cell.spending, []).append(cell)
    min_ruin = {s: min(c.ruin_rate for c in group)
                for s, group in sorted(by_spending.items())}
    passing_levels = [s for s, r in min_ruin.items()
                      if r <= criteria.ruin_ceiling]
    if not passing_levels:
        return Frontier(None, None, min(min_ruin, default=None), None,
                        False, False, min_ruin)

    best = max(passing_levels)
    above = [s for s in min_ruin if s > best]
    next_fail = min(above) if above else None
    best_cell = rank(by_spending[best], criteria)[0]

    estimate = None
    if next_fail is not None:
        r1, r2 = min_ruin[best], min_ruin[next_fail]
        estimate = best + ((criteria.ruin_ceiling - r1) / (r2 - r1)
                           * (next_fail - best))

    return Frontier(
        best_spending=best,
        best_cell=best_cell,
        next_failing_spending=next_fail,
        estimate=estimate,
        near_ceiling=criteria.near_ceiling(best_cell.cell),
        non_monotonic=any(s < best and r > criteria.ruin_ceiling
                          for s, r in min_ruin.items()),
        min_ruin_by_spending=min_ruin)


@dataclass(frozen=True)
class SweepProposal:
    "Spending and equity ranges for a refinement run."
    spending_floor: float
    spending_ceil: float
    spending_step: float
    equity_floor: float
    equity_ceil: float
    equity_step: float
    rationale: str


SPENDING_RESOLUTION = 500.0
EQUITY_STEPS_PCT = (10, 5, 2, 1)


def _refined_spending_step(bracket: float) -> Optional[float]:
    """
    A step dividing bracket into 4 (else 5, else 2) equal parts that is a
    multiple of SPENDING_RESOLUTION, or None if the bracket can't be
    split any finer.
    """
    for parts in (4, 5, 2):
        step = bracket / parts
        if step >= SPENDING_RESOLUTION and step % SPENDING_RESOLUTION == 0:
            return step
    return None


def propose_refinement(cells: Sequence[Cell], criteria: Criteria,
                       spending_step: float,
                       equity_step: float) -> Optional[SweepProposal]:
    """
    Next sweep zooming in on the frontier of cells (of one model), given
    the finest steps used so far. None if there are no cells, or the
    frontier is already bracketed at the finest spending resolution.

    - Frontier bracketed: spending from the last passing to the first
      failing level, split into about 4 equal steps.
    - Every level passes: extend upward at the same step.
    - No level passes: extend downward at the same step.

    Equity spans the top two passing allocations at the frontier, padded
    by half the current step on each side, at the next round step below
    half the current one (10%, 5%, 2%, 1%).
    """
    if not cells:
        return None
    f = frontier(cells, criteria)
    levels = sorted(f.min_ruin_by_spending)

    if f.best_spending is None:
        ceil = levels[0] - spending_step
        floor = max(spending_step, ceil - 4 * spending_step)
        step = spending_step
        rationale = ("No spending level passes the ceiling: extending the "
                     "sweep downward.")
    elif f.next_failing_spending is None:
        floor = f.best_spending + spending_step
        ceil = floor + 4 * spending_step
        step = spending_step
        rationale = ("Every simulated spending level passes: extending "
                     "the sweep upward.")
    else:
        floor, ceil = f.best_spending, f.next_failing_spending
        refined = _refined_spending_step(ceil - floor)
        if refined is None:
            return None
        step = refined
        rationale = (f"Frontier between ${floor:,.0f} and ${ceil:,.0f}: "
                     f"refining at ${step:,.0f} steps.")

    reference_level = (f.best_spending if f.best_spending is not None
                       else levels[0])
    top = [r.cell.equity
           for r in rank([c for c in cells
                          if c.spending == reference_level], criteria)[:2]]

    # Integer percentages, so bounds land exactly on the step grid.
    current_pct = round(equity_step * 100)
    step_pct = next((s for s in EQUITY_STEPS_PCT if 2 * s <= current_pct), 1)
    low = round(min(top) * 100) - current_pct / 2
    high = round(max(top) * 100) + current_pct / 2
    floor_pct = max(0, math.floor(low / step_pct) * step_pct)
    ceil_pct = min(100, math.ceil(high / step_pct) * step_pct)

    return SweepProposal(
        spending_floor=floor, spending_ceil=ceil, spending_step=step,
        equity_floor=floor_pct / 100, equity_ceil=ceil_pct / 100,
        equity_step=step_pct / 100, rationale=rationale)
