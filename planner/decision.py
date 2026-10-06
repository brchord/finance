"""
planner/decision.py

The UI's decision logic, independent of Streamlit: turning results.json
files into cells, merging cells across the runs of a review, gating them on
a ruin-rate ceiling, finding the maximum sustainable spending and ranking
cells. See doc/plans/UI Design.md, "Decision model".
"""

import functools
import math
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DECISION_MODEL = "RegimeSwitchingValuationVARSimulator"
REFERENCE_MODELS = ("RegimeSwitchingBootstrapSimulator",
                    "HybridValuationVARSimulator")
DEFAULT_TAX_REGIME = "pre_tcja_reversion"
# Ages for P(ruin before age), as in the CLI's default ruin_ages.
RUIN_AGES = (75, 85, 95)

CellKey = Tuple[str, float, float, str]


@dataclass(frozen=True)
class Cell:
    """
    One simulated (model, spending, equity, tax regime) combination.
    Ruin timings are in months since retirement. p*_return are total
    NOMINAL returns over the horizon (terminal NAV / initial NAV - 1, with
    terminal NAV in future dollars); p*_real_return deflate each path's
    terminal NAV by its own simulated price level first. The real ones,
    the real bands and ruin_prob_by_age are None for results produced
    before the CLI wrote them.
    """
    model: str
    spending: float
    equity: float
    tax_regime: str
    total_paths: int
    ruin_count: int
    ruin_month_min: Optional[float]
    p5_return: float
    p10_return: float
    p25_return: float
    p50_return: float
    ruin_histogram: Tuple[int, ...] = field(repr=False)
    # Per-year NAV percentiles ({"years", "p5", "p10", "p25", "p50"}), or
    # None for results produced before the CLI wrote them.
    nav_bands: Optional[dict] = field(default=None, repr=False,
                                      compare=False, hash=False)
    # Same percentiles in today's dollars.
    real_nav_bands: Optional[dict] = field(default=None, repr=False,
                                           compare=False, hash=False)
    p5_real_return: Optional[float] = None
    p10_real_return: Optional[float] = None
    p25_real_return: Optional[float] = None
    p50_real_return: Optional[float] = None
    # Unconditional P(ruin before age), keyed by age as a string ("85").
    ruin_prob_by_age: Optional[dict] = field(default=None, repr=False,
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

    def pct_return(self, q: int, real: bool) -> Optional[float]:
        """
        The q-th percentile (5, 10, 25 or 50) total return, real or
        nominal. None when real is asked of a result that predates real
        returns.
        """
        return getattr(self, f"p{q}_{'real_' if real else ''}return")

    def bands(self, real: bool) -> Optional[dict]:
        "real_nav_bands or nav_bands; None when the run predates them."
        return self.real_nav_bands if real else self.nav_bands

    def survival(self) -> List[float]:
        """
        P(still solvent at the end of month m) for each month m of the
        horizon.
        """
        return [1.0 - ruined / self.total_paths
                for ruined in accumulate(self.ruin_histogram)]


def ruin_prob_before(cell: Cell, age: float,
                     retirement_age: float) -> float:
    """
    Unconditional P(ruin before age): the share of all paths ruined
    before it. Computed from the ruin histogram, so it also works for
    results that predate the CLI's ruin_prob_by_age, and with the same
    convention: a ruin in month m happens at age retirement_age + m / 12
    and counts iff that is before age.
    """
    months = math.ceil(round((age - retirement_age) * 12.0, 9))
    n = min(max(months, 0), len(cell.ruin_histogram))
    return sum(cell.ruin_histogram[:n]) / cell.total_paths


def ruin_ages(retirement_age: float, terminal_age: float) -> List[float]:
    """
    The RUIN_AGES inside the horizon. Ages after it would all repeat the
    overall ruin rate.
    """
    return [a for a in RUIN_AGES if retirement_age < a <= terminal_age]


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
             p5_return=e["p5_return"],
             p10_return=e["p10_return"],
             p25_return=e["p25_return"],
             p50_return=e["p50_return"],
             ruin_histogram=tuple(e["ruin_histogram"]),
             nav_bands=e.get("nav_bands"),
             real_nav_bands=e.get("real_nav_bands"),
             p5_real_return=e.get("p5_real_return"),
             p10_real_return=e.get("p10_real_return"),
             p25_real_return=e.get("p25_real_return"),
             p50_real_return=e.get("p50_real_return"),
             ruin_prob_by_age=e.get("ruin_prob_by_age"),
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
class Life:
    """
    One person's mortality, as a Gompertz law: the force of mortality
    grows exponentially with age, with modal_age the most likely age at
    death and dispersion (years) how spread out deaths are around it.

    The defaults fit the CDC's 2022 US life table for males (NVSR vol. 74
    no. 2) for someone alive at 42, after letting death rates fall 1% a
    year from 2022 (the table is a period table: it understates how long
    people alive today will live). Without the improvement the fit is
    modal_age 83.7, dispersion 11.7. See doc/plans/UI Design.md.
    """
    modal_age: float = 88.0
    dispersion: float = 13.0

    def survival(self, age: float, from_age: float) -> float:
        "P(alive at age | alive at from_age)."
        if age <= from_age:
            return 1.0
        b = self.dispersion
        return math.exp(math.exp((from_age - self.modal_age) / b)
                        * (1.0 - math.exp((age - from_age) / b)))

    def life_expectancy(self, from_age: float) -> float:
        "Expected age at death given alive at from_age."
        step, age, total = 1.0 / 12.0, from_age, 0.0
        while True:
            alive = self.survival(age + step / 2, from_age)
            if alive < 1e-9:
                return from_age + total
            total += alive * step
            age += step


@dataclass(frozen=True)
class Household:
    """
    Who the money has to last for. Ruin matters while anyone is alive:
    for a couple, P(alive) is the last-survivor probability, assuming
    independent lifetimes. partner_age_offset is the partner's age minus
    the planner's, so the partner is retirement_age + offset when the
    simulation starts.
    """
    person: Life = Life()
    partner: Optional[Life] = None
    partner_age_offset: float = 0.0

    def p_alive(self, age: float, from_age: float) -> float:
        "P(someone in the household alive when the planner is `age`)."
        alive = self.person.survival(age, from_age)
        if self.partner is None:
            return alive
        partner = self.partner.survival(age + self.partner_age_offset,
                                        from_age + self.partner_age_offset)
        return 1.0 - (1.0 - alive) * (1.0 - partner)


@functools.lru_cache(maxsize=64)
def _ruin_weights(household: Household, retirement_age: float,
                  months: int) -> Tuple[Tuple[float, ...],
                                        Tuple[float, ...]]:
    """
    Per ruin month m (ruin at age retirement_age + m / 12, the
    ruin_histogram convention): P(someone alive then), and the expected
    years someone is alive from then to the end of the horizon.
    """
    alive_at = [household.p_alive(retirement_age + m / 12.0,
                                  retirement_age) for m in range(months)]
    mid = [household.p_alive(retirement_age + (m + 0.5) / 12.0,
                             retirement_age) / 12.0 for m in range(months)]
    years_after = list(accumulate(reversed(mid)))[::-1]
    return tuple(alive_at), tuple(years_after)


# Converts a lifetime ruin tie into an expected-years-in-ruin tie: a path
# ruined while someone is alive is lived in ruin for roughly this long
# (6-9 years on the user's profile). Also the ratio of the original
# defaults (1pp and 0.1 years).
YEARS_PER_RUIN = 10.0
# Differences within this many standard errors are noise, whatever the
# tolerance says.
NOISE_Z = 2.0


@dataclass(frozen=True)
class Criteria:
    """
    The gate and the ranking (doc/plans/UI Design.md, "Decision model").
    A cell passes if its lifetime ruin probability, P(ruined while
    someone in the household is alive), is at most ruin_ceiling. Ruin
    after the horizon isn't simulated, so it doesn't count.

    Ranking ties scale with the ceiling: two cells tie on lifetime ruin
    within tie_share of the ceiling, and on years in ruin within that
    times YEARS_PER_RUIN, or within NOISE_Z standard errors of their
    difference if that is larger.
    """
    ruin_ceiling: float = 0.01
    tie_share: float = 0.2
    p10_tolerance: float = 0.10       # relative, on terminal wealth
    retirement_age: float = 65.0
    household: Household = Household()

    @property
    def ruin_tolerance(self) -> float:
        "Lifetime ruin tie, as a fraction (0.002 = 0.2pp at a 1% ceiling)."
        return self.tie_share * self.ruin_ceiling

    @property
    def years_in_ruin_tolerance(self) -> float:
        "Years-in-ruin tie, in years."
        return self.ruin_tolerance * YEARS_PER_RUIN

    def _weights(self, cell: Cell):
        return _ruin_weights(self.household, self.retirement_age,
                             len(cell.ruin_histogram))

    def ruin(self, cell: Cell) -> float:
        "Lifetime ruin probability: P(ruin while someone is alive)."
        return _ruin_stats(cell, self)[0]

    def ruin_se(self, cell: Cell) -> float:
        "Standard error of ruin() from the number of paths."
        return _ruin_stats(cell, self)[1]

    def years_in_ruin(self, cell: Cell) -> float:
        """
        Expected years lived after ruin, within the horizon, averaged over
        all paths (0 for paths that aren't ruined): an unconditional
        expected shortfall, in years.
        """
        return _ruin_stats(cell, self)[2]

    def years_in_ruin_se(self, cell: Cell) -> float:
        "Standard error of years_in_ruin() from the number of paths."
        return _ruin_stats(cell, self)[3]

    def passes(self, cell: Cell) -> bool:
        return self.ruin(cell) <= self.ruin_ceiling

    def near_ceiling(self, cell: Cell) -> bool:
        "Lifetime ruin within 2 standard errors of the ceiling."
        return abs(self.ruin(cell) - self.ruin_ceiling) <= (
            2 * self.ruin_se(cell))


def _mean_and_se(histogram: Sequence[float], per_path: Sequence[float],
                 n: int) -> Tuple[float, float]:
    """
    Mean over n paths of a per-path value that is per_path[m] for a path
    ruined in month m (histogram[m] of them) and 0 otherwise, and its
    standard error.
    """
    mean = sum(h * w for h, w in zip(histogram, per_path)) / n
    second = sum(h * w * w for h, w in zip(histogram, per_path)) / n
    return mean, math.sqrt(max(second - mean * mean, 0.0) / n)


@functools.lru_cache(maxsize=8192)
def _ruin_stats(cell: Cell,
                criteria: Criteria) -> Tuple[float, float, float, float]:
    """
    (lifetime ruin, its SE, years in ruin, its SE). Per path, lifetime
    ruin is P(alive at the ruin age) and years in ruin the expected years
    alive from then to the end of the horizon (both 0 if never ruined).
    """
    alive, after = criteria._weights(cell)
    ruin = _mean_and_se(cell.ruin_histogram, alive, cell.total_paths)
    years = _mean_and_se(cell.ruin_histogram, after, cell.total_paths)
    return ruin + years


@dataclass(frozen=True)
class Ranked:
    cell: Cell
    rank: int            # 1-based
    passes: bool
    reason: str
    # The ranking step that separated this cell from the rest (one of
    # RANKING_STEPS); every earlier step was a tie. "—" for the last
    # passing cell, "ceiling" for failing ones.
    decided_by: str


RANKING_STEPS = ("lifetime ruin", "years in ruin", "P10 return",
                 "P50 return")


def uses_real_returns(cells: Iterable[Cell]) -> bool:
    """
    Whether cells can be compared on real returns: all of them have them.
    Otherwise (results that predate them) nominal returns are used.
    """
    return all(c.p10_real_return is not None for c in cells)


def _select_best(candidates: List[Cell], criteria: Criteria,
                 real: bool) -> Tuple[Cell, str, str]:
    """
    Anchored lexicographic selection: at each step keep only the
    candidates within tolerance of the best value among those remaining.
    The first step that leaves a single candidate decides; if none does,
    the highest P50 return wins. Returns the winner, an explanation and
    the deciding step's name.
    """
    def ret(c: Cell, q: int) -> float:
        value = c.pct_return(q, real)
        assert value is not None
        return value

    # Lower is better; tied with the best if within the tolerance or the
    # noise of the difference, whichever is larger.
    risk_steps = [
        ("lifetime ruin", criteria.ruin, criteria.ruin_se,
         criteria.ruin_tolerance),
        ("years in ruin", criteria.years_in_ruin, criteria.years_in_ruin_se,
         criteria.years_in_ruin_tolerance),
    ]
    tied_on: List[str] = []
    for name, value, se, tolerance in risk_steps:
        best = min(candidates, key=value)
        candidates = [
            c for c in candidates
            if value(c) - value(best) <= max(
                tolerance, NOISE_Z * math.hypot(se(c), se(best)))]
        if len(candidates) == 1:
            return candidates[0], _explain(tied_on, name), name
        tied_on.append(name)

    # Terminal wealth multiple (1 + return) is >= 0, so a relative
    # tolerance stays meaningful when the return itself is near zero.
    best_p10 = max(1.0 + ret(c, 10) for c in candidates)
    cutoff = best_p10 * (1.0 - criteria.p10_tolerance)
    candidates = [c for c in candidates if 1.0 + ret(c, 10) >= cutoff]
    if len(candidates) == 1:
        return candidates[0], _explain(tied_on, "P10 return"), "P10 return"
    tied_on.append("P10 return")
    # An exact tie on P50 goes to the safer cell, not to list order.
    return (max(candidates, key=lambda c: (ret(c, 50), -criteria.ruin(c),
                                           -criteria.years_in_ruin(c))),
            _explain(tied_on, "P50 return"), "P50 return")


def _explain(tied_on: List[str], decided_by: str) -> str:
    if tied_on:
        return f"tied on {', '.join(tied_on)}; won on {decided_by}"
    return f"won on {decided_by}"


def rank(cells: Sequence[Cell], criteria: Criteria) -> List[Ranked]:
    """
    Ranks cells: passing cells first, ordered by repeated anchored
    selection (pick the best, remove it, repeat), then failing cells by
    ascending lifetime ruin. Returns are real when every cell has them
    (uses_real_returns), else nominal.
    """
    real = uses_real_returns(cells)
    passing = [c for c in cells if criteria.passes(c)]
    failing = sorted((c for c in cells if not criteria.passes(c)),
                     key=criteria.ruin)
    ranked: List[Ranked] = []
    while passing:
        if len(passing) == 1:
            winner, reason, decided_by = (
                passing[0], "last passing cell", "—")
        else:
            winner, reason, decided_by = _select_best(passing, criteria,
                                                      real)
        ranked.append(Ranked(winner, len(ranked) + 1, True, reason,
                             decided_by))
        passing.remove(winner)
    for cell in failing:
        ranked.append(Ranked(cell, len(ranked) + 1, False,
                             "lifetime ruin above ceiling", "ceiling"))
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
    # Min lifetime ruin across allocations, per spending level.
    min_ruin_by_spending: Dict[float, float]


def frontier(cells: Sequence[Cell], criteria: Criteria) -> Frontier:
    "Maximum sustainable spending among cells (of one model)."
    by_spending: Dict[float, List[Cell]] = {}
    for cell in cells:
        by_spending.setdefault(cell.spending, []).append(cell)
    min_ruin = {s: min(criteria.ruin(c) for c in group)
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
