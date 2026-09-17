"""
tax_models/regimes.py

Federal tax-bracket data structures and named "regime scenarios" describing
how bracket structure evolves over a multi-decade simulation horizon.

Design note: tax regime is treated as an explicit SWEEP AXIS (like
equity_allocation or yearly_spending in MCConfig), not as a calibrated
stochastic process. A named scenario is a deterministic RULE for how
today's law evolves given a path's own simulated CPI trajectory -- it is
not a probability-weighted forecast of which rule is "true". Run each
scenario as its own MC batch and compare ruin/spending-safety tables
across them, the same way the CLI already sweeps portfolios.

Because bracket thresholds are CPI-indexed under current law, and CPI is
itself simulated per-path, resolution happens PER PATH PER YEAR using that
path's own realized CPI -- there is no single "the" 2045 bracket table,
there's one per simulated path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, override

@dataclass(frozen=True)
class BracketTable:
    """
    One year's marginal-bracket schedule (ordinary income, or LTCG/qualified
    dividends). `brackets` is a tuple of (rate, floor) pairs sorted
    ascending by floor -- e.g. ((0.10, 0), (0.12, 12_400), ...) means 10%
    applies to the taxable-income slice from 0 up to the next floor, etc.

    Immutable by design: a resolved table represents one year's law for one
    path and should never be mutated in place.
    """
    brackets: Tuple[Tuple[float, float], ...]
    standard_deduction: float

    def tax_on(self, taxable_income: float) -> float:
        """Marginal-bracket tax on taxable income (already net of deduction)."""
        if taxable_income <= 0:
            return 0.0
        tax = 0.0
        for i, (rate, floor) in enumerate(self.brackets):
            next_floor = self.brackets[i + 1][1] if i + 1 < len(self.brackets) else None
            top = min(taxable_income, next_floor) if next_floor is not None else taxable_income
            if top <= floor:
                break
            tax += (top - floor) * rate
        return tax

    def scaled(self, factor: float) -> "BracketTable":
        """New table with every dollar threshold (and the deduction) scaled by `factor`."""
        return BracketTable(
            brackets=tuple((rate, floor * factor) for rate, floor in self.brackets),
            standard_deduction=self.standard_deduction * factor,
        )


@dataclass(frozen=True)
class NIITRule:
    """
    Net Investment Income Tax: flat surtax on investment income above a MAGI
    threshold. `indexed=False` reproduces current law -- thresholds have been
    fixed in nominal dollars since 2013 and are NOT part of the annual
    Rev. Proc. inflation adjustments, so real-dollar reach keeps expanding
    even under the "current law" scenario. Set indexed=True to model the
    (currently hypothetical) case where Congress starts indexing it.
    """
    rate: float
    threshold_single: float
    threshold_joint: float
    indexed: bool = False


@dataclass(frozen=True)
class AnnualTaxLaw:
    """Fully resolved tax law for one simulated calendar year, for one path."""
    ordinary: BracketTable
    ltcg: BracketTable  # dollar thresholds only; the 0/15/20% structure lives in `brackets`
    niit: NIITRule

    def compute_tax(self, ordinary_income: float, preferential_income: float) -> float:
        """
        Total federal tax owed for one calendar year.

        Parameters:
        -----------
        ordinary_income : float
            Taxed at `self.ordinary` marginal rates: interest (T-Bill accrual,
            T-Note coupons) and short-term capital gains.
        preferential_income : float
            Taxed at `self.ltcg` rates, STACKED ON TOP of ordinary_income:
            qualified dividends and long-term capital gains.

        Simplifications (see linear_models.LongSPYWithTreasuryLadders):
            - Single-filer thresholds only (NIIT threshold_single, no
              married/HoH selection yet -- this engine has no filing-status
              concept upstream).
            - No wage income, so MAGI for NIIT purposes is taken to be
              ordinary_income + preferential_income directly (nothing to
              subtract or add back).
            - The standard deduction is applied once, against ordinary_income
              first; any unused deduction spills over to reduce
              preferential_income before it stacks.
        """
        total_income = ordinary_income + preferential_income
        deduction = self.ordinary.standard_deduction

        ordinary_taxable = max(0.0, ordinary_income - deduction)
        unused_deduction = max(0.0, deduction - ordinary_income)
        preferential_taxable = max(0.0, preferential_income - unused_deduction)

        ordinary_tax = self.ordinary.tax_on(ordinary_taxable)

        # Preferential income stacks on top of ordinary taxable income, so its
        # rate is read off the LTCG table between [ordinary_taxable,
        # ordinary_taxable + preferential_taxable] -- not from zero.
        stack_floor = ordinary_taxable
        stack_ceiling = ordinary_taxable + preferential_taxable
        preferential_tax = self.ltcg.tax_on(stack_ceiling) - self.ltcg.tax_on(stack_floor)

        niit_threshold = self.niit.threshold_single
        niit_tax = self.niit.rate * max(0.0, total_income - niit_threshold)

        return ordinary_tax + preferential_tax + niit_tax


# ---------------------------------------------------------------------------
# 2026 baseline (IRS Rev. Proc. 2025-32), single filer.
# Fill in married-filing-jointly / HoH tables the same way if/when needed --
# kept to single here to keep the sketch readable.
# ---------------------------------------------------------------------------

BASELINE_2026_SINGLE = AnnualTaxLaw(
    ordinary=BracketTable(
        brackets=(
            (0.10, 0),
            (0.12, 12_400),
            (0.22, 50_400),
            (0.24, 105_700),
            (0.32, 201_775),
            (0.35, 256_225),
            (0.37, 640_600),
        ),
        standard_deduction=16_100,
    ),
    ltcg=BracketTable(
        brackets=(
            (0.00, 0),
            (0.15, 49_450),
            (0.20, 545_500),
        ),
        standard_deduction=16_100,  # same deduction pool; don't double-apply it (see resolve())
    ),
    niit=NIITRule(rate=0.038, threshold_single=200_000, threshold_joint=250_000, indexed=False),
)


class TaxRegimeScenario(ABC):
    """
    Base class: given the baseline AnnualTaxLaw and a path's own simulated
    CPI index (relative to the simulation's start), produce the resolved
    AnnualTaxLaw for a given simulated year.

    Subclasses encode HOW law evolves along one path's timeline -- not
    whether it will. "Whether" is the scenario sweep itself: instantiate
    each subclass once per MC batch and diff the resulting ruin tables.
    """

    def __init__(self, name: str, baseline: AnnualTaxLaw):
        self.name = name
        self.baseline = baseline

    @abstractmethod
    def resolve(self, year: int, cpi_relative_to_start: float) -> AnnualTaxLaw:
        """
        Parameters:
        -----------
        year : int
            Simulated year index (0 = the year containing month 0).
        cpi_relative_to_start : float
            This PATH's simulated CPI level at `year`, divided by its CPI
            level at simulation start (i.e. cumulative inflation realized
            on this specific path so far -- not a population average).
        """
        raise NotImplementedError


class CurrentLawIndexed(TaxRegimeScenario):
    """
    Scenario 1 -- baseline. Today's brackets/deduction/LTCG thresholds
    carried forward, indexed to this path's own realized CPI. NIIT
    threshold stays nominal (current law: unindexed since 2013).
    """

    @override
    def resolve(self, year, cpi_relative_to_start):
        return AnnualTaxLaw(
            ordinary=self.baseline.ordinary.scaled(cpi_relative_to_start),
            ltcg=self.baseline.ltcg.scaled(cpi_relative_to_start),
            niit=self.baseline.niit,  # unchanged: NIIT thresholds are nominal-frozen by design
        )


class HistoricalAverageDrift(TaxRegimeScenario):
    """
    Scenario 2 -- structure gradually drifts toward a longer-run historical
    average rather than staying pegged to today's exact law. Only the top
    marginal rate is drifted in this sketch (the highest-signal, easiest to
    defend historically); every bracket floor is CPI-indexed throughout, same
    as scenario 1. Fully drifted by `drift_years`, then holds (CPI-indexed)
    from there.

    NOTE: this is the scenario most worth spending real research time on --
    "drift the top rate" is a placeholder for whatever structural average
    you're willing to defend (e.g. average top rate 1980-2025 was somewhere
    in the high-30s/low-40s%; you may instead want to drift bracket COUNT,
    not just the top rate, since 1986-1990 had only 2-3 brackets).
    """

    def __init__(self, name, baseline, target_top_rate: float, drift_years: int):
        super().__init__(name, baseline)
        self.target_top_rate = target_top_rate
        self.drift_years = drift_years

    @override
    def resolve(self, year, cpi_relative_to_start):
        progress = min(1.0, year / max(1, self.drift_years))
        base_top_rate = self.baseline.ordinary.brackets[-1][0]
        drifted_top_rate = base_top_rate + progress * (self.target_top_rate - base_top_rate)

        drifted_brackets = self.baseline.ordinary.brackets[:-1] + (
            (drifted_top_rate, self.baseline.ordinary.brackets[-1][1]),
        )
        drifted_table = BracketTable(
            brackets=drifted_brackets,
            standard_deduction=self.baseline.ordinary.standard_deduction,
        ).scaled(cpi_relative_to_start)

        return AnnualTaxLaw(
            ordinary=drifted_table,
            ltcg=self.baseline.ltcg.scaled(cpi_relative_to_start),
            niit=self.baseline.niit,
        )


class RegimeSwitchAtYear(TaxRegimeScenario):
    """
    Scenario 3 -- current law until `switch_year`, then an entirely
    different AnnualTaxLaw (e.g. a pre-2018/TCJA-sunset style structure)
    takes over and is itself CPI-indexed forward from the switch point.
    Deliberately discontinuous, unlike scenario 2's gradual drift -- this
    is meant to bound the "one sharp reform happens" case rather than
    average it away.
    """

    def __init__(self, name, baseline, switch_year: int, alternate_law: AnnualTaxLaw):
        super().__init__(name, baseline)
        self.switch_year = switch_year
        self.alternate_law = alternate_law

    @override
    def resolve(self, year, cpi_relative_to_start):
        active = self.baseline if year < self.switch_year else self.alternate_law
        return AnnualTaxLaw(
            ordinary=active.ordinary.scaled(cpi_relative_to_start),
            ltcg=active.ltcg.scaled(cpi_relative_to_start),
            niit=active.niit,
        )


# ---------------------------------------------------------------------------
# Placeholder alternate law for RegimeSwitchAtYear -- ILLUSTRATIVE ONLY.
# Approximate pre-TCJA (2017) single-filer structure, expressed in 2026
# dollars. Replace with researched figures before this feeds real numbers;
# the point here is the mechanism, not this specific table.
# ---------------------------------------------------------------------------

PRE_TCJA_STYLE_SINGLE = AnnualTaxLaw(
    ordinary=BracketTable(
        brackets=(
            (0.10, 0),
            (0.15, 12_500),
            (0.25, 50_800),
            (0.28, 122_600),
            (0.33, 252_800),
            (0.35, 452_400),
            (0.396, 510_300),
        ),
        standard_deduction=8_500,  # pre-TCJA standard deduction was much lower
    ),
    ltcg=BracketTable(
        brackets=(
            (0.00, 0),
            (0.15, 49_450),
            (0.20, 545_500),
        ),
        standard_deduction=8_500,
    ),
    niit=NIITRule(rate=0.038, threshold_single=200_000, threshold_joint=250_000, indexed=False),
)


# ---------------------------------------------------------------------------
# Config-facing registry, selected by name from portfolio JSON ("tax_regime").
# Deliberately NOT a name->class map like monte_carlo.SUPPORTED_MODELS/
# model_map: TaxRegimeScenario subclasses aren't zero-arg constructible (they
# need a baseline law, and two of them need drift/switch parameters), so this
# is a name->factory-function map instead. Each call returns a fresh
# instance -- cheap, since these are stateless wrappers around resolve() --
# so callers never need to worry about sharing one across portfolios/workers.
# ---------------------------------------------------------------------------

def build_tax_regime(name: str) -> Optional[TaxRegimeScenario]:
    """Builds a named TaxRegimeScenario for config-driven sweeps. "none" (or
    an unset/omitted config field) means untaxed, returning None."""

    if name is None or name == "none":
        return None

    if name == "current_law_indexed":
        return CurrentLawIndexed("current_law_indexed", BASELINE_2026_SINGLE)

    if name == "historical_average_drift":
        return HistoricalAverageDrift(
            "historical_average_drift", BASELINE_2026_SINGLE,
            target_top_rate=0.396, drift_years=15,
        )

    if name == "pre_tcja_reversion":
        return RegimeSwitchAtYear(
            "pre_tcja_reversion", BASELINE_2026_SINGLE,
            switch_year=10, alternate_law=PRE_TCJA_STYLE_SINGLE,
        )

    raise ValueError(
        f"Unknown tax_regime '{name}'. Supported: "
        "none, current_law_indexed, historical_average_drift, pre_tcja_reversion"
    )

