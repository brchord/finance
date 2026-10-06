"""
monte_carlo.py

Orchestrates multi-process parallel Monte Carlo simulations for any
InvestmentStrategy operating on monthly real-space path outputs from
PathSimulator instances.
"""

import argparse
import copy
import inspect
import json
import logging
import math
import os
import time
import traceback

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import market_modelling.path_simulation as ps
import portfolio_models.fast_ladder as fast_ladder
import portfolio_models.linear_models as lm
import job_files

from market_data.yf_fred_market_data import MarketDataManager
from market_modelling.fast_hybrid_path_simulation import (
    simulate_hybrid_paths_fast)
from market_modelling.fast_regime_switching_path_simulation import (
    simulate_regime_switching_bootstrap_paths_fast,
    simulate_regime_switching_paths_fast)
from market_modelling.path_simulation import (
    HybridValuationVARSimulator, PathSimulator,
    RegimeSwitchingBootstrapSimulator, RegimeSwitchingValuationVARSimulator)
from tax_models.regimes import build_tax_regime

logger = logging.getLogger(__name__)


def _generate_paths_fast(
    path_simulator: PathSimulator,
    simulation_months: int,
    num_paths: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dispatches to a Numba-accelerated path generator when one exists for
    path_simulator's type (market_modelling/fast_hybrid_path_simulation.py,
    fast_regime_switching_path_simulation.py), else falls back to the
    reference PathSimulator.simulate_paths unchanged. Three of the six
    supported models have one; the other three (RawBlockBootstrap,
    VARResidualBootstrap, ValuationAdjustedVAR) fall back.

    Both regime-switching generators are bit-identical to their
    references (tests/test_fast_regime_switching_path_simulation.py
    compares exactly). HybridValuationVARSimulator's is not: its reference
    does each VAR step as a BLAS matrix multiply whose internal summation
    order can't be reproduced per path, so its paths match only to
    ~1e-10 relative (tests/test_fast_hybrid_path_simulation.py).
    """
    if isinstance(path_simulator, HybridValuationVARSimulator):
        return simulate_hybrid_paths_fast(
            path_simulator, simulation_months, num_paths, seed=seed)
    if isinstance(path_simulator, RegimeSwitchingValuationVARSimulator):
        return simulate_regime_switching_paths_fast(
            path_simulator, simulation_months, num_paths, seed=seed)
    if isinstance(path_simulator, RegimeSwitchingBootstrapSimulator):
        return simulate_regime_switching_bootstrap_paths_fast(
            path_simulator, simulation_months, num_paths, seed=seed)
    return path_simulator.simulate_paths(
        simulation_months=simulation_months, num_paths=num_paths, seed=seed)


NAV_BAND_PERCENTILES = (5, 10, 25, 50)
# Ages for the unconditional P(ruin before age) metric (MCConfig.ruin_ages).
DEFAULT_RUIN_AGES = (75, 85, 95)


def nav_bands(annual_navs: np.ndarray) -> dict:
    """
    Per-year NAV percentiles across paths, from annual_navs of shape
    (num_paths, years + 1) (see fast_ladder.annual_snapshot_months):
    {"years": [0, 1, ...], "p5": [...], "p10": [...], ...}, year 0 being
    the initial NAV. Ruined paths count with a NAV of 0. Only the median
    and below are kept (doc/plans/UI Design.md, "Downside first").
    """
    q = np.percentile(annual_navs, NAV_BAND_PERCENTILES, axis=0)
    bands: dict = {"years": list(range(annual_navs.shape[1]))}
    for p, values in zip(NAV_BAND_PERCENTILES, q):
        bands[f"p{p}"] = values.tolist()
    return bands


def price_levels(cpi_paths: np.ndarray,
                 simulation_months: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each path's cumulative price level relative to its first month, the
    deflator that turns the strategy's nominal NAVs into today's dollars.

    LongSPYWithTreasuryLadders grows spending with cpi[m] / cpi[0] and taxes
    with the same ratio, so dividing a NAV by it gives a value consistent
    with the spending and bracket indexing on that path.

    Returns:
    --------
    (annual, terminal): annual has shape (num_paths, years + 1) and lines
    up with annual_navs (column 0 is 1.0, the start; column y the end of
    year y, see fast_ladder.annual_snapshot_months); terminal has shape
    (num_paths,), the level at the final month (matches Terminal NAV).
    """
    base = cpi_paths[:, 0]
    snapshots = fast_ladder.annual_snapshot_months(simulation_months)
    annual = np.ones((cpi_paths.shape[0], snapshots.shape[0] + 1))
    annual[:, 1:] = cpi_paths[:, snapshots] / base[:, None]
    terminal = cpi_paths[:, simulation_months - 1] / base
    return annual, terminal


def ruin_probability_by_age(ruin_histogram: Sequence[float],
                            total_paths: int, retirement_age: float,
                            ages: Sequence[float]) -> Dict[str, float]:
    """
    Unconditional P(ruin before age A) for each A in `ages`: the share of
    ALL paths (not just the ruined ones) that hit zero before that age. A
    path ruined in month m is ruined at age retirement_age + m / 12, the
    same convention as planner.decision.month_to_age.

    Unlike ruin_month_median, which is computed over ruined paths only,
    these are monotonic in risk: a portfolio that rarely ruins can't look
    worse than one that ruins often just because its few ruins happen
    earlier. Keys are the ages formatted
    with :g ("85", "92.5") so they survive a JSON round trip.
    """
    cumulative = np.cumsum(np.asarray(ruin_histogram, dtype=float))
    out: Dict[str, float] = {}
    for age in ages:
        # Ruin month m counts iff retirement_age + m / 12 < age, i.e.
        # m < (age - retirement_age) * 12. round() strips float noise
        # before ceil() (e.g. 25 * 12 computed as 300.00000000000006).
        months_before = math.ceil(round((age - retirement_age) * 12.0, 9))
        n = min(max(months_before, 0), len(cumulative))
        ruined = cumulative[n - 1] if n > 0 else 0.0
        out[f"{age:g}"] = float(ruined) / total_paths
    return out


class MonteCarloEngine:
    """
    Orchestrates parallel Monte Carlo simulations for any InvestmentStrategy
    subclass. Encapsulates execution, chunking, and metric extraction across
    worker processes.
    """

    def __init__(
        self,
        strategy: lm.InvestmentStrategy,
        simulator: PathSimulator,
        simulation_months: int = 360,
        initial_nav: float = 1_000_000.0,
        base_seed: int = 42,
    ):
        """
        Parameters:
        -----------
        strategy : InvestmentStrategy
            Portfolio model implementing
            `run_simulation(spx, yield3m, yield5y, initial_nav, months)`.
        simulator : PathSimulator
            Fitted path simulation engine instance inheriting from
            PathSimulator.
        simulation_months : int, default=360 (30 years)
            Total monthly time horizon for each path simulation.
        initial_nav : float, default=1,000,000.0
            Starting capital for each path run.
        base_seed : int, default=42
            Master RNG seed to derive batch process seeds deterministically.
        """
        self.strategy = strategy
        self.path_sim = simulator
        self.simulation_months = simulation_months
        self.initial_nav = initial_nav
        self.rng = np.random.default_rng(seed=base_seed)

    @staticmethod
    def _execute_strategy_batch(
        strategy: lm.InvestmentStrategy,
        path_simulator: PathSimulator,
        simulation_months: int,
        initial_nav: float,
        num_paths: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """
        Static worker method executing a batch of paths inside an isolated
        process.

        Returns:
        --------
        Tuple containing:
            1. final_spx (np.ndarray): Terminal real SPX index levels
               (num_paths,).
            2. final_navs (np.ndarray): Terminal NAV values (num_paths,).
            3. ruin_histogram (np.ndarray): For ruin paths, the histogram of
                                            the month where ruin occurred
                                            (num_paths).
            4. annual_navs (np.ndarray): NAV at the start and at each
               year-end (num_paths, years + 1); see
               fast_ladder.annual_snapshot_months.
            5. annual_price_levels (np.ndarray): each path's price level
               at the same points, to deflate annual_navs (see
               price_levels()).
            6. terminal_price_levels (np.ndarray): each path's price level
               at the final month, to deflate final_navs (num_paths,).
        """
        final_spx = np.empty(num_paths)
        final_navs = np.empty(num_paths)
        ruin_histogram = np.zeros(simulation_months)
        snapshot_months = fast_ladder.annual_snapshot_months(
            simulation_months)
        annual_navs = np.empty((num_paths, snapshot_months.shape[0] + 1))

        # Generate batch real wealth index paths via simulator interface
        paths = path_simulator.simulate_paths(
            simulation_months=simulation_months,
            num_paths=num_paths,
            seed=seed,
        )

        spx_paths, cpi_paths, tbill_paths, tnote_paths = paths
        annual_price_levels, terminal_price_levels = price_levels(
            cpi_paths, simulation_months)

        for i in range(num_paths):
            # Execute monthly strategy simulation
            # (Note: sim_paths include starting point at idx 0, passing monthly
            #  steps 1:).
            nav_paths = strategy.run_simulation(
                spx=spx_paths[i, :],
                cpi=cpi_paths[i, :],
                yield3m=tbill_paths[i, :],
                yield5y=tnote_paths[i, :],
                initial_nav=initial_nav,
                months=simulation_months,
                full_book=False,
            )

            final_spx[i] = spx_paths[i, -1]
            final_navs[i] = nav_paths[-1]
            annual_navs[i, 0] = initial_nav
            annual_navs[i, 1:] = np.asarray(nav_paths)[snapshot_months]
            if nav_paths[-1] == 0.0:
                ruin_month = np.argmax(nav_paths == 0.0)
                ruin_histogram[ruin_month] += 1

        return (final_spx, final_navs, ruin_histogram, annual_navs,
                annual_price_levels, terminal_price_levels)

    @staticmethod
    def _chunk_sizes(total_paths: int, n_workers: int) -> List[int]:
        """
        Splits total_paths into per-worker chunk sizes. Shared by submit()
        (backend="process") and MonteCarloCLI._run_numba()'s path
        generation (backend="numba"), so both draw the identical sequence
        of per-chunk seeds from a cell's seed and therefore simulate the
        same market paths for the same cell (bit-identical except for
        HybridValuationVARSimulator; see _generate_paths_fast).

        chunk_size is clamped to at least 1: with fewer paths than workers,
        total_paths // n_workers is 0 and the loop below would never
        finish. The clamp only changes anything in that case (one path
        per chunk, fewer chunks than workers); whenever total_paths >=
        n_workers the chunks, and so every chunk seed and result, are the
        same as before.
        """
        chunk_size = max(1, total_paths // n_workers)
        chunks = []

        remaining_paths = total_paths
        while remaining_paths > 0:
            current_batch_size = min(chunk_size, remaining_paths)
            chunks.append(current_batch_size)
            remaining_paths -= current_batch_size
        return chunks

    @staticmethod
    def _execute_cell_batch_fast(
        path_simulator: PathSimulator,
        simulation_months: int,
        initial_nav: float,
        num_paths: int,
        seed: int,
        portfolio_params: List[Tuple[float, float, float, float, tuple]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Worker for backend="numba" (MonteCarloCLI._run_numba): generates
        one chunk of a CELL's market paths once, inside this worker
        process, then evaluates every tax-regime variant of that cell
        (`portfolio_params`, each a (equity_allocation, ladder_allocation,
        yearly_spending, spy_div_yield, flattened_tax_regime) tuple)
        against that single chunk via fast_ladder.run_simulation_fast_batch.

        This is the key difference from backend="process"'s
        _execute_strategy_batch: paths are generated once per cell instead
        of once per tax-regime variant, and the (potentially ~GB-scale)
        path arrays never leave this process -- only the much smaller
        final_navs/ruin_months/annual_navs arrays are pickled back to the
        parent.

        Runs single-threaded (numba.set_num_threads(1)): this function
        itself is already run in parallel by a ProcessPoolExecutor across
        chunks, same as backend="process"; letting run_simulation_fast_
        batch's internal prange also claim every core per worker would
        oversubscribe them.
        """
        import numba
        numba.set_num_threads(1)

        spx, cpi, tbill, tnote = _generate_paths_fast(
            path_simulator, simulation_months, num_paths, seed)
        final_spx = spx[:, -1]
        annual_price_levels, terminal_price_levels = price_levels(
            cpi, simulation_months)

        per_portfolio = []
        for equity, ladder, spending, div, flat in portfolio_params:
            per_portfolio.append(fast_ladder.run_simulation_fast_batch(
                spx, cpi, tbill, tnote, initial_nav, simulation_months,
                equity, ladder, spending, div, *flat))

        return (final_spx, annual_price_levels, terminal_price_levels,
                per_portfolio)

    def submit(
        self,
        executor: ProcessPoolExecutor,
        *,
        total_paths: int,
        n_workers: int
    ) -> List[Future]:
        """
        Splits total_paths into chunks and submits them to an existing
        executor without waiting for results, so several engines can keep the
        same pool busy at once. Chunk seeds are drawn here, in submission
        order, so results stay deterministic. Pass the returned futures to
        collect().
        """
        chunks = self._chunk_sizes(total_paths, n_workers)

        return [
            executor.submit(
                MonteCarloEngine._execute_strategy_batch,
                self.strategy,
                self.path_sim,
                self.simulation_months,
                self.initial_nav,
                batch_size,
                int(self.rng.integers(1 << 31)),
            )
            for batch_size in chunks
        ]

    def collect(
        self,
        futures: List[Future],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """
        Blocks until the futures returned by submit() are done and assembles
        their results.

        Returns:
        --------
        Tuple containing (final_spx, final_navs, ruin_histogram, annual_navs,
        annual_price_levels, terminal_price_levels); see
        _execute_strategy_batch.
        """
        all_final_spx = []
        all_final_navs = []
        all_annual_navs = []
        all_annual_levels = []
        all_terminal_levels = []
        full_ruin_histogram = np.zeros(self.simulation_months)

        # Iterate futures in their original submission order rather than
        # as_completed()'s finish order. Submission (and each chunk's seed)
        # is already deterministic; only the assembly order was not, which
        # silently permuted which array position each simulated path landed
        # in from run to run.
        for future in futures:
            (f_spx, f_navs, f_ruin_histograms, f_annual, f_annual_levels,
             f_terminal_levels) = future.result()
            all_final_spx.append(f_spx)
            all_final_navs.append(f_navs)
            all_annual_navs.append(f_annual)
            all_annual_levels.append(f_annual_levels)
            all_terminal_levels.append(f_terminal_levels)
            full_ruin_histogram += f_ruin_histograms

        return (
            np.concatenate(all_final_spx),
            np.concatenate(all_final_navs),
            full_ruin_histogram,
            np.concatenate(all_annual_navs),
            np.concatenate(all_annual_levels),
            np.concatenate(all_terminal_levels),
        )

    def run(
        self,
        *,
        total_paths: int,
        n_workers: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """
        Convenience wrapper: runs this engine alone on its own process pool.

        Parameters:
        -----------
        total_paths : int
            Total Monte Carlo paths to generate and evaluate.
        n_workers : int
            Number of parallel process workers in the process pool.

        Returns:
        --------
        Tuple containing (final_spx, final_navs, ruin_histogram, annual_navs,
        annual_price_levels, terminal_price_levels); see collect().
        """
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = self.submit(
                executor, total_paths=total_paths, n_workers=n_workers)
            return self.collect(futures)


class MonteCarloCLI:
    """
    Class encapsulating the Command Line Interface functionality for
    kicking off Monte Carlo simulations using this sofware package.
    """
    SUPPORTED_MODELS: list[type[ps.PathSimulator]] = [
                ps.HybridValuationVARSimulator,
                ps.RawBlockBootstrapSimulator,
                ps.RegimeSwitchingBootstrapSimulator,
                ps.RegimeSwitchingValuationVARSimulator,
                ps.VARResidualBootstrapSimulator,
                ps.ValuationAdjustedVARSimulator]

    def __init__(self,
                 input_config_file: str,
                 market_data_file: str,
                 progress_callback: Optional[
                     Callable[[int, int], None]] = None):
        """
        progress_callback, if given, is called as progress_callback(done,
        total) each time another portfolio's results have been collected
        (total is MCConfig.total_portfolios()).
        """
        self.input_file = input_config_file
        self.progress_callback = progress_callback
        self.mdm = MarketDataManager(cache_filepath=market_data_file)
        self.model_map = {m.name(): m for m in self.SUPPORTED_MODELS}
        self.simulation_config = None
        self.raw_results = None
        self.agg_results = None

    class MCConfig:
        """"
        Represents a Monte Carlo simulation configuration, useful to generate
        multiple portfolio configs to feed it to the MC engine.
        """
        PORTFOLIO_CONFIG_TEMPLATE = {
            "type": "LongSPYWithTreasuryLadders",
            "equity_allocation": None,
            "ladder_allocation": None,
            "yearly_spending":  None,
            "dividend_yield": 0.01,
            "tax_regime": "none",
        }

        def __init__(self, *,
                     yearly_spending_floor: float = 150_000,
                     yearly_spending_ceil: float = 300_000,
                     starting_equity: float = 0.75,
                     ending_equity: float = 1.00,
                     weight_increments: float = 0.05,
                     spend_increments: float = 5000.0,
                     initial_nav: float = 1_000_000.0,
                     years_to_simulate: float = 35.0,
                     retirement_age: float = 65.0,
                     total_paths: int = 10_000,
                     n_workers: Optional[int] = os.cpu_count(),
                     models: list[str],
                     tax_regimes: Optional[list[str]] = None,
                     master_seed: Optional[int] = None,
                     dividend_yield: float = 0.01,
                     simulator_params: Optional[dict] = None,
                     ruin_ages: Optional[Sequence[float]] = None):
            """
            dividend_yield: the strategy's SPY cash dividend yield (was a
                fixed 1%).
            simulator_params: constructor keyword arguments for the path
                simulators, e.g. {"initial_cape": 40, "annual_buyback_yield":
                0.015}. Each model receives only the keys its constructor
                accepts; see MonteCarloCLI._validate_simulator_params.
            ruin_ages: ages for the unconditional P(ruin before age) metric
                in the aggregated results (default 75, 85, 95).
            """
            self.yearly_low = yearly_spending_floor
            self.yearly_top = yearly_spending_ceil
            self.equity_low = starting_equity
            self.equity_top = ending_equity
            self.eq_increment = weight_increments
            self.spend_increment = spend_increments
            self.initial_nav = initial_nav
            self.years = years_to_simulate
            # The one place years become months; both backends read
            # simulation_months rather than converting on their own. A JSON
            # config yields a float for "10.0" or "10.5", and the process
            # backend used to pass years * 12 straight through as an array
            # size, which numpy rejects (np.zeros(120.0) -> TypeError).
            months = years_to_simulate * 12
            self.simulation_months = round(months)
            if abs(months - self.simulation_months) > 1e-6:
                logging.warning(
                    "years_to_simulate=%s is %s months, not a whole number; "
                    "simulating %d months.",
                    years_to_simulate, months, self.simulation_months)
            self.retirement_age = retirement_age
            self.total_paths = total_paths
            self.n_workers = n_workers
            self.models = models
            self.tax_regimes = (
                    tax_regimes if tax_regimes is not None else ["none"])
            self.master_seed = master_seed
            self.dividend_yield = dividend_yield
            self.simulator_params = dict(simulator_params or {})
            self.ruin_ages = list(ruin_ages if ruin_ages is not None
                                  else DEFAULT_RUIN_AGES)

        def assumptions(self) -> dict:
            """
            The market and strategy assumptions of this run, recorded in
            both the raw and aggregated results so a results file says what
            it was computed under.
            """
            return {
                "dividend_yield": self.dividend_yield,
                "simulator_params": dict(self.simulator_params),
            }

        def _step_count(self, low, high, increment):
            """
            Number of steps from low to high inclusive, given a fixed
            increment.
            """
            return int(round((high - low) / increment)) + 1

        def portfolio_configs(self):
            """"
            Generates portfolio configuration by sweeping a range
            of yearly spendings, equity allocations, and tax regimes.

            tax_regime is deliberately the INNERMOST loop: run() relies on
            all tax-regime variants of a given (model, equity, spending)
            cell being adjacent in this generator's output, so it can hand
            them the same random seed (common random numbers) instead of
            an independent one each -- otherwise a regime-vs-regime
            comparison is contaminated by full independent-sample noise
            on top of whatever the real tax effect is.
            """
            n_equity = self._step_count(
                self.equity_low, self.equity_top, self.eq_increment)
            n_yearly = self._step_count(
                self.yearly_low, self.yearly_top, self.spend_increment)

            for model in self.models:
                for i in range(n_equity):
                    equity = round(self.equity_low + i * self.eq_increment, 6)
                    fixed = round(1.0 - equity, 6)
                    for j in range(n_yearly):
                        yearly = round(
                            self.yearly_low + j * self.spend_increment, 2)
                        for tax_regime in self.tax_regimes:
                            p = copy.deepcopy(self.PORTFOLIO_CONFIG_TEMPLATE)
                            p["equity_allocation"] = equity
                            p["ladder_allocation"] = fixed
                            p["yearly_spending"] = yearly
                            p["model"] = model
                            p["tax_regime"] = tax_regime
                            p["dividend_yield"] = self.dividend_yield
                            yield p

        def total_portfolios(self):
            """
            Returns the total count of portfolios generated by the given
            config.
            """
            n_equity = self._step_count(
                self.equity_low, self.equity_top, self.eq_increment)
            n_yearly = self._step_count(
                self.yearly_low, self.yearly_top, self.spend_increment)
            return (n_equity * n_yearly *
                    len(self.models) * len(self.tax_regimes))

    def _load_config(self):
        try:
            with open(self.input_file, encoding="utf-8") as f:
                config = json.load(f)
            mc_config = MonteCarloCLI.MCConfig(
                yearly_spending_floor=config["yearly_spending_floor"],
                yearly_spending_ceil=config["yearly_spending_ceil"],
                starting_equity=config["equity_floor"],
                ending_equity=config["equity_ceil"],
                weight_increments=config["weight_increments"],
                spend_increments=config["spend_increments"],
                initial_nav=config["initial_nav"],
                years_to_simulate=config["years_to_simulate"],
                retirement_age=config["retirement_age"],
                total_paths=config["total_paths"],
                n_workers=config["workers"],
                models=config["models"],
                tax_regimes=config.get("tax_regimes", ["none"]),
                master_seed=config.get("master_seed"),
                dividend_yield=config.get("dividend_yield", 0.01),
                simulator_params=config.get("simulator_params"),
                ruin_ages=config.get("ruin_ages"))
            logging.info("Loaded configuration: ")
            kvs = [f"{k.replace('_', ' ').title()}: {v}"
                   for k, v in config.items()]
            logging.info(" ".join(kvs))
            self.simulation_config = mc_config
        except Exception as exc:
            logging.error("Error loading tool configuration: %s", str(exc))
            raise exc

    def _validate_simulator_params(self):
        """
        Rejects simulator_params keys that no selected model accepts (most
        likely a typo, which would otherwise be silently ignored) and logs
        which models ignore which keys.
        """
        config = self.simulation_config
        assert config is not None, "config must be loaded first"
        params = config.simulator_params
        if not params:
            return
        unused = set(params)
        for model_name in config.models:
            if model_name not in self.model_map:
                raise ValueError(f"Model {model_name} not supported")
            accepted = inspect.signature(
                self.model_map[model_name].__init__).parameters
            ignored = sorted(k for k in params if k not in accepted)
            unused -= set(params) - set(ignored)
            if ignored:
                logging.info("%s ignores simulator_params %s",
                             model_name, ", ".join(ignored))
        if unused:
            raise ValueError(
                "simulator_params not accepted by any selected model: "
                f"{', '.join(sorted(unused))}")

    def _build_simulator(self, model_name: str) -> PathSimulator:
        """
        Instantiates (but does not fit) a model, passing it the configured
        simulator_params its constructor accepts.
        """
        config = self.simulation_config
        assert config is not None, "config must be loaded first"
        cls = self.model_map[model_name]
        accepted = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in config.simulator_params.items()
                  if k in accepted}
        return cls(**kwargs)

    def run(self, *, backend: str = "process"):
        """
        Starts the simulation, collects results and stores them into a
        dictionary for further serialization and/or aggreggation analysis.

        Parameters:
        -----------
        backend : str, default="process"
            "process": the original ProcessPoolExecutor-based execution.
                Unchanged; this is what tests/golden/monte_carlo_agg.json
                was generated with, and the default so existing callers and
                the golden snapshot are unaffected.
            "numba": executes the portfolio operator via
                fast_ladder.run_simulation_fast_batch instead of a
                process pool running the scalar run_simulation per path.
                Also fits each path simulator once per model instead of
                once per portfolio, and generates each cell's market paths
                once instead of once per tax-regime variant within it --
                both pure deduplication of work the "process" backend
                redundantly repeats for every tax-regime variant of a cell
                (same cell_seed, so bit-identical results either way), not
                a change to the statistical design. Uses the identical
                chunk-size/seed sequence as "process" (see
                MonteCarloEngine._chunk_sizes), and its fast operator and
                regime-switching path generators are bit-identical to the
                reference, so for the same config "numba" produces exactly
                the same results as "process" -- except for
                HybridValuationVARSimulator, whose fast path generator
                matches only to ~1e-10 relative (see
                _generate_paths_fast), which can flip a decision on rare
                paths.
        """
        if backend == "process":
            return self._run_process_pool()
        if backend == "numba":
            return self._run_numba()
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: process, numba")

    def _run_process_pool(self):
        if self.raw_results is not None:
            raise RuntimeError(
                "CLI run can only be run once per instantiation")

        self._load_config()
        self._validate_simulator_params()
        config = self.simulation_config

        results = {
            "initial_nav": config.initial_nav,
            "years": config.years,
            "total_paths": config.total_paths,
            "assumptions": config.assumptions(),
            "simulations": {},
            "perf_data": {}
        }

        perf_counters = {}
        rng = np.random.default_rng(seed=config.master_seed)
        levels, returns = self.mdm.get_aligned_real_returns()
        total = config.total_portfolios()
        i = 0

        for model_name in config.models:
            if model_name not in self.model_map:
                raise ValueError(f"Model {model_name} not supported")
            results["simulations"][model_name] = []
            perf_counters[model_name] = {}
            perf_counters[model_name]["setup"] = []
            perf_counters[model_name]["simulation"] = []
            perf_counters[model_name]["data_storage"] = []

        portfolios = config.portfolio_configs()

        # Common random numbers: all tax-regime variants of the same
        # (model, equity, spending) cell share one seed, so a regime
        # comparison isolates the tax effect instead of being contaminated
        # by an independent, unrelated draw of 500 market paths per regime.
        cell_key = None
        cell_seed = None

        # One pool for the whole run. Every portfolio's chunks are submitted
        # up front (fitting the next portfolio while workers crunch the
        # previous ones), so workers never idle waiting on the slowest chunk
        # of a portfolio or on serial setup. Results are collected in
        # portfolio order, so output is identical to running them one by one.
        pending = []
        done = 0
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            for p in portfolios:
                model_name = p["model"]
                logging.info(
                    "Submitting simulation #%d out of %d. Progress: %.2f%%",
                    i + 1, total, 100.0 * (i + 1) / total)
                logging.info(
                    "Model: %s, Tax Regime: %s, Yearly Spending: $%.2f, "
                    "Equity: %.2f%%", model_name, p["tax_regime"],
                    p["yearly_spending"], p["equity_allocation"] * 100.0)
                setup_start = time.perf_counter()

                simulator = self._build_simulator(model_name)
                simulator.fit(returns, levels)

                new_cell_key = (model_name, p["equity_allocation"],
                                p["yearly_spending"])
                if new_cell_key != cell_key:
                    cell_key = new_cell_key
                    cell_seed = int(rng.integers(1 << 32))

                strategy = lm.LongSPYWithTreasuryLadders.from_json_object(p)
                mc = MonteCarloEngine(strategy, simulator,
                                      config.simulation_months,
                                      config.initial_nav, cell_seed)
                futures = mc.submit(executor, total_paths=config.total_paths,
                                    n_workers=config.n_workers)
                setup_end = time.perf_counter()

                perf_counters[model_name]["setup"].append(
                    setup_end - setup_start)
                pending.append((p, mc, futures))
                i += 1

            for p, mc, futures in pending:
                model_name = p["model"]
                # Time spent blocked waiting on this portfolio's results;
                # simulation itself overlaps with other portfolios' setup.
                sim_start = time.perf_counter()
                (spx, nav, ruin_histogram, annual_navs, annual_levels,
                 terminal_levels) = mc.collect(futures)
                sim_end = time.perf_counter()

                perf_counters[model_name]["simulation"].append(
                    sim_end - sim_start)

                data_start = time.perf_counter()
                run_output = {
                        "Terminal SPX": spx.tolist(),
                        "Terminal NAV": nav.tolist(),
                        "Terminal Real NAV": (nav / terminal_levels).tolist(),
                }
                results["simulations"][model_name].append({
                    "spending": p["yearly_spending"],
                    "equity": p["equity_allocation"],
                    "ladder": p["ladder_allocation"],
                    "tax_regime": p["tax_regime"],
                    "ruin_histogram": ruin_histogram.tolist(),
                    "nav_bands": nav_bands(annual_navs),
                    "real_nav_bands": nav_bands(annual_navs / annual_levels),
                    "results": run_output
                })
                data_end = time.perf_counter()
                perf_counters[model_name]["data_storage"].append(
                    data_end - data_start)
                done += 1
                self._report_progress(done, total)

        results["perf_data"] = perf_counters
        self.raw_results = results

    def _report_progress(self, done: int, total: int):
        if self.progress_callback is not None:
            self.progress_callback(done, total)

    def _run_numba(self):
        """
        See run()'s "numba" backend docstring.

        Still uses a ProcessPoolExecutor -- not threads. Measured directly:
        threading path generation (plain numpy/Python code, not
        Numba-jitted) made it ~5x SLOWER here, not faster, since the
        simulators' per-step Python loops barely release the GIL and
        concurrent numpy/BLAS calls from several threads mostly just
        contend with each other. Processes avoid that, same as
        backend="process" today.

        What changes from backend="process" is the unit of work: each
        worker call (_execute_cell_batch_fast) now covers one chunk of an
        entire CELL (every tax-regime variant of one (model, equity,
        spending) combination) instead of one chunk of a single portfolio.
        The worker generates that chunk's market paths once, evaluates
        every regime's portfolio operator against them via
        fast_ladder.run_simulation_fast_batch, and returns only the
        resulting final_navs/ruin_months arrays -- the (potentially
        ~GB-scale) path arrays themselves never cross a process boundary,
        and are never regenerated per regime. Each model is also fit()
        once here, not once per portfolio, since fit() depends only on the
        model class and the shared market data.

        Common random numbers: cell seeds and the per-chunk seed sequence
        are drawn identically to backend="process" (see
        MonteCarloEngine._chunk_sizes), so for the same config both
        backends simulate the same market paths for a given cell
        (bit-identical except for HybridValuationVARSimulator; see
        _generate_paths_fast).
        """
        if self.raw_results is not None:
            raise RuntimeError(
                "CLI run can only be run once per instantiation")

        self._load_config()
        self._validate_simulator_params()
        config = self.simulation_config
        simulation_months = config.simulation_months

        results = {
            "initial_nav": config.initial_nav,
            "years": config.years,
            "total_paths": config.total_paths,
            "assumptions": config.assumptions(),
            "simulations": {},
            "perf_data": {}
        }

        perf_counters = {}
        rng = np.random.default_rng(seed=config.master_seed)
        levels, returns = self.mdm.get_aligned_real_returns()
        total = config.total_portfolios()

        for model_name in config.models:
            if model_name not in self.model_map:
                raise ValueError(f"Model {model_name} not supported")
            results["simulations"][model_name] = []
            perf_counters[model_name] = {
                "setup": [], "simulation": [], "data_storage": []}

        # fit() depends only on the model class and the shared market data
        # -- never on a portfolio's allocation/spending/tax regime -- so,
        # unlike backend="process", it only needs to run once per model.
        fitted_simulators = {
            model_name: self._build_simulator(model_name)
            for model_name in config.models
        }
        for sim in fitted_simulators.values():
            sim.fit(returns, levels)

        # Per-year tax bracket tables, flattened once per regime name and
        # reused across every cell that uses it (see
        # fast_ladder.flatten_tax_regime).
        max_years = (simulation_months + 11) // 12 + 1
        tax_regime_flat_cache: dict = {}

        def flat_for(tax_regime_name):
            if tax_regime_name not in tax_regime_flat_cache:
                tax_regime_flat_cache[tax_regime_name] = (
                    fast_ladder.flatten_tax_regime(
                        build_tax_regime(tax_regime_name), max_years))
            return tax_regime_flat_cache[tax_regime_name]

        # Group portfolio_configs()'s output into cells -- tax_regime is
        # deliberately its innermost loop (see MCConfig.portfolio_configs),
        # so consecutive entries sharing (model, equity, spending) are
        # exactly one cell's tax-regime variants.
        cells: List[Tuple[Tuple, List[dict]]] = []
        for p in config.portfolio_configs():
            key = (p["model"], p["equity_allocation"], p["yearly_spending"])
            if not cells or cells[-1][0] != key:
                cells.append((key, []))
            cells[-1][1].append(p)

        # One pool for the whole run, every cell's chunks submitted up
        # front (so workers never idle waiting on the slowest chunk of a
        # cell or on serial setup), collected in cell order afterward.
        pending = []
        done = 0
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            for cell_idx, (cell_key, portfolios) in enumerate(cells):
                model_name = cell_key[0]
                logging.info(
                    "Submitting cell #%d out of %d (%d tax-regime "
                    "variant(s)). Progress: %.2f%%",
                    cell_idx + 1, len(cells), len(portfolios),
                    100.0 * sum(len(ps_) for _, ps_ in cells[:cell_idx + 1])
                    / total)
                logging.info(
                    "Model: %s, Yearly Spending: $%.2f, Equity: %.2f%%",
                    model_name, cell_key[2], cell_key[1] * 100.0)

                setup_start = time.perf_counter()
                cell_seed = int(rng.integers(1 << 32))
                chunk_rng = np.random.default_rng(cell_seed)
                chunks = MonteCarloEngine._chunk_sizes(
                    config.total_paths, config.n_workers)

                portfolio_params = [
                    (p["equity_allocation"], p["ladder_allocation"],
                     p["yearly_spending"], p.get("dividend_yield", 0.01),
                     flat_for(p["tax_regime"]))
                    for p in portfolios
                ]

                chunk_futures = [
                    executor.submit(
                        MonteCarloEngine._execute_cell_batch_fast,
                        fitted_simulators[model_name], simulation_months,
                        config.initial_nav, batch_size,
                        int(chunk_rng.integers(1 << 31)), portfolio_params)
                    for batch_size in chunks
                ]
                setup_end = time.perf_counter()
                perf_counters[model_name]["setup"].append(
                    setup_end - setup_start)
                pending.append((cell_key, portfolios, chunk_futures))

            for cell_key, portfolios, chunk_futures in pending:
                model_name = cell_key[0]
                sim_start = time.perf_counter()
                # Submission order, not completion order -- same
                # determinism rationale as collect() in backend="process".
                chunk_results = [f.result() for f in chunk_futures]
                sim_end = time.perf_counter()
                perf_counters[model_name]["simulation"].append(
                    sim_end - sim_start)

                data_start = time.perf_counter()
                final_spx = np.concatenate(
                    [chunk[0] for chunk in chunk_results])
                annual_levels = np.concatenate(
                    [chunk[1] for chunk in chunk_results])
                terminal_levels = np.concatenate(
                    [chunk[2] for chunk in chunk_results])
                for idx, p in enumerate(portfolios):
                    final_navs = np.concatenate(
                        [chunk[3][idx][0] for chunk in chunk_results])
                    ruin_months = np.concatenate(
                        [chunk[3][idx][1] for chunk in chunk_results])
                    annual_navs = np.concatenate(
                        [chunk[3][idx][2] for chunk in chunk_results])
                    ruin_histogram = np.zeros(simulation_months)
                    ruined = ruin_months[ruin_months >= 0]
                    if ruined.size > 0:
                        counts = np.bincount(
                            ruined, minlength=simulation_months)
                        ruin_histogram = counts[
                            :simulation_months].astype(float)

                    results["simulations"][model_name].append({
                        "spending": p["yearly_spending"],
                        "equity": p["equity_allocation"],
                        "ladder": p["ladder_allocation"],
                        "tax_regime": p["tax_regime"],
                        "ruin_histogram": ruin_histogram.tolist(),
                        "nav_bands": nav_bands(annual_navs),
                        "real_nav_bands": nav_bands(
                            annual_navs / annual_levels),
                        "results": {
                            "Terminal SPX": final_spx.tolist(),
                            "Terminal NAV": final_navs.tolist(),
                            "Terminal Real NAV": (
                                final_navs / terminal_levels).tolist(),
                        },
                    })
                data_end = time.perf_counter()
                perf_counters[model_name]["data_storage"].append(
                    data_end - data_start)
                done += len(portfolios)
                self._report_progress(done, total)

        results["perf_data"] = perf_counters
        self.raw_results = results

    def aggregate(self):
        """
        Processes the raw results of a Monte Carlo simulation
        and generates a dictionary representing the aggregation
        and statistical information of the given run.
        """
        if self.agg_results is not None:
            raise RuntimeError(
                "Data aggregation can only be run once per CLI instance")

        sim_data = self.raw_results["simulations"]
        models = list(sim_data.keys())

        initial_nav = self.raw_results["initial_nav"]
        years = self.raw_results["years"]
        paths = self.raw_results["total_paths"]

        run_stats = {}
        run_stats["initial_nav"] = initial_nav
        run_stats["years_to_simulate"] = years
        run_stats["total_paths"] = paths
        run_stats["retirement_age"] = self.simulation_config.retirement_age
        # None for raw results produced before assumptions were recorded.
        run_stats["assumptions"] = self.raw_results.get("assumptions")
        run_stats["results"] = {}

        for m in models:
            r = sim_data[m]
            run_stats["results"][m] = []
            for sim in r:
                yearly_spending = sim["spending"]
                allocation_str = (
                    f"{sim["equity"]*100.0:02.0f}-"
                    f"{sim["ladder"]*100.0:02.0f}")
                df_data = pd.DataFrame(sim["results"])
                df_data["Returns"] = (
                    (df_data["Terminal NAV"] - initial_nav)
                    / initial_nav
                )
                # Earliest and median ruin month, over ruined paths only.
                # Anything else about ruin timing derives from
                # ruin_histogram (and ruin_prob_by_age below).
                ruin_ages = {
                    i: int(v) for
                    (i, v) in enumerate(sim["ruin_histogram"]) if v > 0
                }
                v = list(ruin_ages.keys())
                f = list(ruin_ages.values())
                ruin_flat_data = np.repeat(v, f)

                if len(ruin_flat_data) > 0:
                    p50 = np.quantile(ruin_flat_data, 0.5)
                    ruin = len(ruin_flat_data)
                    min_ruin_age = int(ruin_flat_data.min())
                else:
                    p50 = None
                    ruin = 0
                    min_ruin_age = None

                entry = {}
                entry["spending"] = yearly_spending
                entry["allocation"] = allocation_str
                entry["equity"] = sim["equity"]
                entry["tax_regime"] = sim["tax_regime"]

                entry["ruin_path_count"] = ruin
                entry["ruin_month_min"] = min_ruin_age
                entry["ruin_month_median"] = (
                        float(p50) if p50 is not None else None)

                entry["p5_return"] = df_data["Returns"].quantile(0.05)
                entry["p10_return"] = df_data["Returns"].quantile(0.1)

                entry["p25_return"] = df_data["Returns"].quantile(0.25)
                entry["p50_return"] = df_data["Returns"].quantile(0.5)
                # The returns above are NOMINAL (Terminal NAV is in
                # future dollars). The real ones deflate each path's
                # terminal NAV by its own price level; None for raw results
                # produced before Terminal Real NAV existed.
                real_navs = sim["results"].get("Terminal Real NAV")
                real_returns = (
                    (np.asarray(real_navs) - initial_nav) / initial_nav
                    if real_navs is not None else None)
                for q, name in ((0.05, "p5"), (0.1, "p10"), (0.25, "p25"),
                                (0.5, "p50")):
                    entry[f"{name}_real_return"] = (
                        float(np.quantile(real_returns, q))
                        if real_returns is not None else None)
                entry["ruin_prob_by_age"] = ruin_probability_by_age(
                    sim["ruin_histogram"], paths,
                    self.simulation_config.retirement_age,
                    self.simulation_config.ruin_ages)
                # Monthly ruin counts (month index since retirement), for
                # survival curves: P(solvent after month m) =
                # 1 - cumsum(ruin_histogram)[m] / total_paths.
                entry["ruin_histogram"] = [
                    int(v) for v in sim["ruin_histogram"]]
                # Per-year NAV percentiles (see nav_bands()); None for raw
                # results produced before they existed.
                entry["nav_bands"] = sim.get("nav_bands")
                # Same bands in today's dollars (see price_levels()).
                entry["real_nav_bands"] = sim.get("real_nav_bands")
                run_stats["results"][m].append(entry)
        self.agg_results = run_stats


def parse_args():
    "CLI argument parser"
    prog_description = """CLI tool that invokes MC simulation across all
    portfolio models using the Long Equity and Fixed Income Ladders strategy.

    Writes the raw and aggregated simulation results as JSON files.
    """
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument("-c", "--config-file",
                        help="Config file in JSON format that contains the "
                             "simulation parameters. Required unless "
                             "--job-dir is given",
                        dest="config_filename")
    parser.add_argument("-r", "--raw-output-file",
                        help="Destination JSON file to store the raw results "
                             "of the simulation. Required unless --job-dir "
                             "is given, in which case the raw results are "
                             "only written if this is given too",
                        dest="raw_output_filename")
    parser.add_argument("-o", "--aggregated-output-file",
                        help="Destination JSON file to store aggregated "
                             "results of the simulation. Required unless "
                             "--job-dir is given",
                        dest="agg_output_filename")
    parser.add_argument("-j", "--job-dir",
                        help="Job folder, as used by the UI: reads "
                             f"{job_files.CONFIG_FILE} from it and writes "
                             f"{job_files.STATUS_FILE} (progress), "
                             f"{job_files.META_FILE} and "
                             f"{job_files.RESULTS_FILE} (aggregated results) "
                             "into it. -c and -o override the config and "
                             "results paths",
                        dest="job_dir")
    parser.add_argument("-m", "--market-cache-file",
                        help="Specifies an alternative parquet market data "
                             "cache file",
                        default="market_data.parquet",
                        dest="market_data_filename")
    parser.add_argument("-b", "--backend",
                        help="Execution backend (see MonteCarloCLI.run): "
                             "'process' runs the original Python simulation "
                             "code; 'numba' runs the compiled fast path, "
                             "which gives identical results (except for "
                             "HybridValuationVARSimulator, which matches "
                             "only to rounding) and is much faster. "
                             "Default: process",
                        choices=["process", "numba"],
                        default="process",
                        dest="backend")
    args = parser.parse_args()

    if args.job_dir is not None:
        job_dir = Path(args.job_dir)
        if args.config_filename is None:
            args.config_filename = str(job_dir / job_files.CONFIG_FILE)
        if args.agg_output_filename is None:
            args.agg_output_filename = str(job_dir / job_files.RESULTS_FILE)
    else:
        missing = [flag for flag, value in (
            ("-c/--config-file", args.config_filename),
            ("-r/--raw-output-file", args.raw_output_filename),
            ("-o/--aggregated-output-file", args.agg_output_filename))
            if value is None]
        if missing:
            parser.error("the following arguments are required unless "
                         f"--job-dir is given: {', '.join(missing)}")
    return args


def _write_meta(job_dir: Path, args):
    "Records what is needed to reproduce a job: seed, code version, backend."
    with open(args.config_filename, encoding="utf-8") as f:
        config = json.load(f)
    job_files.write_json_atomic(job_dir / job_files.META_FILE, {
        "started_at": job_files.now_iso(),
        "master_seed": config.get("master_seed"),
        "engine_commit": job_files.git_commit(Path(__file__).parent),
        "backend": args.backend,
        "market_cache_file": args.market_data_filename,
    })


def _run_and_save(cli: MonteCarloCLI, args):
    cli.run(backend=args.backend)
    if args.raw_output_filename is not None:
        with open(args.raw_output_filename, "w", encoding="utf-8") as f:
            json.dump(cli.raw_results, f, indent=4)

    cli.aggregate()
    job_files.write_json_atomic(
        Path(args.agg_output_filename), cli.agg_results, indent=4)


def main():
    "Main entrypoint"
    logging.basicConfig(
        format="%(asctime)s:%(filename)s:"
               "%(lineno)d:%(levelname)s: %(message)s",
        level=logging.INFO)
    args = parse_args()

    if args.job_dir is None:
        cli = MonteCarloCLI(args.config_filename, args.market_data_filename)
        _run_and_save(cli, args)
        return

    job_dir = Path(args.job_dir)
    status = job_files.StatusWriter(job_dir, pid=os.getpid())
    try:
        _write_meta(job_dir, args)
        cli = MonteCarloCLI(args.config_filename, args.market_data_filename,
                            progress_callback=status.progress)
        _run_and_save(cli, args)
    except BaseException as exc:
        status.failed(
            "".join(traceback.format_exception_only(exc)).strip())
        raise
    status.succeeded()


if __name__ == '__main__':
    main()
