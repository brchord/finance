"""
monte_carlo.py

Orchestrates multi-process parallel Monte Carlo simulations for any
InvestmentStrategy operating on monthly real-space path outputs from
PathSimulator instances.
"""

import argparse
import copy
import json
import logging
import os
import time

from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import market_modelling.path_simulation as ps
import portfolio_models.linear_models as lm

from market_data.yf_fred_market_data import MarketDataManager
from market_modelling.path_simulation import PathSimulator

logger = logging.getLogger(__name__)


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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        """
        final_spx = np.empty(num_paths)
        final_navs = np.empty(num_paths)
        ruin_histogram = np.zeros(simulation_months)

        # Generate batch real wealth index paths via simulator interface
        paths = path_simulator.simulate_paths(
            simulation_months=simulation_months,
            num_paths=num_paths,
            seed=seed,
        )

        spx_paths, cpi_paths, tbill_paths, tnote_paths = paths

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
            if nav_paths[-1] == 0.0:
                ruin_month = np.argmax(nav_paths == 0.0)
                ruin_histogram[ruin_month] += 1

        return final_spx, final_navs, ruin_histogram

    def run(
        self,
        *,
        total_paths: int,
        n_workers: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spawns and manages parallel execution across available CPU cores.

        Parameters:
        -----------
        total_paths : int, default=10000
            Total Monte Carlo paths to generate and evaluate.
        n_workers : int, default=8
            Number of parallel process workers in the process pool.

        Returns:
        --------
        Tuple containing (final_spx, final_navs, ruin_histogram)
        """
        chunk_size = total_paths // n_workers
        chunks = []

        remaining_paths = total_paths
        while remaining_paths > 0:
            current_batch_size = min(chunk_size, remaining_paths)
            chunks.append(current_batch_size)
            remaining_paths -= current_batch_size

        all_final_spx = []
        all_final_navs = []
        full_ruin_histogram = np.zeros(self.simulation_months)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
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

            # Iterate futures in their original submission order rather than
            # as_completed()'s finish order. Submission (and each chunk's
            # seed) is already deterministic; only the assembly order was
            # not, which silently permuted which array position each
            # simulated path landed in from run to run. All futures are
            # already running concurrently by this point, so this costs
            # nothing -- it only changes which already-submitted future
            # .result() blocks on next.
            for future in futures:
                f_spx, f_navs, f_ruin_histograms = future.result()
                all_final_spx.append(f_spx)
                all_final_navs.append(f_navs)
                full_ruin_histogram += f_ruin_histograms

        concatenated_spx = np.concatenate(all_final_spx)
        concatenated_navs = np.concatenate(all_final_navs)

        return (
            concatenated_spx,
            concatenated_navs,
            full_ruin_histogram
        )


class MonteCarloCLI:
    """
    Class encapsulating the Command Line Interface functionality for
    kicking off Monte Carlo simulations using this sofware package.
    """
    SUPPORTED_MODELS = [
                ps.HybridValuationVARSimulator,
                ps.RawBlockBootstrapSimulator,
                ps.RegimeSwitchingBootstrapSimulator,
                ps.RegimeSwitchingValuationVARSimulator,
                ps.VARResidualBootstrapSimulator,
                ps.ValuationAdjustedVARSimulator]

    def __init__(self,
                 input_config_file: str,
                 market_data_file: str):
        self.input_file = input_config_file
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
                     n_workers: int = os.cpu_count(),
                     models: list[str],
                     tax_regimes: list[str] = None,
                     master_seed: Optional[int] = None):
            self.yearly_low = yearly_spending_floor
            self.yearly_top = yearly_spending_ceil
            self.equity_low = starting_equity
            self.equity_top = ending_equity
            self.eq_increment = weight_increments
            self.spend_increment = spend_increments
            self.initial_nav = initial_nav
            self.years = years_to_simulate
            self.retirement_age = retirement_age
            self.total_paths = total_paths
            self.n_workers = n_workers
            self.models = models
            self.tax_regimes = (
                    tax_regimes if tax_regimes is not None else ["none"])
            self.master_seed = master_seed

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
                master_seed=config.get("master_seed"))
            logging.info("Loaded configuration: ")
            kvs = [f"{k.replace('_', ' ').title()}: {v}"
                   for k, v in config.items()]
            logging.info(" ".join(kvs))
            self.simulation_config = mc_config
        except Exception as exc:
            logging.error("Error loading tool configuration: %s", str(exc))
            raise exc

    def run(self):
        """
        Starts the simulation, collects results and stores them into a
        dictionary for further serialization and/or aggreggation analysis.
        """
        if self.raw_results is not None:
            raise RuntimeError(
                "CLI run can only be run once per instantiation")

        self._load_config()
        config = self.simulation_config

        results = {
            "initial_nav": config.initial_nav,
            "years": config.years,
            "total_paths": config.total_paths,
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

        for p in portfolios:
            model_name = p["model"]
            m = self.model_map[model_name]
            logging.info("Running simulation #%d out of %d. Progress: %.2f%%",
                         i + 1, total, 100.0 * (i + 1) / total)
            logging.info("Model: %s, Tax Regime: %s, Yearly Spending: $%.2f, "
                         "Equity: %.2f%%", model_name, p["tax_regime"],
                         p["yearly_spending"], p["equity_allocation"] * 100.0)
            setup_start = time.perf_counter()

            simulator = m()
            simulator.fit(returns, levels)

            new_cell_key = (model_name, p["equity_allocation"],
                            p["yearly_spending"])
            if new_cell_key != cell_key:
                cell_key = new_cell_key
                cell_seed = int(rng.integers(1 << 32))

            strategy = lm.LongSPYWithTreasuryLadders.from_json_object(p)
            mc = MonteCarloEngine(strategy, simulator, config.years * 12,
                                  config.initial_nav, cell_seed)
            setup_end = time.perf_counter()

            perf_counters[model_name]["setup"].append(setup_end - setup_start)

            sim_start = time.perf_counter()
            spx, nav, ruin_histogram = mc.run(
                total_paths=config.total_paths, n_workers=config.n_workers)
            sim_end = time.perf_counter()

            perf_counters[model_name]["simulation"].append(sim_end - sim_start)

            data_start = time.perf_counter()
            run_output = {
                    "Terminal SPX": spx.tolist(),
                    "Terminal NAV": nav.tolist()
            }
            results["simulations"][model_name].append({
                "spending": p["yearly_spending"],
                "equity": p["equity_allocation"],
                "ladder": p["ladder_allocation"],
                "tax_regime": p["tax_regime"],
                "ruin_histogram": ruin_histogram.tolist(),
                "results": run_output
            })
            data_end = time.perf_counter()
            perf_counters[model_name]["data_storage"].append(
                data_end - data_start)
            i += 1

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
        run_stats["results"] = {}

        for m in models:
            r = sim_data[m]
            run_stats["results"][m] = []
            for sim in r:
                yearly_spending = sim["spending"]
                allocation_str = f"{sim["equity"]*100.0:02.0f}-" + \
                                 f"{sim["ladder"]*100.0:02.0f}"
                df_data = pd.DataFrame(sim["results"])
                df_data["Returns"] = (
                    (df_data["Terminal NAV"] - initial_nav)
                    / initial_nav
                )
                # Compute the Expected Shortfall 5 and 10 for ages ending in
                # ruin.
                ruin_ages = {
                    i: int(v) for
                    (i, v) in enumerate(sim["ruin_histogram"]) if v > 0
                }
                v = list(ruin_ages.keys())
                f = list(ruin_ages.values())
                ruin_flat_data = np.repeat(v, f)

                if len(ruin_flat_data) > 0:
                    p5, p10, p50 = np.quantile(
                        ruin_flat_data, [0.05, 0.1, 0.5])
                    ruin_flat_data.sort()
                    es5_idx = np.where(ruin_flat_data <= p5)
                    es10_idx = np.where(ruin_flat_data <= p10)
                    es5 = ruin_flat_data[es5_idx].mean()
                    es10 = ruin_flat_data[es10_idx].mean()
                    ruin = len(ruin_flat_data)
                    min_ruin_age = int(ruin_flat_data[0])
                else:
                    p50 = es5 = es10 = None
                    ruin = 0
                    min_ruin_age = None

                entry = {}
                entry["spending"] = yearly_spending
                entry["allocation"] = allocation_str
                entry["tax_regime"] = sim["tax_regime"]

                entry["ruin_path_count"] = ruin
                entry["ruin_month_min"] = min_ruin_age
                entry["ruin_month_median"] = (
                        float(p50) if p50 is not None else None)
                entry["ruin_month_es5"] = (
                        float(es5) if es5 is not None else None)
                entry["ruin_month_es10"] = (
                        float(es10) if es10 is not None else None)

                entry["p5_return"] = df_data["Returns"].quantile(0.05)
                entry["p10_return"] = df_data["Returns"].quantile(0.1)

                entry["p25_return"] = df_data["Returns"].quantile(0.25)
                entry["p50_return"] = df_data["Returns"].quantile(0.5)
                run_stats["results"][m].append(entry)
        self.agg_results = run_stats


def parse_args():
    "CLI argument parser"
    prog_description = """CLI tool that invokes MC simulation across all
    portfolio models using the Long Equity and Fixed Income Ladders strategy.

    Returns a CSV file with all the simulation results.
    """
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument("-c", "--config-file",
                        help="Config file in JSON format that contains the "
                             "simulation parameters",
                        dest="config_filename",
                        required=True)
    parser.add_argument("-r", "--raw-output-file",
                        help="Destination JSON file to store the raw results "
                             "of the simulation",
                        dest="raw_output_filename",
                        required=True)
    parser.add_argument("-o", "--aggregated-output-file",
                        help="Destination JSON file to store aggregated "
                             "results of the simulation",
                        dest="agg_output_filename",
                        required=True)
    parser.add_argument("-m", "--market-cache-file",
                        help="Specifies an alternative parquet market data "
                             "cache file",
                        default="market_data.parquet",
                        dest="market_data_filename")
    return parser.parse_args()


def main():
    "Main entrypoint"
    logging.basicConfig(
        format="%(asctime)s:%(filename)s:"
               "%(lineno)d:%(levelname)s: %(message)s",
        level=logging.INFO)
    args = parse_args()
    cli = MonteCarloCLI(args.config_filename, args.market_data_filename)
    cli.run()
    with open(args.raw_output_filename, "w", encoding="utf-8") as f:
        json.dump(cli.raw_results, f, indent=4)

    cli.aggregate()
    with open(args.agg_output_filename, "w", encoding="utf-8") as f:
        json.dump(cli.agg_results, f, indent=4)


if __name__ == '__main__':
    main()
