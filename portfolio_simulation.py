"""
portfolio_simulation.py

Entry point to run a full Monte Carlo Simulation for a
given portfolio configuration specified in a JSON file.
"""
import argparse
import json
import logging
import sys
import time
import os

import numpy as np

import market_data.yf_fred_market_data as md
import monte_carlo as mc
import portfolio_models.linear_models as lm

from market_modelling.path_simulation import VARResidualBootstrapSimulator

logger = logging.getLogger(__name__)

def run_single_path(config: dict, portfolio: lm.CombinedPortfolioStrategy):
    "Run a single path simulation and record all transactions."
    logging.info("Executing single path simulation with full book data")
    sim: VARResidualBootstrapSimulator = config["sim"]
    if "seed" in config:
        spx_path, yield3m_path, yield5y_path = sim.simulate_paths(
            config["days"], 1, config["seed"])
    else:
        spx_path, yield3m_path, yield5y_path = sim.simulate_paths(
            config["days"], 1)

    spot_nav = portfolio.run_simulation(
        spx=spx_path[0, :],
        yield3m=yield3m_path[0, :],
        yield5y=yield5y_path[0, :],
        initial_nav=config["nav"],
        days=config["days"],
        full_book=True)

    transactions = portfolio.transaction_book()

    output = {
        "nav_path": spot_nav.tolist(),
        "spx": spx_path[0, :].tolist(),
        "yield3m": yield3m_path[0, :].tolist(),
        "yield5m": yield5y_path[0, :].tolist(),
        "transactions": transactions
    }

    return output


def parse_args():
    "Parses tool's command line arguments."
    prog_description = """This CLI tool examines a specified
    investment portfolio with a gamma of investment strategies and
    is capable or performing analysis by leveraging Monte Carlo simulation.

    The CLI will take a number of input files that both provide data for
    volatility surface calibration in the case of investment strategies
    that involve options, the ability to feed historical market data for
    backtesting purposes and a combination of provided strategies to
    construct a portfolio under analysis.
    """
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument("-p", "--portfolio-config",
                        help="JSON file specifying the portfolio architecture "
                             "to be analyzed.  The examples subdirectory explains "
                             "the file structure this tool consumes.",
                        dest="portfolio_json",
                        required=True)
    parser.add_argument("-o", "--output-file",
                        help="Destination file to store simulation results in "
                             "JSON format",
                        dest="output_file",
                        required=True)
    parser.add_argument("-d", "--days",
                        help="How many days per path to simulate. Default: 252",
                        type=int,
                        default=252,
                        dest="days")
    parser.add_argument("-i", "--initial-nav",
                        help="NAV to start the simulation with. "
                             "Default is $1,000,000",
                        type=float,
                        default=1_000_000.00,
                        dest="initial_nav")
    parser.add_argument("-r", "--random-seed",
                        help="Random number seed used to get consistent path "
                             "simulations across different runs.",
                        type=int,
                        dest="rng_seed")
    parser.add_argument("-n", "--paths",
                        help="Number of paths to simulate to run under for "
                             "the Monte Carlo simulation. Default is 10k paths.",
                        type=int,
                        default=10000,
                        dest="num_paths")
    core_count = os.cpu_count()
    parser.add_argument("-j", "--concurrency",
                        help="Specifies the amount of concurrency to run the simulation "
                             "under.  By default this is the number of cores the host " 
                             "machine has.",
                        type=int,
                        default=core_count,
                        dest="concurrency")
    parser.add_argument("-s", "--single-path",
                        help="Executes the investment strategy simulating only one path "
                             "with full trading book data to analyze trades and returns.",
                        action="store_true",
                        dest="single_path",
                        default=False)
    parser.add_argument("-l", "--log-level",
                        help="Logging level: DEBUG | INFO | WARNING | ERROR",
                        dest="log_level",
                        default="INFO")
    parser.add_argument("-c", "--data-cache",
                        help="Filename to persist cached market data",
                        dest="data_cache",
                        default="market_data.parquet")
    return parser.parse_args()


def main():
    "CLI entry point"
    args = parse_args()
    log_level = args.log_level.upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING
    }

    if log_level not in level_map:
        logging.error("Invalid log level: %s", log_level)
        sys.exit(2)

    logging.basicConfig(
        format="%(asctime)s:%(filename)s:"
               "%(lineno)d:%(levelname)s: %(message)s",
        level=level_map[log_level])
    logging.info("Starting Portfolio Simulation CLI tool.")

    log_msg = f"""Parameters for the simulation:
        Portfolio Config File: {args.portfolio_json}
                  Random Seed: {"Not specified" if args.rng_seed
                                is None else args.rng_seed }
            Paths to simulate: {args.num_paths:,}
                  Concurrency: {args.concurrency}
       Single path simulation: {args.single_path}"""
    logging.info(log_msg)

    # Start the timer
    init_start_time = time.perf_counter()
    p_json = args.portfolio_json

    logging.info("Loading portfolio geometry...")
    portfolio = None

    with open(p_json, encoding="utf-8") as f:
        json_object = json.load(f)
        for strategy in [lm.CombinedPortfolioStrategy,
                         lm.LongSPYStrategy,
                         lm.LongSPYWithTreasuryLadders,
                         lm.FixedIncomeStrategy]:
            portfolio = strategy.from_json_object(json_object)
            if portfolio is not None:
                break
        if portfolio is None:
            raise ValueError("Invalid portfolio specification "
                             f"from file: {p_json}")

    data_manager = md.MarketDataManager(cache_filepath=args.data_cache)
    market_levels, market_returns = data_manager.get_aligned_data(force_refresh=False)

    logging.info("Successfully loaded portfolio architecture")

    simulator = VARResidualBootstrapSimulator()
    simulator.fit(returns_data=market_returns, levels_data=market_levels)

    config = {
        "nav": args.initial_nav,
        "conc": args.concurrency,
        "days": args.days,
        "sim": simulator
    }

    init_end_time = time.perf_counter()
    init_execution_time = init_end_time - init_start_time

    logging.info("Initialization took: %.2f ms", init_execution_time*1000.0)

    single_path_data = None
    if args.single_path:
        if args.rng_seed is not None:
            config["seed"] = args.rng_seed
        sp_start = time.perf_counter()
        single_path_data = run_single_path(config, portfolio)
        sp_end = time.perf_counter()
        sp_exec_time = sp_end - sp_start
        logging.info("Single path simulation took: %.2f ms", sp_exec_time)

    if single_path_data is not None:
        with open(args.output_file, 'w', encoding="utf-8") as f:
            json.dump(single_path_data, f)
            return 0

    logging.info("Orchestrating Monte Carlo Simulation...")
    seed = np.random.default_rng().random()
    if args.rng_seed is not None:
        seed = args.rng_seed

    mcs = mc.MonteCarloEngine(portfolio, simulator, args.days, args.initial_nav, seed)

    mc_start = time.perf_counter()
    spxs, navs, returns, returns_pct, max_dds = mcs.run(
        total_paths=args.num_paths, n_workers=config["conc"])
    mc_end = time.perf_counter()
    total_time = float(mc_end - mc_start)
    logging.info("Monte Carlo simulation completed.  Took: %2fs", total_time)
    with open(args.output_file, "w", encoding=" utf-8") as f:
        print('"Terminal SPX", "Terminal NAV", "Total Return", '
              '"Pct of Initial NAV", "Max Drawdown"', file=f)
        for i, nav in enumerate(navs):
            output_row = f"{spxs[i]:.2f}, {nav:.2f}, {returns[i]:.2f}, " + \
                         f"{returns_pct[i]:.2f}, {max_dds[i]:.2f}"
            print(output_row, file=f)

    return 0


if __name__ == "__main__":
    main()
