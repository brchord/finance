"""
portfolio_simulation.py

Entry point to run a full Monte Carlo Simulation for a
given portfolio configuration specified in a JSON file.
"""
import argparse
import datetime
import json
import logging
import time
import os

import numpy as np

import portfolio_models.linear_models as lm

from market_modelling.dsvi import DynamicSVI
from market_modelling.svcj import SVCJSimulation

logger = logging.getLogger(__name__)

def run_single_path(config: dict, portfolio: lm.CombinedPortfolioStrategy):
    "Run a single path simulation and record all transactions."
    logging.info("Executing single path simulation with full book data")
    svcj = SVCJSimulation()
    spx, vix = svcj.generate_path(config.spx, config.vix, config.days)
    vix3m = svcj.derive_vix3m(vix)
    spot_nav = portfolio.run_simulation(
        spot_spx=spx,
        spot_vix=vix,
        vix3m=vix3m,
        svi=config.svi,
        initial_nav=config.nav,
        days=config.days,
        full_book=True)

    transactions = portfolio.transaction_book()

    output = {
        "nav_path": spot_nav,
        "transactions": transactions
    }

    return output


def run_backtest(config: dict, portfolio: lm.CombinedPortfolioStrategy):
    "Run a single path using historical data and record all transaction data."
    logging.info("Executing single path using historical data for "
                 "backtesting purposes.")
    with open(config["json"], encoding="utf-8") as f:
        raw_data = json.load(f)
        # Extract the time series data and only take closing values for each
        # candle.  Also pair up candles and use each candle time signature
        # to stitch together the VIX and SPX.
        spx_data = raw_data["spx"]
        vix_data = raw_data["vix"]
        vix3m_data = raw_data["vix3m"]

        def clean_time_series(price_data):
            data_dict = {}
            for c in price_data:
                t = c["t"]
                d = datetime.datetime.fromtimestamp(t / 1000)
                date_str = f"{d.year}{d.month:02d}{d.day:02d}"
                data_dict[date_str] = c["c"]
            return data_dict

        spx_dict = clean_time_series(spx_data["data"])
        vix_dict = clean_time_series(vix_data["data"])
        vix3m_dict = clean_time_series(vix3m_data["data"])

        final_set = set(spx_dict.keys()).intersection(
            set(vix_dict.keys()).intersection(set(vix3m_dict.keys())))
        sorted_days = list(final_set)
        sorted_days.sort()

        spx_final = []
        vix_final = []
        vix3m_final = []
        for d in sorted_days:
            spx_final.append(float(spx_dict[d]))
            vix_final.append(float(vix_dict[d]))
            vix3m_final.append(float(vix3m_dict[d]))

        assert len(spx_final) == len(vix_final)
        assert len(vix_final) == len(vix3m_final)

        days = config["days"]
        if len(spx_final) < days:
            raise ValueError(
                "Not enough backtest data in SPX/VIX/VIX3M to run a "
                f"{days} days simulation. Data has {len(spx_final)} "
                "points.")

        spot_nav = portfolio.run_simulation(
            spot_spx=spx_final,
            spot_vix=vix_final,
            vix3m=vix3m_final,
            svi=config["svi"],
            initial_nav=config["nav"],
            days=config["days"],
            full_book=True)

        transactions = portfolio.transaction_book()

        output = {
            "nav_path": spot_nav.tolist(),
            "transactions": transactions
        }

        return output


def run_monte_carlo(config: dict, portfolio: lm.CombinedPortfolioStrategy):
    "Run a monte carlo simulation."
    logging.info("Orchestrating Monte Carlo Simulation...")
    logging.info(config)
    logging.info(portfolio)


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
    parser.add_argument("-s", "--iv-surface",
                        help="Import a JSON file that represents real market "
                             "volatility surface data, for the file format, go to "
                             "the examples subdir in this project.",
                        dest="iv_surface_json",
                        required=True)
    parser.add_argument("-p", "--portfolio-config",
                        help="JSON file specifying the portfolio architecture "
                             "to be analyzed.  The examples subdirectory explains "
                             "the file structure this tool consumes.",
                        dest="portfolio_json",
                        required=True)
    parser.add_argument("-d, --days",
                        help="How many days per path to simulate. Default: 252",
                        type=int,
                        default=252,
                        dest="days",
                        required=True)
    parser.add_argument("-i", "--initial-nav",
                        help="NAV to start the simulation with. "
                             "Default is $1,000,000",
                        type=float,
                        default=1_000_000.00,
                        dest="initial_nav")
    parser.add_argument("-o", "--output-file",
                        help="Destination file to store simulation results in "
                             "JSON format",
                        dest="output_file",
                        required=True)
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
    parser.add_argument("-b", "--backtest",
                        help="Performs a backtest against market data specified by "
                             "the given JSON file enabling full trading book data for "
                             "further analysis. See the examples subdir for its "
                             "structure",
                        dest="backtest_json")
    core_count = os.cpu_count()
    parser.add_argument("-j", "--concurrency",
                        help="Specifies the amount of concurrency to run the simulation "
                             "under.  By default this is the number of cores the host " 
                             "machine has.",
                        type=int,
                        default=core_count,
                        dest="concurrency")
    parser.add_argument("--single-path",
                        help="Executes the investment strategy simulating only one path "
                             "with full trading book data to analyze trades and returns.",
                        action="store_true",
                        dest="single_path",
                        default=False)
    return parser.parse_args()


def main():
    "CLI entry point"
    logging.basicConfig(
        format="%(asctime)s:%(filename)s:"
               "%(lineno)d:%(levelname)s: %(message)s",
        level=logging.INFO)
    logging.info("Starting Portfolio Simulation CLI tool.")
    args = parse_args()
    log_msg = f"""Parameters for the simulation:
              IV Surface File: {args.iv_surface_json}
        Portfolio Config File: {args.portfolio_json}
                  Random Seed: {"Not specified" if args.rng_seed
                                is None else args.rng_seed }
            Paths to simulate: {args.num_paths:,}
                Backtest data: {"Not specified" if args.backtest_json
                                is None else args.backtest_json}
                  Concurrency: {args.concurrency}
       Single path simulation: {args.single_path}"""
    logging.info(log_msg)

    # Start the timer
    init_start_time = time.perf_counter()
    ivs_json = args.iv_surface_json
    p_json = args.portfolio_json

    logging.info("Attempting to load IV surface data from %s", ivs_json)
    surface_strikes = None
    surface_ivs = None
    surface_spot = None
    surface_expiration = None
    surface_atm_iv = None

    with open(ivs_json, encoding="utf-8") as f:
        iv_surface = json.load(f)
        # Convert IBKR IV string into a floating point number
        # and separate the zipped time series (Strike, IV) into
        # independent arrays.
        surface_spot = iv_surface["spot_spx"]
        surface_atm_iv = iv_surface["spot_vix"]
        today = datetime.date.today()
        exp_str = iv_surface["expiration"]
        exp_yr = int(exp_str[0:4])
        exp_m = int(exp_str[4:6])
        exp_d = int(exp_str[6:])
        expiration = datetime.date(exp_yr, exp_m, exp_d)
        surface_expiration = (expiration - today).days
        surface_expiration *= 1.0/365.0
        surface_data = iv_surface["iv_surface"]
        surface_strikes = np.zeros(len(surface_data))
        surface_ivs = np.zeros(len(surface_data))
        for i, pair in enumerate(surface_data):
            surface_strikes[i] = pair[0]
            iv = float(pair[1][:-1])
            surface_ivs[i] = iv

    svi = DynamicSVI(surface_strikes, surface_ivs, surface_spot, surface_expiration)
    logging.info("Successfully loaded IV surface volatility data.")
    logging.info("Loading portfolio geometry...")
    portfolio = None

    with open(p_json, encoding="utf-8") as f:
        json_object = json.load(f)
        portfolio = lm.CombinedPortfolioStrategy.from_json_object(json_object)

    config = {
        "nav": args.initial_nav,
        "conc": args.concurrency,
        "spot": surface_spot,
        "vix": surface_atm_iv,
        "svi": svi,
        "days": args.days
    }
    logging.info("Successfully loaded portfolio architecture")

    init_end_time = time.perf_counter()
    init_execution_time = init_end_time - init_start_time

    logging.info("Initialization took: %.2f ms", init_execution_time*1000.0)

    single_path_data = None
    if args.single_path:
        sp_start = time.perf_counter()
        single_path_data = run_single_path(config, portfolio)
        sp_end = time.perf_counter()
        sp_exec_time = sp_end - sp_start
        logging.info("Single path simulation took: %.2f ms", sp_exec_time)

    if args.backtest_json:
        config["json"] = args.backtest_json
        backtest_start = time.perf_counter()
        single_path_data = run_backtest(config, portfolio)
        backtest_end = time.perf_counter()
        backtest_time = backtest_end - backtest_start
        logging.info("Backtest path simulation took: %.2f ms", backtest_time)

    if single_path_data is not None:
        with open(args.output_file, 'w', encoding="utf-8") as f:
            json.dump(single_path_data, f)
            return 0

    run_monte_carlo(config, portfolio)
    return 0


if __name__ == "__main__":
    main()
