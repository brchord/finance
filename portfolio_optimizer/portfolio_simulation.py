"""
portfolio_simulation.py

Entry point to run a full Monte Carlo Simulation for a
given portfolio configuration specified in a JSON file.
"""
import argparse
import time
import os

def main():
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
                             "the given JSON file. See the examples subdir for its "
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
    
    args = parser.parse_args()
    # Start the timer
    start_time = time.perf_counter()
    end_time = time.perf_counter()
    execution_time = end_time - start_time


if __name__ == "__main__":
    main()
