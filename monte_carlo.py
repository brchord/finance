"""
monte_carlo.py
Orchestrates Monte Carlo simulation leveraging the local machine's
concurrency.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from portfolio_models.linear_models import InvestmentStrategy
from market_modelling.svcj import SVCJSimulation
from market_modelling.dsvi import DynamicSVI

class MonteCarloEngine:
    """
    Orchestrates parallel Monte Carlo simulations for any InvestmentStrategy
    subclass.
    
    Encapsulates execution, chunking, and metric extraction within a clean
    object-oriented structure.
    """

    def __init__(self,
                 strategy: InvestmentStrategy,
                 svi: DynamicSVI,
                 start_spx: float,
                 start_vix: float,
                 days: int,
                 initial_nav: float,
                 base_seed: int):
        self.strategy = strategy
        self.svi = svi
        self.start_spx = start_spx
        self.start_vix = start_vix
        self.days = days
        self.initial_nav = initial_nav
        self.rng = np.random.default_rng(seed=base_seed)


    @staticmethod
    def _execute_strategy_batch(
        strategy: InvestmentStrategy,
        svi: DynamicSVI,
        start_spx: float,
        start_vix: float,
        days: int,
        initial_nav: float,
        num_paths: int,
        seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Static worker method executing a batch of paths inside a separate process.
        Maintains picklability for ProcessPoolExecutor while avoiding standalone
        module-level clutter.

        Returns the following market metrics for each given path:

        1. Terminal NAV.
        2. Total return.
        3. Annualized return.
        4. Max drawdown.
        """
        final_navs = np.empty(num_paths)
        total_returns = np.empty(num_paths)
        returns_as_pct = np.empty(num_paths)
        max_drawdowns = np.empty(num_paths)

        svcj = SVCJSimulation(start_spx, start_vix)
        spx_paths, vix_paths, vix3m_paths = svcj.simulate_paths(
            days, num_paths, seed)

        for i in range(num_paths):
            path_navs = strategy.run_simulation(
                spot_spx=spx_paths[:, i],
                spot_vix=vix_paths[:, i],
                vix3m=vix3m_paths[:, i],
                initial_nav=initial_nav,
                svi=svi,
                days=days,
                full_book=False)

            final_navs[i] = path_navs[-1]
            total_returns[i] = (path_navs[-1] - initial_nav) / initial_nav
            returns_as_pct[i] = path_navs[-1] / initial_nav
            peak = np.maximum.accumulate(path_navs)
            drawdowns = (peak - path_navs) / peak
            max_drawdowns[i] = -np.max(drawdowns)

        return final_navs, total_returns, returns_as_pct, max_drawdowns


    def run(
        self, *,
        total_paths: int = 10000,
        n_workers: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Spawns and manages the parallel execution pool across available worker cores.
        """
        chunk_size = max(100, total_paths // (n_workers * 4))
        chunks = []

        remaining_paths = total_paths
        while remaining_paths > 0:
            current_batch_size = min(chunk_size, remaining_paths)
            chunks.append(current_batch_size)
            remaining_paths -= current_batch_size

        all_final_navs = []
        all_final_returns = []
        all_final_pct_of_nav = []
        all_max_drawdowns = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    MonteCarloEngine._execute_strategy_batch,
                    self.strategy,
                    self.svi,
                    self.start_spx,
                    self.start_vix,
                    self.days,
                    self.initial_nav,
                    batch_size,
                    int(self.rng.integers(1 << 31))
                )
                for batch_size in chunks
            ]
            for future in as_completed(futures):
                f_navs, returns, pct_returns, m_dds = future.result()
                all_final_navs.append(f_navs)
                all_final_returns.append(returns)
                all_final_pct_of_nav.append(pct_returns)
                all_max_drawdowns.append(m_dds)

        return np.concatenate(all_final_navs), \
            np.concatenate(all_final_returns), \
            np.concatenate(all_final_pct_of_nav), \
            np.concatenate(all_max_drawdowns)
