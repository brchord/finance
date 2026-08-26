from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from portfolio_models.linear_models import InvestmentStrategy

class MonteCarloEngine:
    """
    Orchestrates parallel Monte Carlo simulations for any InvestmentStrategy subclass.
    Encapsulates execution, chunking, and metric extraction within a clean object-oriented structure.
    """

    def __init__(self, strategy_class: type[InvestmentStrategy], init_kwargs: dict | None = None):
        self.strategy_class = strategy_class
        self.init_kwargs = init_kwargs or {}

    @staticmethod
    def _execute_strategy_batch(
        strategy_class: type[InvestmentStrategy],
        init_kwargs: dict,
        initial_nav: float,
        days: int,
        num_paths: int,
        base_seed: int | None,
        worker_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Static worker method executing a batch of paths inside a separate process.
        Maintains picklability for ProcessPoolExecutor while avoiding standalone module-level clutter.

        Returns the following market metrics for each given path:

        1. Terminal NAV.
        2. Total return.
        3. Annualized return.
        4. Max drawdown.
        """
        strategy = strategy_class(**init_kwargs)
        
        rng_seed = None if base_seed is None else base_seed + worker_id
        np.random.seed(rng_seed)
        
        final_navs = np.empty(num_paths)
        total_returns = np.empty(num_paths)
        annualized_returns = np.empty(num_paths)
        max_drawdowns = np.empty(num_paths)
        
        for i in range(num_paths):
            path_navs = strategy.run_simulation(initial_nav=initial_nav, days=days)
            
            final_navs[i] = path_navs[-1]
            total_returns[i] = (path_navs[-1] - initial_nav) / initial_nav
            annualized_returns[i] = np.pow(total_returns[i], 252.0 / days)
            peak = np.maximum.accumulate(path_navs)
            drawdowns = (peak - path_navs) / peak
            max_drawdowns[i] = np.max(drawdowns)
            
        return final_navs, total_returns, annualized_returns, max_drawdowns

    def run(
        self,
        initial_nav: float = 100000.0,
        days: int = 252,
        total_paths: int = 10000,
        n_workers: int = 8,
        base_seed: int | None = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Spawns and manages the parallel execution pool across available worker cores.
        """
        chunk_size = max(100, total_paths // (n_workers * 4))
        chunks = []
        
        remaining_paths = total_paths
        worker_id = 0
        while remaining_paths > 0:
            current_batch_size = min(chunk_size, remaining_paths)
            chunks.append((current_batch_size, worker_id))
            remaining_paths -= current_batch_size
            worker_id += 1

        all_final_navs = []
        all_final_returns = []
        all_final_ann_returns = []
        all_max_drawdowns = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    self._execute_strategy_batch,
                    self.strategy_class,
                    self.init_kwargs,
                    initial_nav,
                    days,
                    batch_size,
                    base_seed,
                    wid
                )
                for batch_size, wid in chunks
            ]

            for future in as_completed(futures):
                f_navs, returns, ann_returns, m_dds = future.result()
                all_final_navs.append(f_navs)
                all_final_returns.append(returns)
                all_final_ann_returns.append(ann_returns)
                all_max_drawdowns.append(m_dds)

        return np.concatenate(all_final_navs), \
            np.concatenate(all_final_returns), \
            np.concatenate(all_final_ann_returns), \
            np.concatenate(all_max_drawdowns)
