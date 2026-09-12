"""
monte_carlo.py

Orchestrates multi-process parallel Monte Carlo simulations for any InvestmentStrategy
operating on monthly real-space path outputs from PathSimulator instances.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import numpy as np

from market_modelling.path_simulation import PathSimulator
from portfolio_models.linear_models import InvestmentStrategy

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Orchestrates parallel Monte Carlo simulations for any InvestmentStrategy subclass.
    Encapsulates execution, chunking, and metric extraction across worker processes.
    """

    def __init__(
        self,
        strategy: InvestmentStrategy,
        simulator: PathSimulator,
        simulation_months: int = 360,
        initial_nav: float = 1_000_000.0,
        base_seed: int = 42,
    ):
        """
        Parameters:
        -----------
        strategy : InvestmentStrategy
            Portfolio model implementing `run_simulation(spx, yield3m, yield5y, initial_nav, months)`.
        simulator : PathSimulator
            Fitted path simulation engine instance inheriting from PathSimulator.
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
        strategy: InvestmentStrategy,
        path_simulator: PathSimulator,
        simulation_months: int,
        initial_nav: float,
        num_paths: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Static worker method executing a batch of paths inside an isolated process.

        Returns:
        --------
        Tuple containing:
            1. final_spx (np.ndarray): Terminal real SPX index levels (num_paths,).
            2. final_navs (np.ndarray): Terminal NAV values (num_paths,).
            3. max_drawdowns (np.ndarray): Maximum peak-to-trough drawdowns (num_paths,).
            4. path_trajectories (np.ndarray): Complete monthly NAV matrix (num_paths, simulation_months).
        """
        final_spx = np.empty(num_paths)
        final_navs = np.empty(num_paths)
        max_drawdowns = np.empty(num_paths)
        path_trajectories = np.empty((num_paths, simulation_months))

        # Generate batch real wealth index paths via simulator interface
        sim_paths: Dict[str, np.ndarray] = path_simulator.simulate_paths(
            simulation_months=simulation_months,
            num_paths=num_paths,
            seed=seed,
        )

        spx_paths = sim_paths["spx_real"]
        tbill_paths = sim_paths["tbill_real"]
        tnote_paths = sim_paths["tnote_real"]

        for i in range(num_paths):
            # Execute monthly strategy simulation
            # (Note: sim_paths include starting point at idx 0, passing monthly steps 1:)
            path_navs = strategy.run_simulation(
                spx=spx_paths[i, 1:],
                yield3m=tbill_paths[i, 1:],
                yield5y=tnote_paths[i, 1:],
                initial_nav=initial_nav,
                months=simulation_months,
                full_book=False,
            )

            path_trajectories[i, :] = path_navs
            final_spx[i] = spx_paths[i, -1]
            final_navs[i] = path_navs[-1]

            # Calculate path maximum drawdown
            peak = np.maximum.accumulate(path_navs)
            drawdowns = (peak - path_navs) / peak
            max_drawdowns[i] = -np.max(drawdowns)

        return final_spx, final_navs, max_drawdowns, path_trajectories

    def run(
        self,
        *,
        total_paths: int = 10000,
        n_workers: int = 8,
        return_trajectories: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Spawns and manages parallel execution across available CPU cores.

        Parameters:
        -----------
        total_paths : int, default=10000
            Total Monte Carlo paths to generate and evaluate.
        n_workers : int, default=8
            Number of parallel process workers in the process pool.
        return_trajectories : bool, default=False
            If True, returns full monthly NAV matrix (total_paths, simulation_months).

        Returns:
        --------
        Tuple containing (final_spx, final_navs, max_drawdowns, optional_trajectories)
        """
        chunk_size = max(100, total_paths // (n_workers * 4))
        chunks = []

        remaining_paths = total_paths
        while remaining_paths > 0:
            current_batch_size = min(chunk_size, remaining_paths)
            chunks.append(current_batch_size)
            remaining_paths -= current_batch_size

        all_final_spx = []
        all_final_navs = []
        all_max_drawdowns = []
        all_trajectories = [] if return_trajectories else None

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

            for future in as_completed(futures):
                f_spx, f_navs, m_dds, trajectories = future.result()
                all_final_spx.append(f_spx)
                all_final_navs.append(f_navs)
                all_max_drawdowns.append(m_dds)
                if return_trajectories:
                    all_trajectories.append(trajectories)

        concatenated_spx = np.concatenate(all_final_spx)
        concatenated_navs = np.concatenate(all_final_navs)
        concatenated_dds = np.concatenate(all_max_drawdowns)
        concatenated_trajectories = (
            np.vstack(all_trajectories) if return_trajectories else None
        )

        return (
            concatenated_spx,
            concatenated_navs,
            concatenated_dds,
            concatenated_trajectories,
        )