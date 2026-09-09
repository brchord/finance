"""
residual_boostrap.py

Implements a Vectorized VAR Filtered Block Boostrap path simulation
for Monte Carlo portfolio simulation.
"""

import numpy as np
import pandas as pd

class VARResidualBootstrapSimulator:
    """
    Vectorized VAR(p) Filtered Block Bootstrap Engine.
    Conforms to API specification for multi-asset 3-fold path generation.
    """

    def __init__(self, lag_order: int = 2, residual_block_size: int = 21):
        """
        Parameters:
        - lag_order: Order p for VAR(p) estimation (default: 2 daily lags).
        - residual_block_size: Block size for residual resampling (default:
          21 daily trading steps ~ 1 month).
        """
        # Initial level state anchors
        self.initial_spx_level: float | None = None
        self.initial_yield_3m: float | None = None
        self.initial_yield_5y: float | None = None

        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.coefficient_matrix: np.ndarray | None = None
        self.residual_matrix: np.ndarray | None = None
        self.num_variables: int | None = None
        self.historical_seed_matrix: np.ndarray | None = None


    def fit(self, returns_data: pd.DataFrame, levels_data: pd.DataFrame) -> None:
        """
        Fits the VAR(p) model using OLS and captures the latest initial market levels.
        """
        increment_matrix = returns_data.values
        total_observations, num_variables = increment_matrix.shape
        self.num_variables = num_variables
        p = self.lag_order

        target_matrix = increment_matrix[p:]
        effective_sample_size = len(target_matrix)

        # Construct design matrix Z
        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(
                increment_matrix[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        # OLS Solution
        self.coefficient_matrix = (
            np.linalg.pinv(design_matrix.T @ design_matrix)
            @ (design_matrix.T @ target_matrix))
        self.residual_matrix = target_matrix - (design_matrix @ self.coefficient_matrix)
        self.historical_seed_matrix = increment_matrix[-p:]

        # Extract baseline starting conditions from the most recent historical observation
        self.initial_spx_level = float(levels_data["spx_close"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

    def simulate_paths(
        self,
        trading_days: int = 252,
        num_paths: int = 10000,
        seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        API-compatible simulation endpoint.
        
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (spx_paths, yield_3m_paths, yield_5y_paths) each of shape (num_paths, trading_days).
        """
        if self.coefficient_matrix is None or self.residual_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() before simulating paths.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        effective_sample_size, num_variables = self.residual_matrix.shape
        p = self.lag_order
        block_size = self.residual_block_size

        # 1. Resample empirical residual blocks
        num_blocks = int(np.ceil(trading_days / block_size))
        max_start_index = effective_sample_size - block_size

        random_block_starts = self.rng.integers(
            0, max_start_index + 1, size=(num_paths, num_blocks))

        bootstrapped_residuals = np.zeros((num_paths, num_blocks * block_size, num_variables))
        for path_idx in range(num_paths):
            sampled_blocks = [self.residual_matrix[start : start + block_size]
                              for start in random_block_starts[path_idx]]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :trading_days, :]

        # 2. Vectorized VAR forward propagation
        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))
        simulated_increments = np.zeros((num_paths, trading_days, num_variables))

        for step in range(trading_days):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_increments = (
                current_design_matrix @ self.coefficient_matrix
                + bootstrapped_residuals[:, step, :])
            simulated_increments[:, step, :] = predicted_increments

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        # 3. Integrate increments to level paths
        spx_cumulative_log_returns = np.cumsum(simulated_increments[:, :, 0], axis=1)
        yield_3m_cumulative_diffs = np.cumsum(simulated_increments[:, :, 1], axis=1)
        yield_5y_cumulative_diffs = np.cumsum(simulated_increments[:, :, 2], axis=1)

        spx_paths = self.initial_spx_level * np.exp(spx_cumulative_log_returns)
        yield_3m_paths = np.maximum(0.0, self.initial_yield_3m + yield_3m_cumulative_diffs)
        yield_5y_paths = np.maximum(0.0, self.initial_yield_5y + yield_5y_cumulative_diffs)

        return spx_paths, yield_3m_paths, yield_5y_paths
