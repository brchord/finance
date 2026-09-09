"""
residual_boostrap.py

Implements a Vectorized VAR Filtered Block Boostrap path simulation
for Monte Carlo portfolio simulation.
"""

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

class BlockBootstrapHTMSimulator:
    """
    Non-parametric Circular Block Bootstrap Simulator for Long Equity paired with a 
    Held-To-Maturity (HTM) short-and-intermediate Treasury sleeve.

    Resamples contiguous historical return/yield blocks to preserve empirical non-linear 
    cross-asset tail dependence, jump clustering, and yield-curve shifts without 
    imposing parametric distribution assumptions.
    """

    def __init__(
        self,
        historical_data: Optional[np.ndarray] = None,
        block_size_days: int = 1260,
        initial_spx: float = 6000.0,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        historical_data : np.ndarray, shape (N, 3), optional
            Matched daily historical time-series matrix:
              - Col 0: SPX daily log returns (ln(P_t / P_{t-1}))
              - Col 1: 3-Month T-Bill rate (annualized decimal, e.g., 0.045)
              - Col 2: 5-Year T-Note yield (annualized decimal, e.g., 0.042)
            Calibration Source: FRED series ('SP500', 'DTB3', 'DGS5').
            Default: Generates a synthetic proxy if None.

        block_size_days : int, default=1260 (~5 trading years)
            Length of contiguous historical blocks resampled during simulation.
            Calibration Range: 756 to 1260 days (3 to 5 trading years).
              - Lower bounds (<504 days) destroy multi-year drawdown cycles.
              - Upper bounds (>1764 days) artificially limit block sample diversity.

        initial_spx : float, default=6000.0
            Starting spot index level for the S&P 500.
        """
        self.block_size = block_size_days
        self.initial_spx = initial_spx

        if historical_data is not None:
            if historical_data.ndim != 2 or historical_data.shape[1] != 3:
                raise ValueError("historical_data must be shape (N, 3): "
                                 "[SPX_ret, TBill_rate, TNote_yield]")
            self.data = historical_data
        else:
            rng = np.random.default_rng(42)
            n_obs = 10000
            spx_ret = rng.normal(0.00038, 0.01, n_obs)
            tbill_rate = np.clip(
                0.035 + 0.015 * np.sin(np.linspace(0, 10 * np.pi, n_obs)) +
                rng.normal(0, 0.001, n_obs), 0.0, 0.12)
            tnote_yield = np.clip(tbill_rate + 0.008 + rng.normal(0, 0.0008, n_obs), 0.002, 0.14)
            self.data = np.column_stack([spx_ret, tbill_rate, tnote_yield])

        if len(self.data) < self.block_size:
            raise ValueError("Historical data length must exceed block_size_days.")

    def simulate_paths(
        self,
        num_days: int = 17640,
        num_paths: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
        -----------
        num_days : int, default=17640 (~70 trading years)
            Total daily simulation steps.
        num_paths : int, default=10000
            Number of Monte Carlo paths generated.
        seed : int, optional
            RNG seed for exact reproducibility.

        Returns:
        --------
        Dict[str, np.ndarray]:
            - 'spx': Array shape (num_paths, num_days + 1) of SPX price levels.
            - 'tbill_yield': Array shape (num_paths, num_days + 1) of 3M rates.
            - 'tnote_yield': Array shape (num_paths, num_days + 1) of 5Y yields.
        """
        rng = np.random.default_rng(seed)
        n_obs = len(self.data)
        num_blocks = int(np.ceil(num_days / self.block_size))
        max_start_idx = n_obs - self.block_size

        start_indices = rng.integers(0, max_start_idx + 1, size=(num_paths, num_blocks))

        sampled_blocks = np.zeros((num_paths, num_blocks * self.block_size, 3))
        for p in range(num_paths):
            block_list = [
                self.data[start_indices[p, b] : start_indices[p, b] + self.block_size]
                for b in range(num_blocks)]
            sampled_blocks[p] = np.vstack(block_list)

        path_data = sampled_blocks[:, :num_days, :]

        spx_ret = path_data[:, :, 0]
        spx_paths = np.zeros((num_paths, num_days + 1))
        spx_paths[:, 0] = self.initial_spx
        spx_paths[:, 1:] = self.initial_spx * np.cumprod(1.0 + spx_ret, axis=1)

        tbill_paths = np.zeros((num_paths, num_days + 1))
        tnote_paths = np.zeros((num_paths, num_days + 1))
        tbill_paths[:, 0] = self.data[0, 1]
        tnote_paths[:, 0] = self.data[0, 2]
        tbill_paths[:, 1:] = path_data[:, :, 1]
        tnote_paths[:, 1:] = path_data[:, :, 2]

        return {"spx": spx_paths, "tbill_yield": tbill_paths, "tnote_yield": tnote_paths}


class VARResidualBootstrapSimulator:
    """
    Vectorized VAR(p) Filtered Block Bootstrap Engine.
    Fits an unconstrained Vector Autoregression on daily increments and resamples 
    empirical OLS residual blocks.

    Note: Lacks a macro valuation error-correction term (CAPE drag).
    Useful for short-horizon (1-3 yr) risk assessment; tends to exhibit variance 
    explosion over 70-year horizons.
    """

    def __init__(self, lag_order: int = 2, residual_block_size: int = 21):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=2
            Order p of the daily VAR(p) model.
            Calibration Range: 1 to 5 daily lags.
            Higher lags (>5) overfit daily microstructural noise without improving macro fit.

        residual_block_size : int, default=21 (~1 trading month)
            Block size for bootstrapping empirical VAR residual matrices.
            Calibration Range: 10 to 63 days (~2 weeks to 3 months).
            Preserves residual auto-correlation and volatility clustering.
        """
        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.initial_spx_level: Optional[float] = None
        self.initial_yield_3m: Optional[float] = None
        self.initial_yield_5y: Optional[float] = None

    def fit(self, returns_data: pd.DataFrame, levels_data: pd.DataFrame) -> None:
        """
        Fits VAR(p) coefficients using OLS.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Matrix of daily increments: 
                ['spx_log_return', 'yield_3m_diff', 'yield_5y_diff']
        levels_data : pd.DataFrame
            Matrix of daily spot levels:
                ['spx_close', 'yield_3m', 'yield_5y']
        """
        increment_matrix = returns_data.values
        total_observations, _ = increment_matrix.shape
        p = self.lag_order

        target_matrix = increment_matrix[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(increment_matrix[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        self.residual_matrix = target_matrix - (design_matrix @ self.coefficient_matrix)
        self.historical_seed_matrix = increment_matrix[-p:]

        self.initial_spx_level = float(levels_data["spx_close"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

    def simulate_paths(
        self, trading_days: int = 252, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple of arrays (spx_paths, yield_3m_paths, yield_5y_paths),
            each with shape (num_paths, trading_days).
        """
        if self.coefficient_matrix is None or self.residual_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        rng = np.random.default_rng()
        if seed is not None:
            rng = np.random.default_rng(seed)

        effective_sample_size, num_variables = self.residual_matrix.shape
        p = self.lag_order
        block_size = self.residual_block_size

        num_blocks = int(np.ceil(trading_days / block_size))
        max_start_index = effective_sample_size - block_size
        random_block_starts = rng.integers(0, max_start_index + 1, size=(num_paths, num_blocks))

        bootstrapped_residuals = np.zeros((num_paths, num_blocks * block_size, num_variables))
        for path_idx in range(num_paths):
            sampled_blocks = [
                self.residual_matrix[start : start + block_size]
                for start in random_block_starts[path_idx]]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :trading_days, :]

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))
        simulated_increments = np.zeros((num_paths, trading_days, num_variables))

        for step in range(trading_days):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_increments = current_design_matrix @ self.coefficient_matrix + \
                bootstrapped_residuals[:, step, :]
            simulated_increments[:, step, :] = predicted_increments

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        spx_cum_log_ret = np.cumsum(simulated_increments[:, :, 0], axis=1)
        yield_3m_cum_diff = np.cumsum(simulated_increments[:, :, 1], axis=1)
        yield_5y_cum_diff = np.cumsum(simulated_increments[:, :, 2], axis=1)

        spx_paths = self.initial_spx_level * np.exp(spx_cum_log_ret)
        yield_3m_paths = np.maximum(0.0, self.initial_yield_3m + yield_3m_cum_diff)
        yield_5y_paths = np.maximum(0.0, self.initial_yield_5y + yield_5y_cum_diff)

        return spx_paths, yield_3m_paths, yield_5y_paths


class ValuationAdjustedVARSimulator:
    """
    Parametric Gaussian VAR(p) Engine with Cyclical Valuation (CAPE) Mean-Reversion.

    Combines linear VAR transition mechanics with path-dependent valuation (CAPE) drag,
    assuming multivariate normal innovations N(0, Sigma_e).
    """

    def __init__(
        self,
        lag_order: int = 2,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=2
            VAR daily lag order p. Calibration Range: 1 to 5.

        target_cape : float, default=22.0
            Long-term equilibrium Shiller CAPE median.
            Calibration Range: 18.0 to 24.0.
              - 18.0 reflects the full 150-year historical U.S. equity median.
              - 24.0 reflects the modern post-1990 high-margin/low-cost capital baseline.

        cape_reversion_speed : float (phi), default=0.05
            Annual speed of CAPE mean-reversion toward target_cape.
            Calibration Range: 0.03 to 0.08.
              - 0.03 implies a ~23-year valuation half-life.
              - 0.08 implies an ~8.5-year valuation half-life.

        valuation_drag_coef : float (gamma), default=0.015
            Annual equity return drag per unit of log-valuation gap:
            Penalty = -gamma * (ln(CAPE_t) - ln(CAPE*)).
            Calibration Range: 0.010 to 0.025.
            Estimated via OLS of 10-year forward returns against starting ln(CAPE).

        annual_earnings_growth : float, default=0.02
            Expected annual real earnings growth rate.
            Calibration Range: 0.015 to 0.025 (1.5% to 2.5% real growth).
        """
        self.lag_order = lag_order
        self.target_cape = target_cape
        self.phi_cape = cape_reversion_speed
        self.gamma_cape = valuation_drag_coef
        self.earnings_growth = annual_earnings_growth

        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_cov_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.initial_spx_level: Optional[float] = None
        self.initial_yield_3m: Optional[float] = None
        self.initial_yield_5y: Optional[float] = None
        self.initial_cape: float = 34.0

    def fit(
        self,
        returns_data: pd.DataFrame,
        levels_data: pd.DataFrame,
        initial_cape: float = 34.0,
    ) -> None:
        """
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily increments matrix.
        levels_data : pd.DataFrame
            Daily levels matrix.
        initial_cape : float, default=34.0
            Spot Shiller CAPE ratio at the start of simulation.
            Calibration Range: Query current Yale/Shiller dataset (typically 25.0 to 38.0).
        """
        self.initial_cape = initial_cape
        increment_matrix = returns_data.values
        total_observations, _ = increment_matrix.shape
        p = self.lag_order

        target_matrix = increment_matrix[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(increment_matrix[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        residuals = target_matrix - (design_matrix @ self.coefficient_matrix)

        self.residual_cov_matrix = np.cov(residuals, rowvar=False)
        self.historical_seed_matrix = increment_matrix[-p:]

        self.initial_spx_level = float(levels_data["spx_close"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

    def simulate_paths(
        self, trading_days: int = 252, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See InvestmentStrategy for documentation."""
        if self.coefficient_matrix is None or self.residual_cov_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        rng = np.random.default_rng(seed)
        num_variables = self.residual_cov_matrix.shape[0]
        p = self.lag_order
        dt = 1.0 / 252.0

        gaussian_shocks = rng.multivariate_normal(
            mean=np.zeros(num_variables),
            cov=self.residual_cov_matrix,
            size=(trading_days, num_paths))

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))

        spx_paths = np.zeros((num_paths, trading_days))
        yield_3m_paths = np.zeros((num_paths, trading_days))
        yield_5y_paths = np.zeros((num_paths, trading_days))

        curr_spx = np.full(num_paths, self.initial_spx_level)
        curr_3m = np.full(num_paths, self.initial_yield_3m)
        curr_5y = np.full(num_paths, self.initial_yield_5y)

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        for step in range(trading_days):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_increments = current_design_matrix @ self.coefficient_matrix + \
                gaussian_shocks[step, :, :]

            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -self.gamma_cape * valuation_gap * dt
            predicted_increments[:, 0] += valuation_penalty

            spx_log_ret = predicted_increments[:, 0]
            diff_3m = predicted_increments[:, 1]
            diff_5y = predicted_increments[:, 2]

            curr_spx *= np.exp(spx_log_ret)
            curr_3m = np.maximum(0.0, curr_3m + diff_3m)
            curr_5y = np.maximum(0.0, curr_5y + diff_5y)

            log_cape += (spx_log_ret - self.earnings_growth * dt) - \
                self.phi_cape * valuation_gap * dt

            spx_paths[:, step] = curr_spx
            yield_3m_paths[:, step] = curr_3m
            yield_5y_paths[:, step] = curr_5y

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        return spx_paths, yield_3m_paths, yield_5y_paths


class HybridValuationVARSimulator:
    """
    Hybrid VECM/VAR Filtered Residual Bootstrap Engine.

    Integrates long-horizon macro valuation mean-reversion (CAPE drag) with 
    non-parametric empirical block-bootstrapped VAR residuals.
    Preserves empirical fat tails, cross-asset correlation, and volatility clustering 
    without unconstrained variance explosion over multi-decade horizons.
    """

    def __init__(
        self,
        lag_order: int = 2,
        residual_block_size: int = 21,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=2
            VAR daily lag order p. Calibration Range: 1 to 5.

        residual_block_size : int, default=21 (~1 trading month)
            Block size for empirical residual block bootstrapping.
            Calibration Range: 10 to 63 steps (~2 weeks to 3 months).

        target_cape : float, default=22.0
            Long-term equilibrium Shiller CAPE median. Calibration Range: 18.0 to 24.0.

        cape_reversion_speed : float (phi), default=0.05
            Annual CAPE mean-reversion speed. Calibration Range: 0.03 to 0.08.

        valuation_drag_coef : float (gamma), default=0.015
            Annual equity return penalty per unit log-valuation gap.
            Calibration Range: 0.010 to 0.025.

        annual_earnings_growth : float, default=0.02
            Expected annual real baseline earnings growth. Calibration Range: 0.015 to 0.025.
        """
        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.target_cape = target_cape
        self.phi_cape = cape_reversion_speed
        self.gamma_cape = valuation_drag_coef
        self.earnings_growth = annual_earnings_growth

        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.initial_spx_level: Optional[float] = None
        self.initial_yield_3m: Optional[float] = None
        self.initial_yield_5y: Optional[float] = None
        self.initial_cape: float = 34.0

    def fit(
        self,
        returns_data: pd.DataFrame,
        levels_data: pd.DataFrame,
        initial_cape: float = 34.0,
    ) -> None:
        """
        Fits VAR(p) via OLS, extracts empirical residuals, and logs initial conditions.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily increments matrix.
        levels_data : pd.DataFrame
            Daily levels matrix.
        initial_cape : float, default=34.0
            Starting Shiller CAPE. Calibration Range: 25.0 to 38.0.
        """
        self.initial_cape = initial_cape
        increment_matrix = returns_data.values
        total_observations, _ = increment_matrix.shape
        p = self.lag_order

        target_matrix = increment_matrix[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(increment_matrix[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        self.residual_matrix = target_matrix - (design_matrix @ self.coefficient_matrix)
        self.historical_seed_matrix = increment_matrix[-p:]

        self.initial_spx_level = float(levels_data["spx_close"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

    def simulate_paths(
        self, trading_days: int = 252, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See InvestmentStrategy for documentation."""
        if self.coefficient_matrix is None or self.residual_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        effective_sample_size, num_variables = self.residual_matrix.shape
        p = self.lag_order
        block_size = self.residual_block_size
        dt = 1.0 / 252.0

        num_blocks = int(np.ceil(trading_days / block_size))
        max_start_index = effective_sample_size - block_size
        random_block_starts = np.random.randint(
            0, max_start_index + 1, size=(num_paths, num_blocks))
        bootstrapped_residuals = np.zeros((num_paths, num_blocks * block_size, num_variables))
        for path_idx in range(num_paths):
            sampled_blocks = [self.residual_matrix[start : start + block_size]
                              for start in random_block_starts[path_idx]]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :trading_days, :]

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))

        spx_paths = np.zeros((num_paths, trading_days))
        yield_3m_paths = np.zeros((num_paths, trading_days))
        yield_5y_paths = np.zeros((num_paths, trading_days))

        curr_spx = np.full(num_paths, self.initial_spx_level)
        curr_3m = np.full(num_paths, self.initial_yield_3m)
        curr_5y = np.full(num_paths, self.initial_yield_5y)

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        for step in range(trading_days):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_increments = current_design_matrix @ self.coefficient_matrix + \
                bootstrapped_residuals[:, step, :]

            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -self.gamma_cape * valuation_gap * dt
            predicted_increments[:, 0] += valuation_penalty

            spx_log_ret = predicted_increments[:, 0]
            diff_3m = predicted_increments[:, 1]
            diff_5y = predicted_increments[:, 2]

            curr_spx *= np.exp(spx_log_ret)
            curr_3m = np.maximum(0.0, curr_3m + diff_3m)
            curr_5y = np.maximum(0.0, curr_5y + diff_5y)

            log_cape += (spx_log_ret - self.earnings_growth * dt) - \
                self.phi_cape * valuation_gap * dt

            spx_paths[:, step] = curr_spx
            yield_3m_paths[:, step] = curr_3m
            yield_5y_paths[:, step] = curr_5y

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        return spx_paths, yield_3m_paths, yield_5y_paths
