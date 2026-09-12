"""
path_simulation.py

Vectorized VAR(p) Real-Space Path Simulators for Monte Carlo Portfolio Analysis.
Operates on monthly real total returns [spx_real, yield_3m_real, yield_5y_real]
to capture joint cross-asset behavior and inflation dynamics upstream.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class PathSimulator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, returns_data: pd.DataFrame, levels_data: Optional[pd.DataFrame] = None) -> None:
        pass

    @abstractmethod
    def simulate_paths(
        self,
        simulation_months: int = 360,
        num_paths: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        return {}


class VARResidualBootstrapSimulator(PathSimulator):
    """
    Vectorized VAR(p) Filtered Block Bootstrap Engine in Monthly Real Space.
    Fits an unconstrained Vector Autoregression on monthly real total returns and 
    resamples empirical OLS residual blocks.

    Note: Lacks a macro valuation error-correction term (CAPE drag).
    Useful for short-to-medium horizon risk assessment; can exhibit variance 
    expansion over multi-decade horizons without structural valuation anchors.
    """

    def __init__(self, lag_order: int = 1, residual_block_size: int = 12):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=1
            Order p of the monthly VAR(p) model.
            Calibration Range: 1 to 3 monthly lags.
            Higher lags (>3) risk overfitting monthly sample size without improving macro fit.

        residual_block_size : int, default=12 (~1 calendar year)
            Block size in months for bootstrapping empirical VAR residual matrices.
            Calibration Range: 6 to 24 months (~0.5 to 2 years).
            Preserves residual autocorrelation, regime persistence, and volatility clustering.
        """
        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None

    @classmethod
    def name(cls):
        return "VARResidualBootstrapSimulator"

    def fit(self, returns_data: pd.DataFrame, levels_data: Optional[pd.DataFrame] = None) -> None:
        """
        Fits VAR(p) coefficients using OLS on 3D real monthly returns.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Matrix of monthly real total returns: 
                ['spx_real', 'yield_3m_real', 'yield_5y_real']
        levels_data : pd.DataFrame, optional
            Included for interface compatibility across path simulators.
        """
        real_returns = returns_data[["spx_real", "yield_3m_real", "yield_5y_real"]].values
        total_observations, _ = real_returns.shape
        p = self.lag_order

        target_matrix = real_returns[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(real_returns[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        self.residual_matrix = target_matrix - (design_matrix @ self.coefficient_matrix)
        self.historical_seed_matrix = real_returns[-p:]

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
        -----------
        simulation_months : int, default=360 (30 years)
            Total monthly simulation steps.
        num_paths : int, default=10000
            Number of Monte Carlo paths generated.
        seed : int, optional
            RNG seed for exact reproducibility.

        Returns:
        --------
        Dict[str, np.ndarray]:
            - 'spx_real': Array shape (num_paths, simulation_months + 1) of cumulative real wealth index.
            - 'tbill_real': Array shape (num_paths, simulation_months + 1) of 3M T-Bill real wealth index.
            - 'tnote_real': Array shape (num_paths, simulation_months + 1) of 5Y T-Note real wealth index.
            - 'simulated_returns': Array shape (num_paths, simulation_months, 3) of generated monthly real returns.
        """
        if self.coefficient_matrix is None or self.residual_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        rng = np.random.default_rng(seed)
        effective_sample_size, num_variables = self.residual_matrix.shape
        p = self.lag_order
        block_size = self.residual_block_size

        num_blocks = int(np.ceil(simulation_months / block_size))
        max_start_index = effective_sample_size - block_size
        random_block_starts = rng.integers(0, max_start_index + 1, size=(num_paths, num_blocks))

        bootstrapped_residuals = np.zeros((num_paths, num_blocks * block_size, num_variables))
        for path_idx in range(num_paths):
            sampled_blocks = [
                self.residual_matrix[start : start + block_size]
                for start in random_block_starts[path_idx]
            ]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :simulation_months, :]

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))
        simulated_real_returns = np.zeros((num_paths, simulation_months, num_variables))

        for step in range(simulation_months):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_returns = current_design_matrix @ self.coefficient_matrix + \
                bootstrapped_residuals[:, step, :]
            simulated_real_returns[:, step, :] = predicted_returns

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_returns

        # Reconstruct normalized real cumulative wealth paths (Starting at 1.0)
        spx_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 0], axis=1)])
        tbill_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 1], axis=1)])
        tnote_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 2], axis=1)])

        return {
            "spx_real": spx_real_index,
            "tbill_real": tbill_real_index,
            "tnote_real": tnote_real_index,
            "simulated_returns": simulated_real_returns,
        }


class ValuationAdjustedVARSimulator(PathSimulator):
    """
    Parametric Gaussian VAR(p) Engine with Cyclical Valuation (CAPE) Mean-Reversion in Monthly Real Space.

    Combines linear VAR transition mechanics with path-dependent valuation (CAPE) drag operating 
    on monthly real returns, assuming multivariate normal innovations N(0, Sigma_e).
    """

    def __init__(
        self,
        lag_order: int = 1,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=1
            VAR monthly lag order p. Calibration Range: 1 to 3.

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
            Estimated via OLS of 10-year forward real returns against starting ln(CAPE).

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
        self.initial_cape: float = 34.0

    @classmethod
    def name(cls):
        return "ValuationAdjustedVARSimulator"

    def fit(
        self,
        returns_data: pd.DataFrame,
        levels_data: Optional[pd.DataFrame] = None,
        initial_cape: float = 34.0,
    ) -> None:
        """
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Monthly real returns matrix: ['spx_real', 'yield_3m_real', 'yield_5y_real'].
        levels_data : pd.DataFrame, optional
            Included for interface compatibility across path simulators.
        initial_cape : float, default=34.0
            Spot Shiller CAPE ratio at the start of simulation.
            Calibration Range: Query current Yale/Shiller dataset (typically 25.0 to 38.0).
        """
        self.initial_cape = initial_cape
        real_returns = returns_data[["spx_real", "yield_3m_real", "yield_5y_real"]].values
        total_observations, _ = real_returns.shape
        p = self.lag_order

        target_matrix = real_returns[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(real_returns[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        residuals = target_matrix - (design_matrix @ self.coefficient_matrix)

        self.residual_cov_matrix = np.cov(residuals, rowvar=False)
        self.historical_seed_matrix = real_returns[-p:]

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
        -----------
        simulation_months : int, default=360
            Total monthly simulation steps.
        num_paths : int, default=10000
            Number of Monte Carlo paths generated.
        seed : int, optional
            RNG seed for exact reproducibility.

        Returns:
        --------
        Dict[str, np.ndarray]: Real wealth indices and return matrices.
        """
        if self.coefficient_matrix is None or self.residual_cov_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        rng = np.random.default_rng(seed)
        num_variables = self.residual_cov_matrix.shape[0]
        p = self.lag_order
        dt = 1.0 / 12.0  # Monthly time step

        gaussian_shocks = rng.multivariate_normal(
            mean=np.zeros(num_variables),
            cov=self.residual_cov_matrix,
            size=(simulation_months, num_paths),
        )

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))
        simulated_real_returns = np.zeros((num_paths, simulation_months, num_variables))

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        for step in range(simulation_months):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_returns = current_design_matrix @ self.coefficient_matrix + gaussian_shocks[step, :, :]

            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -self.gamma_cape * valuation_gap * dt
            predicted_returns[:, 0] += valuation_penalty

            simulated_real_returns[:, step, :] = predicted_returns

            spx_real_ret = predicted_returns[:, 0]
            log_cape += (spx_real_ret - self.earnings_growth * dt) - self.phi_cape * valuation_gap * dt

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_returns

        spx_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 0], axis=1)])
        tbill_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 1], axis=1)])
        tnote_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 2], axis=1)])

        return {
            "spx_real": spx_real_index,
            "tbill_real": tbill_real_index,
            "tnote_real": tnote_real_index,
            "simulated_returns": simulated_real_returns,
        }


class HybridValuationVARSimulator(PathSimulator):
    """
    Hybrid VECM/VAR Filtered Residual Bootstrap Engine in Monthly Real Space.

    Integrates long-horizon macro valuation mean-reversion (CAPE drag) with 
    non-parametric empirical block-bootstrapped 3D real VAR residuals.
    Preserves empirical fat tails, cross-asset correlation, and volatility clustering 
    without unconstrained variance explosion over multi-decade horizons.
    """

    def __init__(
        self,
        lag_order: int = 1,
        residual_block_size: int = 12,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=1
            VAR monthly lag order p. Calibration Range: 1 to 3.

        residual_block_size : int, default=12 (~1 calendar year)
            Block size for empirical residual block bootstrapping.
            Calibration Range: 6 to 24 months (~0.5 to 2 years).

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
        self.initial_cape: float = 34.0

    @classmethod
    def name(cls):
        return "HybridValuationVARSimulator"

    def fit(
        self,
        returns_data: pd.DataFrame,
        levels_data: Optional[pd.DataFrame] = None,
        initial_cape: float = 34.0,
    ) -> None:
        """
        Fits VAR(p) via OLS, extracts empirical real residuals, and logs initial conditions.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Monthly real returns matrix: ['spx_real', 'yield_3m_real', 'yield_5y_real'].
        levels_data : pd.DataFrame, optional
            Included for interface compatibility across path simulators.
        initial_cape : float, default=34.0
            Starting Shiller CAPE. Calibration Range: 25.0 to 38.0.
        """
        self.initial_cape = initial_cape
        real_returns = returns_data[["spx_real", "yield_3m_real", "yield_5y_real"]].values
        total_observations, _ = real_returns.shape
        p = self.lag_order

        target_matrix = real_returns[p:]
        effective_sample_size = len(target_matrix)

        design_components = [np.ones((effective_sample_size, 1))]
        for lag in range(1, p + 1):
            design_components.append(real_returns[p - lag : total_observations - lag])
        design_matrix = np.hstack(design_components)

        self.coefficient_matrix = np.linalg.pinv(
            design_matrix.T @ design_matrix) @ (design_matrix.T @ target_matrix)
        self.residual_matrix = target_matrix - (design_matrix @ self.coefficient_matrix)
        self.historical_seed_matrix = real_returns[-p:]

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
        -----------
        simulation_months : int, default=360
            Total monthly simulation steps.
        num_paths : int, default=10000
            Number of Monte Carlo paths generated.
        seed : int, optional
            RNG seed for exact reproducibility.

        Returns:
        --------
        Dict[str, np.ndarray]: Real wealth indices and return matrices.
        """
        if self.coefficient_matrix is None or self.residual_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        rng = np.random.default_rng(seed)
        effective_sample_size, num_variables = self.residual_matrix.shape
        p = self.lag_order
        block_size = self.residual_block_size
        dt = 1.0 / 12.0

        num_blocks = int(np.ceil(simulation_months / block_size))
        max_start_index = effective_sample_size - block_size
        random_block_starts = rng.integers(0, max_start_index + 1, size=(num_paths, num_blocks))

        bootstrapped_residuals = np.zeros((num_paths, num_blocks * block_size, num_variables))
        for path_idx in range(num_paths):
            sampled_blocks = [
                self.residual_matrix[start : start + block_size]
                for start in random_block_starts[path_idx]
            ]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :simulation_months, :]

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))
        simulated_real_returns = np.zeros((num_paths, simulation_months, num_variables))

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        for step in range(simulation_months):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            predicted_returns = current_design_matrix @ self.coefficient_matrix + bootstrapped_residuals[:, step, :]

            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -self.gamma_cape * valuation_gap * dt
            predicted_returns[:, 0] += valuation_penalty

            simulated_real_returns[:, step, :] = predicted_returns

            spx_real_ret = predicted_returns[:, 0]
            log_cape += (spx_real_ret - self.earnings_growth * dt) - self.phi_cape * valuation_gap * dt

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_returns

        spx_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 0], axis=1)])
        tbill_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 1], axis=1)])
        tnote_real_index = np.hstack([np.ones((num_paths, 1)), np.cumprod(1.0 + simulated_real_returns[:, :, 2], axis=1)])

        return {
            "spx_real": spx_real_index,
            "tbill_real": tbill_real_index,
            "tnote_real": tnote_real_index,
            "simulated_returns": simulated_real_returns,
        }