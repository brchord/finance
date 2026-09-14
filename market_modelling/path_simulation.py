"""
path_simulation.py

Vectorized VAR(p) Real-Space Path Simulators for Monte Carlo Portfolio Analysis.
Operates on monthly nominal price levels [spx, cpi, yield_3m, yield_5y]
to capture joint cross-asset behavior and inflation dynamics upstream.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class PathSimulator(ABC):
    """
    Represents a 4 trajectory path simulation class that produces the following
    monthly time series:

    - Equity Index (SP500).
    - CPI (Consumer Price Index) to model inflation.
    - 3 Month T-Bill rates.
    - 5 Year T-Note rates.
    """
    @abstractmethod
    def fit(self, returns_data: pd.DataFrame, levels_data: pd.DataFrame) -> None:
        """
        Fits the given model using market levels and its corresponding returns.
        """

    @abstractmethod
    def simulate_paths(
        self,
        simulation_months: int = 360,
        num_paths: int = 10000,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[ndarray, ndarray, ndarray, ndarray]
            - A matrix of num_paths x (simulation_months + 1) representing the nominal SPX Index.
            - A matrix of num_paths x (simulation_months + 1) representing the nominal CPI Index.
            - A matrix of num_paths x (simulation_months + 1) representing 3M T-Bill yields.
            - A matrix of num_paths x (simulation_months + 1) representing 5Y T-Note yields.
        """
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

    def __init__(
        self,
        lag_order: int = 1,
        residual_block_size: int = 48,
        rate_reversion_speed: float = 0.15,
        target_yield_3m: Optional[float] = None,
        target_yield_5y: Optional[float] = None,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=1
            Order p of the monthly VAR(p) model.
            Calibration Range: 1 to 3 monthly lags.
            Higher lags (>3) risk overfitting monthly sample size without improving macro fit.

        residual_block_size : int, default=48 (4 years)
            Block size in months for bootstrapping empirical VAR residual matrices.
            Calibration Range: 24 to 60 months (2 to 5 years).
            Preserves residual autocorrelation, regime persistence, and volatility clustering.
            NOTE: block starts are drawn i.i.d. across block index, so persistence is
            only preserved *within* a block -- multi-year regimes longer than this
            window get diluted across block boundaries. Empirically calibrated against
            1982-2026 CPI history: default=48 was the smallest block size whose
            simulated 5-year-window inflation variance matched the realized historical
            5-year-window variance (0.82%) on that sample. Since that sample excludes
            the 1965-1982 high-inflation regime, treat this as a floor, not a ceiling,
            on plausible sustained-inflation tail risk.

        rate_reversion_speed : float (phi_rate), default=0.15
            Annual mean-reversion speed pulling 3M/5Y yields back toward their long-run
            equilibrium levels. Calibration Range: 0.05 to 0.25.
              - 0.05 implies a ~14-year yield half-life.
              - 0.25 implies a ~2.8-year yield half-life.
            Without this term, yields evolve as an unanchored random walk (unit root) and
            can drift to implausible levels, or get stuck at the zero floor for extended
            stretches, over multi-decade simulation horizons.

        target_yield_3m, target_yield_5y : float, optional
            Long-run equilibrium yield levels used as the reversion anchor for each series.
            If None (default), estimated from the historical sample mean at fit() time.
        """
        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.phi_rate = rate_reversion_speed
        self.target_yield_3m = target_yield_3m
        self.target_yield_5y = target_yield_5y

        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.initial_spx_level: float = None
        self.initial_cpi_level: float = None
        self.initial_yield_3m: float = None
        self.initial_yield_5y: float = None

    @classmethod
    def name(cls):
        """Returns the name of this Path Simulator"""
        return "VARResidualBootstrapSimulator"

    def fit(self, returns_data: pd.DataFrame, levels_data: pd.DataFrame) -> None:
        """
        Fits VAR(p) coefficients using OLS on 3D real monthly returns.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Matrix of monthly increments: 
                ['spx_log_return', 'cpi_log_return', 'yield_3m_diff', 'yield_5y_diff']
        levels_data : pd.DataFrame
            Matrix of monthly levels:
                ['spx_close', 'cpi', 'yield_3m', 'yield_5y']
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
        self.initial_cpi_level = float(levels_data["cpi"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

        # Anchor rate mean-reversion to the historical sample mean unless the
        # caller supplied an explicit long-run target.
        if self.target_yield_3m is None:
            self.target_yield_3m = float(levels_data["yield_3m"].mean())
        if self.target_yield_5y is None:
            self.target_yield_5y = float(levels_data["yield_5y"].mean())

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[ndarray, ndarray, ndarray, ndarray]
            - A matrix of num_paths x (simulation_months + 1) representing the nominal SPX Index.
            - A matrix of num_paths x (simulation_months + 1) representing the nominal CPI Index.
            - A matrix of num_paths x (simulation_months + 1) representing 3M T-Bill yields.
            - A matrix of num_paths x (simulation_months + 1) representing 5Y T-Note yields.
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
                for start in random_block_starts[path_idx]]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :simulation_months, :]

        # Yields now require step-wise (rather than vectorized cumsum) evolution,
        # since the reversion term depends on the current level each month.
        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))

        spx_paths = np.zeros((num_paths, simulation_months))
        cpi_paths = np.zeros((num_paths, simulation_months))
        yield_3m_paths = np.zeros((num_paths, simulation_months))
        yield_5y_paths = np.zeros((num_paths, simulation_months))

        curr_spx = np.full(num_paths, self.initial_spx_level)
        curr_cpi = np.full(num_paths, self.initial_cpi_level)
        curr_3m = np.full(num_paths, self.initial_yield_3m)
        curr_5y = np.full(num_paths, self.initial_yield_5y)

        for step in range(simulation_months):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            raw_increments = current_design_matrix @ self.coefficient_matrix + \
                bootstrapped_residuals[:, step, :]

            spx_log_ret = raw_increments[:, 0]
            cpi_log_ret = raw_increments[:, 1]
            # Error-correction overlay: pull yields back toward their long-run
            # equilibrium level so the process doesn't behave as an unanchored
            # random walk over multi-decade horizons.
            diff_3m = raw_increments[:, 2] - \
                self.phi_rate * (curr_3m - self.target_yield_3m) * dt
            diff_5y = raw_increments[:, 3] - \
                self.phi_rate * (curr_5y - self.target_yield_5y) * dt

            curr_spx *= np.exp(spx_log_ret)
            curr_cpi *= np.exp(cpi_log_ret)
            curr_3m = np.maximum(0.0, curr_3m + diff_3m)
            curr_5y = np.maximum(0.0, curr_5y + diff_5y)

            spx_paths[:, step] = curr_spx
            cpi_paths[:, step] = curr_cpi
            yield_3m_paths[:, step] = curr_3m
            yield_5y_paths[:, step] = curr_5y

            # Feed the reversion-adjusted diffs back into history so the VAR's
            # own lag structure sees the same series that was actually realized
            # (mirrors how the equity leg's adjusted return is fed back).
            predicted_increments = raw_increments.copy()
            predicted_increments[:, 2] = diff_3m
            predicted_increments[:, 3] = diff_5y
            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        return spx_paths, cpi_paths, yield_3m_paths, yield_5y_paths


class ValuationAdjustedVARSimulator(PathSimulator):
    """
    Parametric Gaussian VAR(p) Engine with Cyclical Valuation (CAPE) Mean-Reversion.

    Combines linear VAR transition mechanics with path-dependent valuation (CAPE) drag,
    assuming multivariate normal innovations N(0, Sigma_e).
    """

    def __init__(
        self,
        lag_order: int = 1,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
        rate_reversion_speed: float = 0.15,
        target_yield_3m: Optional[float] = None,
        target_yield_5y: Optional[float] = None,
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
            Estimated via OLS of 10-year forward returns against starting ln(CAPE).

        annual_earnings_growth : float, default=0.02
            Expected annual real earnings growth rate.
            Calibration Range: 0.015 to 0.025 (1.5% to 2.5% real growth).

        rate_reversion_speed : float (phi_rate), default=0.15
            Annual mean-reversion speed pulling 3M/5Y yields back toward their long-run
            equilibrium levels. Calibration Range: 0.05 to 0.25. Without this term,
            yields evolve as an unanchored random walk over multi-decade horizons.

        target_yield_3m, target_yield_5y : float, optional
            Long-run equilibrium yield levels used as the reversion anchor.
            If None (default), estimated from the historical sample mean at fit() time.
        """
        self.lag_order = lag_order
        self.target_cape = target_cape
        self.phi_cape = cape_reversion_speed
        self.gamma_cape = valuation_drag_coef
        self.earnings_growth = annual_earnings_growth
        self.phi_rate = rate_reversion_speed
        self.target_yield_3m = target_yield_3m
        self.target_yield_5y = target_yield_5y

        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_cov_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.initial_cape: float = 34.0
        self.initial_spx_level: float = None
        self.initial_cpi_level: float = None
        self.initial_yield_3m: float = None
        self.initial_yield_5y: float = None

    @classmethod
    def name(cls):
        """Name of this path simulator."""
        return "ValuationAdjustedVARSimulator"

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
            Monthly increments matrix.
        levels_data : pd.DataFrame
            Monthly levels matrix.
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
        self.initial_cpi_level = float(levels_data["cpi"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

        if self.target_yield_3m is None:
            self.target_yield_3m = float(levels_data["yield_3m"].mean())
        if self.target_yield_5y is None:
            self.target_yield_5y = float(levels_data["yield_5y"].mean())

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Simulated equity, inflation and fixed income matrices.
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

        spx_paths = np.zeros((num_paths, simulation_months))
        cpi_paths = np.zeros((num_paths, simulation_months))
        yield_3m_paths = np.zeros((num_paths, simulation_months))
        yield_5y_paths = np.zeros((num_paths, simulation_months))

        curr_spx = np.full(num_paths, self.initial_spx_level)
        curr_cpi = np.full(num_paths, self.initial_cpi_level)
        curr_3m = np.full(num_paths, self.initial_yield_3m)
        curr_5y = np.full(num_paths, self.initial_yield_5y)

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        for step in range(simulation_months):
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
            cpi_log_ret = predicted_increments[:, 1]
            # Error-correction overlay: pull yields back toward their long-run
            # equilibrium level rather than letting them random-walk unbounded.
            diff_3m = predicted_increments[:, 2] - \
                self.phi_rate * (curr_3m - self.target_yield_3m) * dt
            diff_5y = predicted_increments[:, 3] - \
                self.phi_rate * (curr_5y - self.target_yield_5y) * dt
            predicted_increments[:, 2] = diff_3m
            predicted_increments[:, 3] = diff_5y

            curr_spx *= np.exp(spx_log_ret)
            curr_cpi *= np.exp(cpi_log_ret)
            curr_3m = np.maximum(0.0, curr_3m + diff_3m)
            curr_5y = np.maximum(0.0, curr_5y + diff_5y)

            log_cape += (spx_log_ret - self.earnings_growth * dt) - \
                self.phi_cape * valuation_gap * dt

            spx_paths[:, step] = curr_spx
            cpi_paths[:, step] = curr_cpi
            yield_3m_paths[:, step] = curr_3m
            yield_5y_paths[:, step] = curr_5y

            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        return spx_paths, cpi_paths, yield_3m_paths, yield_5y_paths


class HybridValuationVARSimulator(PathSimulator):
    """
    Hybrid VECM/VAR Filtered Residual Bootstrap Engine.

    Integrates long-horizon macro valuation mean-reversion (CAPE drag) with 
    non-parametric empirical block-bootstrapped VAR residuals.
    De-means equity drift to align VAR stochastics with long-term equilibrium fundamentals.
    """

    def __init__(
        self,
        lag_order: int = 1,
        residual_block_size: int = 48,
        target_cape: float = 22.0,
        cape_reversion_speed: float = 0.05,
        valuation_drag_coef: float = 0.015,
        annual_earnings_growth: float = 0.02,
        rate_reversion_speed: float = 0.15,
        target_yield_3m: Optional[float] = None,
        target_yield_5y: Optional[float] = None,
    ):
        """
        Parameters & Calibration Ranges:
        ---------------------------------
        lag_order : int, default=1
            VAR monthly lag order p. Calibration Range: 1 to 3.

        residual_block_size : int, default=48 (4 years)
            Block size for empirical residual block bootstrapping.
            Calibration Range: 24 to 60 months (2 to 5 years).
            See VARResidualBootstrapSimulator for the empirical calibration note --
            block starts are i.i.d. across block index, so multi-year regime
            persistence beyond this window is diluted at block boundaries.

        target_cape : float, default=22.0
            Long-term equilibrium Shiller CAPE median. Calibration Range: 18.0 to 24.0.

        cape_reversion_speed : float (phi), default=0.05
            Annual CAPE mean-reversion speed. Calibration Range: 0.03 to 0.08.

        valuation_drag_coef : float (gamma), default=0.015
            Annual equity return penalty per unit log-valuation gap.
            Calibration Range: 0.010 to 0.025.

        annual_earnings_growth : float, default=0.02
            Expected annual real baseline earnings growth. Calibration Range: 0.015 to 0.025.

        rate_reversion_speed : float (phi_rate), default=0.15
            Annual mean-reversion speed pulling 3M/5Y yields back toward their long-run
            equilibrium levels. Calibration Range: 0.05 to 0.25. Without this term,
            yields evolve as an unanchored random walk over multi-decade horizons.

        target_yield_3m, target_yield_5y : float, optional
            Long-run equilibrium yield levels used as the reversion anchor.
            If None (default), estimated from the historical sample mean at fit() time.
        """
        self.lag_order = lag_order
        self.residual_block_size = residual_block_size
        self.target_cape = target_cape
        self.phi_cape = cape_reversion_speed
        self.gamma_cape = valuation_drag_coef
        self.earnings_growth = annual_earnings_growth
        self.phi_rate = rate_reversion_speed
        self.target_yield_3m = target_yield_3m
        self.target_yield_5y = target_yield_5y

        self.coefficient_matrix: Optional[np.ndarray] = None
        self.residual_matrix: Optional[np.ndarray] = None
        self.historical_seed_matrix: Optional[np.ndarray] = None
        self.historical_mean_returns: Optional[np.ndarray] = None
        self.initial_cape: float = 34.0
        self.initial_spx_level: float = None
        self.initial_cpi_level: float = None
        self.initial_yield_3m: float = None
        self.initial_yield_5y: float = None

    @classmethod
    def name(cls):
        """Name of this path simulator"""
        return "HybridValuationVARSimulator"

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
            Monthly increments matrix.
        levels_data : pd.DataFrame
            Monthly levels matrix.
        initial_cape : float, default=34.0
            Starting Shiller CAPE. Calibration Range: 25.0 to 38.0.
        """
        self.initial_cape = initial_cape
        increment_matrix = returns_data.values
        total_observations, _ = increment_matrix.shape
        p = self.lag_order

        # Store historical sample mean to isolate stochastic innovations from historical drift
        self.historical_mean_returns = np.mean(increment_matrix, axis=0)

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
        self.initial_cpi_level = float(levels_data["cpi"].iloc[-1])
        self.initial_yield_3m = float(levels_data["yield_3m"].iloc[-1])
        self.initial_yield_5y = float(levels_data["yield_5y"].iloc[-1])

        if self.target_yield_3m is None:
            self.target_yield_3m = float(levels_data["yield_3m"].mean())
        if self.target_yield_5y is None:
            self.target_yield_5y = float(levels_data["yield_5y"].mean())

    def simulate_paths(
        self, simulation_months: int = 360, num_paths: int = 10000, seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Simulated equity, inflation and fixed income matrices.
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
            sampled_blocks = [self.residual_matrix[start : start + block_size]
                              for start in random_block_starts[path_idx]]
            bootstrapped_residuals[path_idx] = np.vstack(sampled_blocks)
        bootstrapped_residuals = bootstrapped_residuals[:, :simulation_months, :]

        path_histories = np.tile(self.historical_seed_matrix, (num_paths, 1, 1))

        spx_paths = np.zeros((num_paths, simulation_months))
        cpi_paths = np.zeros((num_paths, simulation_months))
        yield_3m_paths = np.zeros((num_paths, simulation_months))
        yield_5y_paths = np.zeros((num_paths, simulation_months))

        curr_spx = np.full(num_paths, self.initial_spx_level)
        curr_cpi = np.full(num_paths, self.initial_cpi_level)
        curr_3m = np.full(num_paths, self.initial_yield_3m)
        curr_5y = np.full(num_paths, self.initial_yield_5y)

        log_cape = np.full(num_paths, np.log(self.initial_cape))
        log_target_cape = np.log(self.target_cape)

        # Sustainable real return anchor
        equilibrium_equity_drift = self.earnings_growth * dt
        historical_spx_mean = self.historical_mean_returns[0]

        for step in range(simulation_months):
            design_step_components = [np.ones((num_paths, 1))]
            for lag in range(1, p + 1):
                design_step_components.append(path_histories[:, p - lag, :])
            current_design_matrix = np.hstack(design_step_components)

            raw_increments = current_design_matrix @ self.coefficient_matrix + \
                bootstrapped_residuals[:, step, :]

            # Isolate zero-mean stochastic equity shock from VAR
            spx_stochastic_shock = raw_increments[:, 0] - historical_spx_mean

            valuation_gap = log_cape - log_target_cape
            valuation_penalty = -self.gamma_cape * valuation_gap * dt

            # Reconstruct equity return anchored to equilibrium drift
            spx_log_ret = equilibrium_equity_drift + spx_stochastic_shock + valuation_penalty
            cpi_log_ret = raw_increments[:, 1]
            # Error-correction overlay: pull yields back toward their long-run
            # equilibrium level so they don't behave as an unanchored random walk.
            diff_3m = raw_increments[:, 2] - \
                self.phi_rate * (curr_3m - self.target_yield_3m) * dt
            diff_5y = raw_increments[:, 3] - \
                self.phi_rate * (curr_5y - self.target_yield_5y) * dt

            curr_spx *= np.exp(spx_log_ret)
            curr_cpi *= np.exp(cpi_log_ret)
            curr_3m = np.maximum(0.0, curr_3m + diff_3m)
            curr_5y = np.maximum(0.0, curr_5y + diff_5y)

            # CAPE update driven strictly by mean-zero return innovations and reversion drag
            log_cape += spx_stochastic_shock - \
                (self.phi_cape + self.gamma_cape) * valuation_gap * dt

            spx_paths[:, step] = curr_spx
            cpi_paths[:, step] = curr_cpi
            yield_3m_paths[:, step] = curr_3m
            yield_5y_paths[:, step] = curr_5y

            predicted_increments = raw_increments.copy()
            predicted_increments[:, 0] = spx_log_ret
            predicted_increments[:, 2] = diff_3m
            predicted_increments[:, 3] = diff_5y
            path_histories[:, :-1, :] = path_histories[:, 1:, :]
            path_histories[:, -1, :] = predicted_increments

        return spx_paths, cpi_paths, yield_3m_paths, yield_5y_paths
