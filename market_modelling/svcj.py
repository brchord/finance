"""
Stochastic Volatility with Correlated Jumps
Time Series Simulation
"""

import numpy as np


class SVCJSimulation:
    """
    Stochastic Volatility with Correlated Jumps (SVCJ) Path Generator for SPX,
    VIX, and VIX3M.

    Generates joint trajectories for SPX price, 30-Day Spot VIX, and 3-Month
    VIX (VIX3M) using an Euler-Maruyama log-space discretization with full
    truncation boundary handling for zero/negative variance states.

    Empirical Reference Ranges
    (Eraker, Johannes & Polson, 2003 / Eraker, 2004):
    -------------------------------------------------------------------------
    mu       : [0.07, 0.09]    - Annualized drift (7% to 9%).
    kappa    : [3.0, 5.0]      - Mean reversion speed (half-life ~1.5 to
                                 2.8 months).
    theta    : [0.0225, 0.0324]- Long-run variance (15% to 18% annualized vol).
    sigma_v  : [0.25, 0.35]    - Volatility of volatility.
    rho      : [-0.85, -0.70]  - Price-variance correlation
                                 (diffusive leverage).
    lambda_j : [1.0, 2.0]      - Annual jump intensity (~1 to 2 jumps
                                 per year).
    mu_v     : [0.02, 0.05]    - Mean size of variance jump.
    mu_y     : [-0.08, -0.03]  - Mean log-price jump size (-3% to -8%).
    sigma_y  : [0.03, 0.06]    - Price jump volatility.
    rho_j    : [-1.5, -0.5]    - Co-jump price-variance coupling coefficient.
    """

    def __init__(
        self,
        initial_spx: float,
        initial_vix: float,
        mu: float = 0.07,
        kappa: float = 4.0,
        theta: float = 0.0324,
        sigma_v: float = 0.30,
        rho: float = -0.85,
        lambda_j: float = 1.5,
        mu_v: float = 0.035,
        mu_y: float = -0.08,
        sigma_y: float = 0.045,
        rho_j: float = -1.0,
        variance_risk_premium: float = 1.2
    ):
        """
        Parameters
        ----------
        initial_spx : float
            Starting spot price level for SPX (S_0).
        initial_vix : float
            Starting volatility level as a decimal or percentage (e.g., 0.16 or
            16.0 for 16% VIX).
            Converted internally to initial variance
            (V_0 = (initial_vix / 100)^2 if > 1.0).
        mu : float, default 0.08
            Annualized drift / expected rate of return for the underlying
            price.
        kappa : float, default 4.0
            Physical (P-measure) mean-reversion speed parameter for variance.
        theta : float, default 0.0256
            Physical (P-measure) long-run variance target level.
        sigma_v : float, default 0.30
            Volatility of volatility (diffusive noise scale for variance
            process).
        rho : float, default -0.75
            Correlation coefficient between price and variance Brownian
            motions.
        lambda_j : float, default 1.5
            Jump intensity parameter (expected Poisson jump arrivals per year).
        mu_v : float, default 0.03
            Mean of exponential variance jump size J^V ~ Exp(scale = mu_v).
        mu_y : float, default -0.05
            Unconditional mean log-price jump size J^S.
        sigma_y : float, default 0.04
            Standard deviation of log-price jump size J^S.
        rho_j : float, default -1.0
            Co-jump dependency parameter (E[J^S | J^V] = mu_y + rho_j * J^V).
        variance_risk_premium : float, default 1.2
            Multiplier mapping physical variance dynamics (P) to
            risk-neutral (Q) expectations used for derivative pricing and
            VIX/VIX3M calculation.
        """
        # Initial Conditions (Normalizing initial_vix to decimal if passed
        # on 0-100 scale)
        self.initial_spx = initial_spx
        self.initial_vix_decimal = (
                initial_vix / 100.0 if initial_vix > 1.0 else initial_vix)
        self.initial_variance = self.initial_vix_decimal ** 2

        # Continuous Diffusive Parameters (Physical P-measure)
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

        # Jump Parameters
        self.lambda_j = lambda_j
        self.mu_v = mu_v
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self.rho_j = rho_j

        # Risk-Neutral (Q-measure) Parameters for VIX Expectations
        self.vrp = variance_risk_premium
        self.kappa_q = self.kappa * 0.85
        self.theta_q = self.theta * self.vrp
        self.lambda_q = self.lambda_j * 1.1
        self.mu_v_q = self.mu_v * 1.15

        # Verify Feller Condition
        feller_ratio = (2 * self.kappa * self.theta) / (self.sigma_v ** 2)
        if feller_ratio <= 1.0:
            raise ValueError(f"Warning: Feller condition not strictly met "
                             f"(Ratio: {feller_ratio:.2f} <= 1.0). "
                             f"Variance truncation active.")

    def _calculate_jump_compensator(self) -> float:
        """Calculates k_j = E[exp(J^S) - 1] to compensate price drift."""
        denom = 1.0 - (self.rho_j * self.mu_v)
        if denom <= 0:
            raise ValueError("Denominator (1 - rho_j * mu_v) must be positive "
                             "for finite expected jump size.")

        expected_exp_j = np.exp(self.mu_y + 0.5 * (self.sigma_y ** 2)) / denom
        return expected_exp_j - 1.0

    def _compute_vix_index(self,
                           current_variance: np.ndarray,
                           term_days: float = 30.0) -> np.ndarray:
        """
        Computes expected VIX index value under Q for a given term horizon in
        calendar days.
        term_days = 30.0 for standard VIX; term_days = 90.0 (0.25 yrs) for
                         VIX3M.
        """
        tau = term_days / 365.0
        theta_total_q = self.theta_q + (
            self.lambda_q * self.mu_v_q / self.kappa_q)

        integrated_var_exp = (
            theta_total_q + (current_variance - theta_total_q) *
            ((1.0 - np.exp(-self.kappa_q * tau)) / (self.kappa_q * tau))
        )
        return np.sqrt(np.maximum(integrated_var_exp, 0.0)) * 100.0

    def simulate_paths(
        self,
        trading_days: int = 252,
        num_paths: int = 10000,
        seed: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates simulated price, spot VIX (30-day), and VIX3M trajectories.

        Parameters
        ----------
        trading_days : int
            Total number of trading days to simulate (e.g., 252 for 1 year).
        num_paths : int
            Number of Monte Carlo paths to generate.
        seed : int, optional
            Random seed for path generation reproducibility.

        Returns
        -------
        spx_paths : np.ndarray
            Price paths array of shape (trading_days + 1, num_paths).
        vix_paths : np.ndarray
            Standard 30-Day VIX index paths array of shape
            (trading_days + 1, num_paths).
        vix3m_paths : np.ndarray
            Synthetic 3-Month VIX (VIX3M) index paths array of shape
            (trading_days + 1, num_paths).
        """
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        dt = 1.0 / 252.0
        sqrt_dt = np.sqrt(dt)

        # Preallocate outputs
        spx_paths = np.zeros((trading_days + 1, num_paths))
        vix_paths = np.zeros((trading_days + 1, num_paths))
        vix3m_paths = np.zeros((trading_days + 1, num_paths))

        # Internal state tracking array for continuous variance process
        variance_state = np.zeros(num_paths)
        variance_state[:] = self.initial_variance

        spx_paths[0] = self.initial_spx
        vix_paths[0] = self._compute_vix_index(
            variance_state, term_days=30.0)
        vix3m_paths[0] = self._compute_vix_index(
            variance_state, term_days=90.0)

        # Jump compensator drift correction
        k_j = self._calculate_jump_compensator()
        drift_price = self.mu - (self.lambda_j * k_j)

        # Correlated Brownian motion setup
        rho_complement = np.sqrt(1.0 - self.rho ** 2)

        for step in range(trading_days):
            spx_curr = spx_paths[step]

            # Full Truncation Scheme for zero/negative variance boundary
            # protection.
            variance_pos = np.maximum(variance_state, 0.0)
            sqrt_variance_pos = np.sqrt(variance_pos)

            # Correlated Gaussian innovations
            variance_shocks = rng.standard_normal(num_paths)
            independent_shocks = rng.standard_normal(num_paths)
            price_shocks = (self.rho * variance_shocks +
                            rho_complement * independent_shocks)

            # Poisson jump arrivals per step
            num_jumps = rng.poisson(self.lambda_j * dt, size=num_paths)

            # Sample jump magnitudes (Co-jump dynamics)
            variance_jump = rng.exponential(
                scale=self.mu_v, size=num_paths) * (num_jumps > 0)

            price_jump_mean = self.mu_y + (self.rho_j * variance_jump)
            price_jump = (rng.normal(
                loc=price_jump_mean, scale=self.sigma_y, size=num_paths) *
                (num_jumps > 0))

            # Continuous + Jump update for latent Variance state
            d_variance = (
                self.kappa * (self.theta - variance_pos) * dt
                + self.sigma_v * sqrt_variance_pos * sqrt_dt * variance_shocks
                + variance_jump
            )
            variance_state = np.maximum(variance_state + d_variance, 0.0)

            # Continuous + Jump update for Price (Log-space discretization)
            d_log_spx = (
                (drift_price - 0.5 * variance_pos) * dt
                + sqrt_variance_pos * sqrt_dt * price_shocks
                + price_jump
            )
            spx_paths[step + 1] = spx_curr * np.exp(d_log_spx)

            # Map updated variance state directly to 30-Day VIX and
            # 90-Day VIX3M.
            vix_paths[step + 1] = self._compute_vix_index(
                variance_state, term_days=30.0)
            vix3m_paths[step + 1] = self._compute_vix_index(
                variance_state, term_days=90.0)

        return spx_paths[:-1, :], vix_paths[:-1, :], vix3m_paths[:-1, :]
