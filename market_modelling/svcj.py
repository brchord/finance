"""
Stochastic Volatility with Correlated Jumps
Time Series Simulation
"""

import math
import numpy as np

class SVCJSimulation:
    """
    Stochastic Volatility with Correlated Jumps (SVCJ)
    Instituitonal parameters for S&P 500.
    """

    def __init__(self, *,
                 seed = None,    # RNG seed
                 mu = 0.08,      # Equity drift
                 kappa = 4.5,    # VIX mean reversion speed
                 theta = 0.04,   # Long-term variance
                 sigma_v = 0.4,  # Volatility of volatility
                 rho = -0.65,    # Price/vol correlation (leverage effect)
                 lambda_j = 1.5, # Expected jumps per year
                 mu_j = -0.05,   # Mean price jump size
                 sigma_j = 0.06, # Volatility of price jump
                 mu_v = 0.08):   # Mean variance jump size
        """
        Initializes the model with the canonical parameters used
        to simulate the behavior or equities markets.
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.mu_v = mu_v
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()


    def _derive_vix3m(self, vix_path: list[float]):
        """
        Derives VIX3M path directly from a pre-calculated VIX path using affine mapping.
        """
        # Helper functions for affine coefficients A(tau) and B(tau)
        def get_a_b(tau):
            b = (1.0 - np.exp(-self.kappa * tau)) / (self.kappa * tau)
            a = self.theta * (1.0 - b)
            return a, b

        a30, b30 = get_a_b(30/365.0)
        a90, b90 = get_a_b(90/365.0)

        # Compute linear mapping coefficients alpha and beta
        beta = b90 / b30
        alpha = a90 - beta * a30

        # Map VIX^2 to VIX3M^2 via linear transformation, then take the square root
        vix3m_path = np.sqrt(alpha + beta * (vix_path ** 2))

        return vix3m_path


    def generate_path(self, start_spot: float, start_atm_iv: float, days=252):
        """
        Generates a potential market as two time series that represent
        the underlying and its corresponding 30 forward implied volatility.

          start_spot: represents the starting value of the underlying.
        start_atm_iv: starting value of 30 IV for the underlying.
                days: size of the simulated path in trading days.
        """
        dt = 1.0 / 252.0
        spot = np.zeros(days)
        spot[0] = start_spot

        atm_iv = np.zeros(days)
        atm_iv[0] = start_atm_iv * start_atm_iv

        for t in range(1, days):
            z1 = self.rng.standard_normal()
            z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * self.rng.standard_normal()

            # Poisson Jump
            n = self.rng.poisson(self.lambda_j * dt)
            j_s = 0
            j_v = 0

            if n > 0:
                z3 = self.rng.standard_normal()
                j_s = self.mu_j + self.sigma_j * z3
                # Variance jumps are positive and exponentially distributed
                j_v = self.rng.exponential(self.mu_v)

            # Variance process (Euler-Maruyama, ensuring V > 0)
            v_prev = max(atm_iv[t-1], 1e-6)
            atm_iv[t] = v_prev + self.kappa * (self.theta - v_prev) * dt + \
                   self.sigma_v * math.sqrt(v_prev * dt) * z2 + j_v
            atm_iv[t] = max(atm_iv[t], 7.225e-3)

            # Price process
            spot[t] = spot[t-1] * np.exp((self.mu - 0.5 * v_prev) * dt + \
                                   math.sqrt(v_prev * dt) * z1 + j_s)

        vix3m = self._derive_vix3m(atm_iv)
        return spot, np.sqrt(atm_iv), np.sqrt(vix3m)
