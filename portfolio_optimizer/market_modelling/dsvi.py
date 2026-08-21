"""
Dynamic Stochastic Volatility Interpolation.
"""

import numpy as np
from scipy.optimize import minimize

class DynamicSVI:
    """
    Dynamic Stochastic Volatility Interpolation

    Simulates a volatility surface using a L-BFGS-B optimization
    model fitted using real volatility smirks observed in actual
    equities markets.
    """
    def __init__(self, strikes_market, iv_market, spot_price, yearly_exp):
        """
        Initializes Dynamic Surface Volatility Interpolation
        with data extracted from a real observed market options chain.
        strikes_market: list of observed strike prices from real market data
        iv_market: corresponding implied volatility for the above strikes
        spot_price: current underlying spot price
        yearly_exp: option maturity in years
        """
        self.yearly_exp = yearly_exp
        k_market = np.log(strikes_market / spot_price)
        self.a0, self.b, self.rho, self.m, self.sigma = self._fit_svi(
            k_market, iv_market, yearly_exp)
        # Precompute the shape constant C (contribution of shape to ATM variance)
        self.shape_constant = self.b * (-self.rho * self.m + np.sqrt(self.m**2 + self.sigma**2))


    def _fit_svi(self, k_market, iv_market, exp, initial_guess=None):
        w_market = (iv_market ** 2) * exp

        def svi_total_variance(params, k):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def objective(params):
            _, b, rho, _, sigma = params
            if b < 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e6
            w_fit = svi_total_variance(params, k_market)
            return np.sum((w_fit - w_market) ** 2)

        if initial_guess is None:
            initial_guess = [w_market[len(w_market)//2], 0.1, -0.4, 0.0, 0.1]

        bounds = [
            (-np.inf, np.inf),
            (0.0, np.inf),
            (-0.999, 0.999),
            (-np.inf, np.inf),
            (1e-4, np.inf)
        ]

        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        return result.x

    def get_iv_curve(self, simulated_atm_iv, strikes, forward, current_exp):
        """
        Extrapolates the full OTM IV curve given a simulated ATM IV and the
        current (potentially shrunk) time-to-expiry current_exp.
        """
        # 1. Convert simulated ATM IV to total variance for the current T
        w_atm_target = (simulated_atm_iv ** 2) * current_exp

        # 2. Dynamically adjust 'a' to match the simulated ATM variance
        a_t = w_atm_target - self.shape_constant

        # 3. Compute log-moneyness for target strikes
        k = np.log(strikes / forward)

        # 4. Evaluate Raw SVI total variance
        w_t = a_t + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2))

        # 5. Convert back to implied volatility using the current expiration
        iv_curve = np.sqrt(np.maximum(w_t, 1e-8) / current_exp)
        return iv_curve
