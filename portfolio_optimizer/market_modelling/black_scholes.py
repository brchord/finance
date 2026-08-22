"""
Black-Scholes Options Pricing and Greeks.
"""

import math
import numpy as np

def norm_cdf(x: float):
    """
    Represents the cumulative distribution
    function for the normal distribution.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def option_price(spot: float, strike: float, expiration: float,
                 rf_rate: float, sigma: float, is_call=False):
    """
    Compute an option price using the Black-Scholes-Myrton
    options pricing model.
    """
    print(f"""Pricing option:
                  Spot: ${spot:,.2f}
                Strike: ${strike:,.2f}
            Expiration:  {expiration * 365.0:.0f} DTEs
        Risk Free Rate:  {rf_rate * 100:.2f}%
    Implied Volatility:  {sigma * 100:.2f}%
              Is Call?:  {is_call}""")

    t = max(expiration, 1e-5)
    d1 = (np.log(spot / strike) + (rf_rate + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if is_call:
        return spot * norm_cdf(d1) - strike * np.exp(-rf_rate * t) * norm_cdf(d2)
    return strike * np.exp(-rf_rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def option_delta(spot: float, strike: float, expiration: float,
                 rf_rate: float, sigma: float, is_call=False):
    """
    Compute an option delta using the Black-Scholes-Myrton
    options pricing model.
    """
    t = max(expiration, 1e-5)
    d1 = (np.log(spot / strike) + (rf_rate + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    if is_call:
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0
