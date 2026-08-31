"""
linear_models.py

Contains a set of simple linear models that represent well known
investment strategies:

1. Fixed Income.
2. Long SP500.
3. A single strategy built out of a linear combination
   of the ones above.
"""

import math
from abc import ABC, abstractmethod
from typing import override

import numpy as np

from market_modelling.dsvi import DynamicSVI

class InvestmentStrategy(ABC):
    """
    Abstract class representing a generic investment strategy.
    Designed to be inherited from different classes that represent
    a different investment strategy.
    """
    def __init__(self):
        super().__init__()
        self.verbose = False
        self.book = []


    def enable_verbosity(self):
        """
        Whether or not print diagnostics on the
        standard output.
        """
        self.verbose = True


    def log(self, s: str):
        "Simple logging"
        if self.verbose:
            print(s)

    @classmethod
    def from_json_object(cls, o):
        """Builds a portfolio instance out of a JSON parsed object."""
        raise NotImplementedError("Do not invoke this directly")


    @abstractmethod
    def run_simulation(
        self, *,
        spot_spx: list[float],        # Time series for SPX underlying price.
        spot_vix: list[float],        # Time series for the spot VIX.
        vix3m: list[float],           # Time series for the VIX3M.
        svi: DynamicSVI,              # Stochastic Volatility Inspired IV Model.
        initial_nav: float,           # NAV to start the simulation with.
        days: int,                    # Days to run the simulation
        full_book=False) -> np.array: # Track full options book for debugging.
        """
        Starts the investment portfolio simulation.
        returns: a time series representing the daily changes in NAV.
        """
        return np.full(initial_nav, days)


    def transaction_book(self):
        """Returns the full trading book for the last simulated path that
        enabled full book tracking.
        """
        return self.book



class FixedIncomeStrategy(InvestmentStrategy):
    """
    Represents a traditional Fixed Income investment strategy
    that can either perform daily, monthly or continuous compounding.

    This strategy assumes all the yields from the fixed income
    instrument are fully reinvested into the same asset.
    """
    def __init__(self,
                 rate=0.03,            # Annualized interest rate
                 compounding='daily'): # 'daily', 'monthly' or
                                       # 'continuous' compounding
        super().__init__()
        self.rate = rate
        comp = compounding.lower()

        if comp not in ['daily', 'monthly', 'continous']:
            raise ValueError("Invalid interest compounding policy")

        self.compounding = comp


    def run_simulation(
        self, *,
        spot_spx: list[float],        # Time series for SPX underlying price.
        spot_vix: list[float],        # Time series for the spot VIX.
        vix3m: list[float],           # Time series for the VIX3M.
        svi: DynamicSVI,              # Stochastic Volatility Inspired IV Model.
        initial_nav: float,           # NAV to start the simulation with.
        days: int,                    # Days to run the simulation
        full_book=False) -> np.array: # Track full options book for debugging.
        """Run portfolio simulation (see parent's class docstring)."""
        daily_rate = self.rate / 252.0
        monthly_rate = self.rate / 12.0
        path = np.zeros(days)
        current_nav = initial_nav
        for d in range(0, days):
            if self.compounding == 'daily':
                current_nav *= 1 + daily_rate
            elif self.compounding == 'monthly':
                if d % 21 == 0:
                    current_nav *= 1 + monthly_rate
            else:
                current_nav *= math.exp(self.rate / 252.0)
            path[d] = current_nav
        return path

    @classmethod
    @override
    def from_json_object(cls, o):
        """
        Builds a lambda that given an underlying, monthly volatility and
        3 month forward volatility simulated paths returns an instance of
        this portfolio strategy from a JSON parsed object. The structure
        must have the following shape:
        {
            "type": "FixedIncomeStrategy",
            "rate": interest_rate,
            "compounding": "continuous" | "daily" | "monthly"
        }
        """
        if o["type"] != "FixedIncomeStrategy":
            return None
        return FixedIncomeStrategy(float(o["rate"]), o["compounding"])


class LongSPYStrategy(InvestmentStrategy):
    """
    Represent a simple long SP500 investment strategy that also accounts
    for quarterly dividend distributions.
    """
    def __init__(self,
                 avg_yield=0.0105): # SP500's average dividend yield.
        super().__init__()
        self.log(f"""Initializing long SPY portfolio strategy:
      Average Dividend Yield: {avg_yield*100.0:.2f}%""")
        self.avg_yield = avg_yield


    def run_simulation(
        self, *,
        spot_spx: list[float],        # Time series for SPX underlying price.
        spot_vix: list[float],        # Time series for the spot VIX.
        vix3m: list[float],           # Time series for the VIX3M.
        svi: DynamicSVI,              # Stochastic Volatility Inspired IV Model.
        initial_nav: float,           # NAV to start the simulation with.
        days: int,                    # Days to run the simulation
        full_book=False) -> np.array: # Track full options book for debugging.
        """Run portfolio simulation (see parent's class docstring)."""
        shares = initial_nav / (spot_spx[0] / 10.0)
        quarterly_yield = self.avg_yield / 4.0

        cash = 0.0
        path = np.zeros(days)
        for d in range(0, days):
            if d % 63 == 0:
                cash += shares * quarterly_yield
            nav = shares * (spot_spx[d] / 10.0) + cash
            path[d] = nav
        return path


    @classmethod
    @override
    def from_json_object(cls, o):
        """
        Builds a lambda that given an underlying, monthly volatility and
        3 month forward volatility simulated paths returns an instance of
        this portfolio strategy from a JSON parsed object. The structure
        must have the following shape:
        {
            "type": "LongSPYStrategy",
            "avg_yield": dividend_yield
        }
        """
        if o["type"] != "LongSPYStrategy":
            return None
        return LongSPYStrategy(float(o["avg_yield"]))


class CombinedPortfolioStrategy(InvestmentStrategy):
    """
    Computes the total return of a lineal combination of multiple
    Investment Strategies.
    components: list of tuples of the form (strategy, weigh) for each
                piece of the combined portfolio
    raises: ValueError if the weights are either negative or the total
            portfolio weight exceeds 1.0
    """
    def __init__(self, components: list[tuple[InvestmentStrategy, float]]):
        super().__init__()
        total_weight = sum(weigh for (_, weigh) in components)
        non_positive_weights = list(filter(lambda x: x <= 0.0,
                                       [weight for (_, weight) in components]))
        if len(non_positive_weights) > 0:
            raise ValueError("Negative weight for portfolio components")

        if total_weight > 1.0:
            raise ValueError("Portfolio weighs more than 100%")

        if total_weight < 1.0:
            self.log(f"""Warning: Portfolio components don't add to 100%,
                  keeping the remainder {(1.0 - total_weight) * 100:.2f}%
                  in cash""")
        self.components = components


    def run_simulation(
        self, *,
        spot_spx: list[float],        # Time series for SPX underlying price.
        spot_vix: list[float],        # Time series for the spot VIX.
        vix3m: list[float],           # Time series for the VIX3M.
        svi: DynamicSVI,              # Stochastic Volatility Inspired IV Model.
        initial_nav: float,           # NAV to start the simulation with.
        days: int,                    # Days to run the simulation
        full_book=False) -> np.array: # Track full options book for debugging.
        """Run portfolio simulation (see parent's class docstring)."""
        total_weight = 0.0
        result = np.zeros(days)

        for portfolio, weight in self.components:
            total_weight += weight
            partial = portfolio.run_simulation(
                spot_spx=spot_spx, spot_vix=spot_vix, vix3m=vix3m,
                svi=svi, initial_nav=initial_nav * weight,
                days=days, full_book=full_book)
            result += partial
        if (1.0 - total_weight) >= 1e3:
            rem_cash = initial_nav * (1.0 - total_weight)
            result += rem_cash

        return result


    @classmethod
    @override
    def from_json_object(cls, o):
        from portfolio_models.short_put_model import ShortSPXPutStrategy
        from portfolio_models.put_credit_spreads_model \
            import SPXPutCreditSpreadStrategy

        if o["type"] != "CombinedPortfolioStrategy":
            return None
        models = [FixedIncomeStrategy, LongSPYStrategy,
                  ShortSPXPutStrategy, SPXPutCreditSpreadStrategy]
        components = o["components"]
        portfolios = []
        for c in components:
            p = c["portfolio"]
            w = float(c["weight"])
            for m in models:
                po = m.from_json_object(p)
                if po is not None:
                    portfolios.append((po, w))
                    continue
        return CombinedPortfolioStrategy(portfolios)
