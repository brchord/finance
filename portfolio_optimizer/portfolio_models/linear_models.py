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

import numpy as np

class InvestmentStrategy(ABC):
    """
    Abstract class representing a generic investment strategy.
    Designed to be inherited from different classes that represent
    a different investment strategy.
    """
    def __init__(self):
        super().__init__()
        self.verbose = False


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


    @abstractmethod
    def run_simulation(self, initial_nav: float, days: int) -> np.array:
        """
        Starts the investment portfolio simulation.
        initial_nav: Portfolio's Net Asset Value.
        days: How many days to run the simulation.
        returns: either the final portfolio's NAV or a time series
                 representing the daily changes in NAV.
        """
        return np.full(initial_nav, days)


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


    def run_simulation(self, initial_nav: float, days: int) -> np.array:
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


class LongSPYStrategy(InvestmentStrategy):
    """
    Represent a simple long SP500 investment strategy that also accounts
    for quarterly dividend distributions.
    """
    def __init__(self,
                 spot_spx: list[float], # Time series for SPX spot price.
                 avg_yield=0.0105):     # SP500's average dividend yield.
        super().__init__()
        self.log(f"""Initializing long SPY portfolio strategy:
            Initial SPX Spot: {spot_spx[0]:,.2f}
      Average Dividend Yield: {avg_yield*100.0:.2f}%""")
        self.spy_spot = [x / 10.0 for x in spot_spx]
        self.avg_yield = avg_yield


    def run_simulation(self, initial_nav: float, days: int) -> np.array:
        """Run portfolio simulation (see parent's class docstring)."""
        shares = initial_nav / self.spy_spot[0]
        quarterly_yield = self.avg_yield / 4.0

        cash = 0.0
        path = np.zeros(days)
        for d in range(0, days):
            if d % 63 == 0:
                cash += shares * quarterly_yield
            nav = shares * self.spy_spot[d] + cash
            path[d] = nav
        return path


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
        if len(filter(weigh for (_, weigh) in components if weigh < 0.0)) > 0:
            raise ValueError("Negative weight for portfolio components")

        if total_weight > 1.0:
            raise ValueError("Portfolio weighs more than 100%")

        if total_weight < 1.0:
            self.log(f"""Warning: Portfolio components don't add to 100%,
                  keeping the remainder {(1.0 - total_weight) * 100:.2f}%
                  in cash""")
        self.components = components


    def run_simulation(self, initial_nav, days) -> np.array:
        """Run portfolio simulation (see parent's class docstring)."""
        total_weight = 0.0
        result = np.zeros(days)

        for portfolio, weight in self.components:
            total_weight += weight
            partial = portfolio.run_simulation(
                initial_nav * weight, days)
            result += partial
        if (1.0 - total_weight) >= 1e3:
            rem_cash = initial_nav * (1.0 - total_weight)
            result += rem_cash

        return result
