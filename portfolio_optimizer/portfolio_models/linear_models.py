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

class InvestmentStrategy(ABC):
    """
    Abstract class representing a generic investment strategy.
    Designed to be inherited from different classes that represent
    a different investment strategy.
    """
    @abstractmethod
    def run_simulation(self, initial_nav: float, days: int, daily_nav=False):
        """
        Starts the investment portfolio simulation.
        initial_nav: Portfolio's Net Asset Value.
        days: How many days to run the simulation.
        daily_nav: Whether or not return a time series of daily
                   NAV values or simply the terminal NAV after the
                   simulation.
        returns: either the final portfolio's NAV or a time series
                 representing the daily changes in NAV.
        """
        return [initial_nav] * days if daily_nav else initial_nav


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


    def run_simulation(self, initial_nav: float, days: int, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        daily_rate = self.rate / 252.0
        monthly_rate = self.rate / 12.0
        if not daily_nav:
            final_nav = 0.0
            if self.compounding == 'daily':
                final_nav = initial_nav * math.pow(1 + daily_rate,
                                                   days)
            elif self.compounding == 'monthly':
                final_nav = initial_nav * math.pow(1 + monthly_rate,
                                                   days / 21.0)
            else:
                time_in_years = days / 252.0
                final_nav = initial_nav * math.exp(
                    self.rate * time_in_years)
            return final_nav

        path = []
        current_nav = initial_nav
        for d in range(0, days):
            if self.compounding == 'daily':
                current_nav *= 1 + daily_rate
            elif self.compounding == 'monthly':
                if d % 21 == 0:
                    current_nav *= 1 + monthly_rate
            else:
                current_nav *= math.exp(self.rate / 252.0)
            path.append(current_nav)
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
        print(f"""Initializing long SPY portfolio strategy:
            Initial SPX Spot: {spot_spx[0]:,.2f}
      Average Dividend Yield: {avg_yield*100.0:.2f}%""")
        self.spy_spot = [x / 10.0 for x in spot_spx]
        self.avg_yield = avg_yield


    def run_simulation(self, initial_nav: float, days: int, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        shares = initial_nav / self.spy_spot[0]
        total_quarters = math.floor(days / 63.0)
        quarterly_yield = self.avg_yield / 4.0

        if not daily_nav:
            return shares * self.spy_spot[-1] + total_quarters * quarterly_yield * shares

        cash = 0.0
        path = []
        for d in range(0, days):
            if d % 63 == 0:
                cash += shares * quarterly_yield
            nav = shares * self.spy_spot[d] + cash
            path.append(nav)
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
            print(f"""Warning: Portfolio components don't add to 100%,
                  keeping the remainder {(1.0 - total_weight) * 100:.2f}%
                  in cash""")
        self.components = components


    def run_simulation(self, initial_nav, days, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        total_weight = 0.0
        if daily_nav:
            result = []
        else:
            result = 0.0
        for portfolio, weight in self.components:
            total_weight += weight
            partial = portfolio.run_simulation(
                initial_nav * weight, days, daily_nav)
            if daily_nav:
                result = [x + y for (x, y) in zip(result, partial)]
            else:
                result += partial
        if (1.0 - total_weight) >= 1e3:
            rem_cash = initial_nav * (1.0 - total_weight)
            if daily_nav:
                result = [x + rem_cash for x in result]
            else:
                result += rem_cash

        return result
