"""
linear_models.py

Contains a set of simple linear models that represent well known
investment strategies:

1. Fixed Income.
2. Long SP500.
3. A single strategy built out of a linear combination
   of the ones above.
"""

import logging
import math

from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd

from market_modelling.dsvi import DynamicSVI

logger = logging.getLogger(__name__)

class InvestmentStrategy(ABC):
    """
    Abstract class representing a generic investment strategy.
    Designed to be inherited from different classes that represent
    a different investment strategy.
    """
    def __init__(self):
        super().__init__()
        self.book = []


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
                 compounding='daily',  # 'daily', 'monthly' or
                                       # 'continuous' compounding
                 monthly=0.0):    # Monthly distribution
        super().__init__()
        self.rate = rate
        self.monthly = monthly
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
                    # For monthly compounding, also mark
                    # the deposit in the transaction book.
                    if full_book:
                        self.book.append({
                            "day": d,
                            "trade": "dividend",
                            "price": current_nav * monthly_rate,
                        })
                    current_nav *= 1 + monthly_rate
            else:
                current_nav *= math.exp(self.rate / 252.0)

            if d % 21 == 0:
                # Subract monthly distribution
                if self.monthly != 0.0:
                    current_nav -= self.monthly
                    # Mark it in the transactions book
                    if full_book:
                        self.book.append({
                            "day": d,
                            "trade": "withdrawal",
                            "price": -self.monthly
                        })

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
            "compounding": "continuous" | "daily" | "monthly",
            "monthly": monthly_withdrawals
        }
        """
        if o["type"] != "FixedIncomeStrategy":
            return None
        return FixedIncomeStrategy(
            float(o["rate"]), o["compounding"], float(o["monthly"]))


class LongSPYStrategy(InvestmentStrategy):
    """
    Represent a simple long SP500 investment strategy that also accounts
    for quarterly dividend distributions.
    """
    def __init__(self,
                 avg_yield=0.0105,  # SP500's average dividend yield.
                 monthly=0.0):      # Monthly withdrawals
        super().__init__()
        logging.debug("Initializing long SPY portfolio strategy:")
        logging.debug("Average Dividend Yield: %.2f%%", avg_yield)
        logging.debug("Monthly Distribution: %.2f", monthly)
        self.avg_yield = avg_yield
        self.monthly = monthly


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

        # Record the initial long equity trade
        if full_book:
            self.book.append({
                "day": 0,
                "trade": "buy",
                "size": shares,
                "price": spot_spx[0] / 10.0,
            })
        cash = 0.0
        path = np.zeros(days)
        for d in range(0, days):
            spy_price = spot_spx[d] / 10.0
            if d % 63 == 0:
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "dividend",
                        "price": shares * spy_price * quarterly_yield
                    })
                cash += shares * spy_price * quarterly_yield
            if d % 21 == 0:
                # Take monthly distribution from dividends
                # and shares.
                if self.monthly != 0:
                    # First try to withdraw from floating cash
                    deficit = self.monthly
                    if cash >= self.monthly:
                        cash -= self.monthly
                        if full_book:
                            self.book.append({
                                "day": d,
                                "trade": "withdrawal",
                                "price": self.monthly
                            })
                        deficit = 0.0
                    else:
                        deficit -= cash
                        prev_cash = cash
                        cash = 0.0
                        if full_book and prev_cash > 0:
                            self.book.append({
                                "day": d,
                                "trade": "withdrawal",
                                "price": prev_cash
                            })
                    assert deficit >= 0
                    assert cash >= 0
                    if deficit > 0:
                        shares_to_sell = deficit / spy_price
                        shares -= shares_to_sell
                        if full_book:
                            self.book.append({
                                "day": d,
                                "trade": "sell",
                                "size": shares_to_sell,
                                "price": spy_price,
                                "total": self.monthly
                            })
            nav = shares * spy_price + cash
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
            "avg_yield": dividend_yield,
            "monthly": monthly_withdrawals
        }
        """
        if o["type"] != "LongSPYStrategy":
            return None
        return LongSPYStrategy(float(o["avg_yield"]),
                               float(o["monthly"]))


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
            rem = 1.0 - total_weight
            logging.debug("Warning: Portfolio components don't add to 100%,")
            logging.debug("keeping the remainder %.2f%% in cash", rem)
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
        books = []

        for portfolio, weight in self.components:
            total_weight += weight
            partial = portfolio.run_simulation(
                spot_spx=spot_spx, spot_vix=spot_vix, vix3m=vix3m,
                svi=svi, initial_nav=initial_nav * weight,
                days=days, full_book=full_book)
            result += partial
            if full_book:
                books.append(portfolio.transaction_book())
        if (1.0 - total_weight) >= 1e3:
            rem_cash = initial_nav * (1.0 - total_weight)
            result += rem_cash

        # Record all trades from all portfolios in a combined book.
        # This is a naive implementation that merges books by linearly
        # scanning each sub-book day by day so it's not meant for
        # large portfolios or large scale multi-path simulations.
        if full_book:
            for d in range(0, days):
                for b in books:
                    day_trades = [entry for entry in b if entry["day"] == d]
                    self.book.extend(day_trades)

        return result


    @classmethod
    @override
    def from_json_object(cls, o):
        from portfolio_models.short_put_model import ShortSPXPutStrategy
        from portfolio_models.put_credit_spreads_model \
            import SPXPutCreditSpreadStrategy

        if o["type"] != "CombinedPortfolioStrategy":
            return None
        models = [SPXPutCreditSpreadStrategy,
                  LongSPYWithTreasuryLadders,
                  ShortSPXPutStrategy,
                  LongSPYStrategy,
                  FixedIncomeStrategy]
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


class LongSPYWithTreasuryLadders(InvestmentStrategy):
    """
    Simulates a portfolio comprised of a configurable long SPY sleeve
    and a fixed income T-Bill/T-Note ladder composed of
        - 20% short term T-Bills
        - 20% 2-year T-Note
        - 40% 3-year T-Note
        - 20% 5-year T-Note

    It properly models monthly withdrawals adjusted for inflation, proper
    distributions for each T-Note and equity dividends at their expected
    schedules.

    It automatically rebalances the portfolio attempting to sell equity
    at favorable prices (when price action shows a spot price going above
    the SMA-180, SMA-90 and the EMA-30 simultaneosly) and buying the
    corresponding T-Note that comes to the next maturity of the oldest one
    on the trading book.
    """
    def __init__(self,
                 equity_allocation: float,
                 ladder_allocation: float,
                 yearly_spending: float,
                 monthly_tbill_rate: float = 0.038,
                 average_tnote_rate: float = 0.042,
                 spy_avg_dividend_yield: float = 0.01,
                 average_inflation: float = 0.034):
        if equity_allocation + ladder_allocation != 1.0:
            raise ValueError("Portfolio allocation must sum up to 100%")
        super().__init__()
        self.equity_allocation = equity_allocation
        self.ladder_allocation = ladder_allocation
        self.yearly_spending = yearly_spending
        self.tbill_rate = monthly_tbill_rate
        self.tnote_rate = average_tnote_rate
        self.spy_div_yield = spy_avg_dividend_yield
        self.inflation = average_inflation


    def run_simulation(self, *, spot_spx, spot_vix, vix3m, svi, initial_nav, days, full_book=False):
        monthly_withdrawal = self.yearly_spending / 12
        tbill_daily_rate = self.tbill_rate / 252
        tnote_half_rate = self.tnote_rate / 2

        current_nav = initial_nav
        spot_spx = spot_spx / 10.0
        spy_price = spot_spx[0]
        spy_position_size = math.ceil(initial_nav * self.equity_allocation / spy_price)
        tnote_amount = initial_nav * self.ladder_allocation / 5.0

        tnotes = {
            0: (252 * 2, tnote_amount),
            1: (252 * 3, tnote_amount * 2),
            2: (252 * 5, tnote_amount)
        }
        tnote_id = 3
        cash = initial_nav - tnote_amount * 4 - spy_price * spy_position_size

        if full_book:
            self.book.append({
                "day": 0,
                "trade": "buy",
                "symbol": "SPY",
                "price": spy_price,
                "size": spy_position_size,
                "total": spy_price * spy_position_size,
                "description": f"Initial long equity at {self.equity_allocation*100.0}%"
            })
            self.book.append({
                "day": 0,
                "trade": "buy",
                "symbol": "T-Note 2 Years",
                "price": tnote_amount,
                "rate": self.tnote_rate,
                "maturity": 252 * 2,
                "description": "Initial T-Note ladder setup"
            })
            self.book.append({
                "day": 0,
                "trade": "buy",
                "symbol": "T-Note 3 Years",
                "price": tnote_amount * 2,
                "rate": self.tnote_rate,
                "maturity": 252 * 3,
                "description": "Initial T-Note ladder setup"
            })
            self.book.append({
                "day": 0,
                "trade": "buy",
                "symbol": "T-Note 5 Years",
                "price": tnote_amount,
                "rate": self.tnote_rate,
                "maturity": 252 * 5,
                "description": "Initial T-Note ladder setup"
            })
            self.book.append({
                "day": 0,
                "trade": "buy",
                "price": cash,
                "rate": self.tbill_rate,
                "description": "Cash equivalents in short term t-bills"
            })

        spx_df = pd.Series(spot_spx)
        ema30 = spx_df.ewm(span=20, adjust=False).mean().values
        sma90 = spx_df.rolling(window=63).mean()
        sma180 = spx_df.rolling(window=126).mean()
        return_path = np.zeros(days)

        for d in range(0, days):
            day_num = d + 1
            day_spy = spot_spx[d]
            transaction_day = False

            # Account for monthly withdrawals
            if day_num % 21 == 0:
                cash -= monthly_withdrawal
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "withdrawal",
                        "price": -monthly_withdrawal,
                        "description": "Monthly expense distribution"
                    })
                    transaction_day = True
            # Account for SPY dividend distribution
            if day_num % 63 == 0:
                cash_dividend = spy_position_size * day_spy * self.spy_div_yield / 4.0
                cash += cash_dividend
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "dividend",
                        "price": cash_dividend,
                        "description": "SPY dividend payment"
                    })
                    transaction_day = True

            cash *= 1 + tbill_daily_rate

            # Account for t-note distribution
            expired_notes = []
            for idx, (maturity, amount) in tnotes.items():
                tnote_day = maturity - d - 1
                if tnote_day % (252 / 2) == 0:
                    tnote_coupon = amount * tnote_half_rate
                    cash += tnote_coupon
                    if full_book:
                        self.book.append({
                            "day": d,
                            "trade": "dividend",
                            "price": tnote_coupon,
                            "description": "T-Note coupon payment"
                        })
                        transaction_day = True
                    if tnote_day <= 0:
                        expired_notes.append(idx)

            for expired_note in expired_notes:
                (maturity, amount) = tnotes.pop(expired_note)
                cash += amount
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "deposit",
                        "price": amount,
                        "description": "T-Note maturity reached"
                    })
                    transaction_day = True

            tnote_position = sum(amount for _, (_, amount) in tnotes.items())

            current_nav = cash + spy_position_size * day_spy + tnote_position
            fixed_income_position = cash + tnote_position
            # If the fixed income position goes below 80% of what it should be,
            # consider rebalancing the portfolio if equity pricing conditions are
            # favorable (i.e. crosses  SMA 180, SMA 90 and EMA 30).
            if fixed_income_position / current_nav <= self.ladder_allocation * 0.8:
                # Current spot is beyond EMA30, SMA90 and SMA120.
                if d >= 180 and day_spy >= ema30[d] and day_spy >= sma90[d] \
                            and day_spy >= sma180[d]:
                    needed_amount = current_nav * self.ladder_allocation \
                         - fixed_income_position
                    selling_position = math.ceil(needed_amount / day_spy)
                    if full_book:
                        self.book.append({
                            "day": d,
                            "trade": "sell",
                            "symbol": "SPY",
                            "size": selling_position,
                            "price": day_spy,
                            "description": "Portfolio rebalance to replenish "
                                           "fixed income"
                        })
                        transaction_day = True
                    spy_position_size -= selling_position
                    cash += selling_position * day_spy

            # If the liquid cash reserves fall below 6 months of runway, liquidate
            # equity to fund in case of an emergency
            # NOTE: Runway is defined as 6 months of needed extra liquidity
            #       on top of what dividend distributions provide
            expected_dividends = spy_position_size * spy_price * self.spy_div_yield / 2.0
            six_month_needs = self.yearly_spending / 2.0
            current_runway = cash - (six_month_needs - expected_dividends)
            if current_runway < 0:
                selling_position = math.ceil(-current_runway / day_spy)
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "sell",
                        "price": day_spy,
                        "size": selling_position,
                        "description": "Selling shares to cover 6 months of runway"
                    })
                    transaction_day = True
                spy_position_size -= selling_position
                cash += selling_position * day_spy

            # If we have too much cash sitting, it's important to move it to the
            # T-Note ladder, check monthly to simulate monthly auctioning.
            if day_num % 21 == 0 and cash >= 2 * current_nav * self.ladder_allocation / 5:
                tnote_maturity = 0
                if len(tnotes) == 0:
                    tnote_maturity = 252*2
                else:
                    furthest_maturity = max(m for _, (m, _) in tnotes.items()) - d
                    for m in [252*2, 252*3, 252*5]:
                        if furthest_maturity < m:
                            tnote_maturity = m
                            break
                assert tnote_maturity > 0
                tnote_tranche = current_nav * self.ladder_allocation / 5
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "buy",
                        "symbol": f"T-Note {int(tnote_maturity / 252)} Years",
                        "price": tnote_tranche,
                        "rate": self.tnote_rate,
                        "description": "T-Note ladder replenish"
                    })
                    transaction_day = True
                tnotes[tnote_id] = (d + tnote_maturity, tnote_tranche)
                tnote_id += 1
                cash -= tnote_tranche

            # Adjust monthly withdrawals for inflation each year
            if day_num % 252 == 0:
                monthly_withdrawal *= 1 + self.inflation

            # Mark to market all positions
            if full_book and transaction_day:
                self.book.append({
                    "day": d,
                    "trade": "mtm",
                    "symbol": "Cash",
                    "price": cash,
                    "description": "Cash MTM"
                })
                self.book.append({
                    "day": d,
                    "trade": "mtm",
                    "symbol": "SPY",
                    "price": day_spy,
                    "size": spy_position_size,
                    "total": spy_position_size * day_spy,
                    "description": "SPY MTM"
                })
                for _, (m, v) in tnotes.items():
                    self.book.append({
                        "day": d,
                        "trade": "mtm",
                        "symbol": "T-Note",
                        "maturity": m,
                        "price": v,
                        "description": "T-Note MTM"
                    })

            return_path[d] = current_nav

        return return_path


    @classmethod
    def from_json_object(cls, o):
        """
        Builds a lambda that given an underlying, monthly volatility and
        3 month forward volatility simulated paths returns an instance of
        this portfolio strategy from a JSON parsed object. The structure
        must have the following shape:
        {
            "type": "LongSPYWithTreasuryLadders",
            "equity_allocation": 0.9,
            "ladder_allocation": 0.1,
            "yearly_spending": 60_000,
            "monthly_tbill_rate": 0.038,
            "average_tnote_rate": 0.042,
            "dividend_yield": 0.01,
            "average_inflation": 0.034
        }
        """
        if o["type"] != "LongSPYWithTreasuryLadders":
            return None

        return LongSPYWithTreasuryLadders(
            o["equity_allocation"],
            o["ladder_allocation"],
            o["yearly_spending"],
            o["monthly_tbill_rate"],
            o["average_tnote_rate"],
            o["dividend_yield"],
            o["average_inflation"])
