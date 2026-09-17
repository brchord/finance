"""
linear_models.py

Contains a set of simple linear models that represent well-known
investment strategies modified for monthly real-space simulation paths:

Long SP500 with Treasuries ladders (Bills and Notes).
"""

import logging
import math

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Tuple, override

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaxLotTracker:
    """
    FIFO tax lot tracker for a single security within one simulated path.

    Tracks purchase lots (shares, cost basis per share, purchase month) and,
    on each sale, consumes the oldest lots first, splitting the realized gain
    into short-term and long-term based on a holding-period threshold.

    This intentionally only tracks *realized* gains -- unrealized gains on
    remaining lots are not taxable events and are left alone.
    """

    # Approximation: IRS long-term treatment requires holding for MORE than
    # 1 year (i.e. 366+ days). Since this engine steps monthly, "more than
    # 12 whole months held" is the closest we can resolve; a lot sold in
    # exactly its 12th month after purchase is treated as short-term here,
    # which is slightly conservative (biases toward the higher-tax bucket).
    LONG_TERM_HOLDING_MONTHS = 12

    # Below this share count a lot is considered fully consumed. Guards
    # against float accumulation leaving a lot "open" with ~1e-13 shares.
    _EPSILON_SHARES = 1e-9

    def __init__(self):
        # each: [shares_remaining, cost_basis_per_share, purchase_month]
        self.lots: deque[list] = deque()
        self.realized_short_term_gain: float = 0.0
        self.realized_long_term_gain: float = 0.0

    def buy(self, shares: float, price: float, month: int) -> None:
        """Opens a new lot. No-op for non-positive share counts."""
        if shares <= 0:
            return
        self.lots.append([float(shares), float(price), int(month)])

    def sell(self, shares_to_sell: float, price: float, month: int) -> Tuple[float, float]:
        """
        Consumes lots oldest-first to cover a sale of `shares_to_sell` at `price`.

        Returns:
        --------
        (short_term_gain, long_term_gain) : Tuple[float, float]
            Realized gain (can be negative, i.e. a loss) attributable to this
            single sale, split by holding-period bucket. Also accumulated
            into self.realized_short_term_gain / self.realized_long_term_gain.
        """
        remaining = float(shares_to_sell)
        st_gain = 0.0
        lt_gain = 0.0

        while remaining > self._EPSILON_SHARES and self.lots:
            lot = self.lots[0]
            lot_shares, lot_basis, lot_month = lot
            consumed = min(remaining, lot_shares)
            gain = consumed * (price - lot_basis)

            if (month - lot_month) >= self.LONG_TERM_HOLDING_MONTHS:
                lt_gain += gain
            else:
                st_gain += gain

            lot[0] -= consumed
            remaining -= consumed
            if lot[0] <= self._EPSILON_SHARES:
                self.lots.popleft()

        if remaining > self._EPSILON_SHARES:
            # Selling more shares than we have lots for is a bug upstream
            # (e.g. spy_position_size and the tracker have drifted apart).
            raise RuntimeError(
                f"TaxLotTracker: tried to sell {shares_to_sell} shares at month "
                f"{month} but only {shares_to_sell - remaining} were covered by open lots."
            )

        self.realized_short_term_gain += st_gain
        self.realized_long_term_gain += lt_gain
        return st_gain, lt_gain

    def total_shares_held(self) -> float:
        """Total open shares across all remaining lots (should match position size)."""
        return sum(lot[0] for lot in self.lots)

    def average_cost_basis(self) -> float:
        """Share-weighted average cost basis across currently open lots (0 if none open)."""
        total_shares = self.total_shares_held()
        if total_shares <= self._EPSILON_SHARES:
            return 0.0
        return sum(lot[0] * lot[1] for lot in self.lots) / total_shares


class InvestmentStrategy(ABC):
    """
    Abstract class representing a generic investment strategy.
    Designed to be inherited by classes representing different investment strategies.
    """

    def __init__(self):
        super().__init__()
        self.book: List[Dict] = []

    @classmethod
    def from_json_object(cls, o: dict):
        """Builds a portfolio instance out of a JSON parsed object."""
        raise NotImplementedError("Do not invoke this directly")

    @abstractmethod
    def run_simulation(
        self,
        *,
        spx: np.ndarray,         # SPX monthly time series
        cpi: np.ndarray,         # CPI monthly pct changes
        yield3m: np.ndarray,     # 3M T-Bill yield
        yield5y: np.ndarray,     # 5Y T-Note yield
        initial_nav: float,      # Initial NAV
        months: int,             # Total months to run the simulation
        full_book: bool = False, # Track full transaction book for debugging
    ) -> np.ndarray:
        """
        Starts the investment portfolio simulation on a monthly time step.
        returns: Time series array (length = months) representing monthly NAV progression.
        """
        return np.full(months, initial_nav)

    def transaction_book(self) -> List[Dict]:
        """Returns the full trading book for the last simulated path."""
        return self.book


class LongSPYWithTreasuryLadders(InvestmentStrategy):
    """
    Simulates a portfolio comprised of a long SPY sleeve and a fixed income T-Bill/T-Note ladder:
        - Short-term T-Bills (Cash buffer)
        - 2-year T-Note (24 months)
        - 3-year T-Note (36 months)
        - 5-year T-Note (60 months)

    Operates entirely on a monthly time increment (months).
    """

    def __init__(
        self,
        equity_allocation: float,
        ladder_allocation: float,
        yearly_spending: float,
        spy_avg_dividend_yield: float = 0.01,
    ):
        if not math.isclose(equity_allocation + ladder_allocation, 1.0, abs_tol=1e-4):
            raise ValueError("Portfolio allocation must sum up to 100%")
        super().__init__()
        self.equity_allocation = equity_allocation
        self.ladder_allocation = ladder_allocation
        self.yearly_spending = yearly_spending
        self.spy_div_yield = spy_avg_dividend_yield
        self.spy_lots = TaxLotTracker()
        self.realized_gains: List[Tuple[int, float, float]] = []

    @staticmethod
    def _get_needed_liquidity(
        monthly: float,
        current_month: int,
        tnotes: Dict[int, Tuple[int, float, float]],
    ) -> float:
        if len(tnotes) == 0:
            return monthly * 12.0

        closest_maturity = min(mat for _, (mat, _, _) in tnotes.items())
        months_needed = max(1, closest_maturity - current_month)
        return monthly * months_needed

    @override
    def run_simulation(
        self,
        *,
        spx: np.ndarray,
        cpi: np.ndarray,
        yield3m: np.ndarray,
        yield5y: np.ndarray,
        initial_nav: float,
        months: int,
        full_book: bool = False,
    ) -> np.ndarray:
        self.book.clear()
        self.spy_lots = TaxLotTracker()
        self.realized_gains = []
        monthly_withdrawal = self.yearly_spending / 12.0
        spx_prices = spx / 10.0
        spy_price = spx_prices[0]

        spy_position_size = math.ceil(initial_nav * self.equity_allocation / spy_price)
        self.spy_lots.buy(spy_position_size, spy_price, 0)
        tnote_amount = initial_nav * self.ladder_allocation / 5.0

        # Note maturities expressed in months: 2Y=24m, 3Y=36m, 5Y=60m
        tnotes: Dict[int, Tuple[int, float, float]] = {
            0: (24, tnote_amount, yield5y[0]),
            1: (36, tnote_amount * 2, yield5y[0]),
            2: (60, tnote_amount, yield5y[0]),
        }
        tnote_id = 3
        cash = initial_nav - (tnote_amount * 4) - (spy_price * spy_position_size)

        if full_book:
            self.book.append({
                "month": 0,
                "trade": "buy",
                "symbol": "SPY",
                "price": spy_price,
                "size": spy_position_size,
                "total": spy_price * spy_position_size,
                "description": f"Initial long equity at {self.equity_allocation * 100.0}%",
            })
            self.book.append({
                "month": 0,
                "trade": "buy",
                "symbol": "T-Note 2 Years",
                "price": tnote_amount,
                "rate": yield5y[0],
                "maturity_month": 24,
                "description": "Initial T-Note ladder setup",
            })
            self.book.append({
                "month": 0,
                "trade": "buy",
                "symbol": "T-Note 3 Years",
                "price": tnote_amount * 2,
                "rate": yield5y[0],
                "maturity_month": 36,
                "description": "Initial T-Note ladder setup",
            })
            self.book.append({
                "month": 0,
                "trade": "buy",
                "symbol": "T-Note 5 Years",
                "price": tnote_amount,
                "rate": yield5y[0],
                "maturity_month": 60,
                "description": "Initial T-Note ladder setup",
            })
            self.book.append({
                "month": 0,
                "trade": "buy",
                "price": cash,
                "rate": yield3m[0],
                "description": "Cash equivalents in short-term T-bills",
            })

        spx_series = pd.Series(spx_prices)
        # Mapped technical indicators from daily to monthly equivalents:
        # EMA-30 days ~ EMA-1.5 months; SMA-90 days ~ SMA-4 months; SMA-180 days ~ SMA-9 months
        sma2 = spx_series.rolling(window=2, min_periods=1).mean().values
        sma4 = spx_series.rolling(window=4, min_periods=1).mean().values
        sma9 = spx_series.rolling(window=9, min_periods=1).mean().values

        cpi_pct = pd.Series(cpi).pct_change().shift(-1)
        cpi_pct.iloc[-1] = 0.0

        return_path = np.zeros(months)

        for m in range(months):
            day_spy = spx_prices[m]
            transaction_month = False

            # Monthly withdrawal execution
            cash -= monthly_withdrawal
            if full_book:
                self.book.append({
                    "month": m,
                    "trade": "withdrawal",
                    "price": -monthly_withdrawal,
                    "description": "Monthly expense distribution",
                })
                transaction_month = True

            # Quarterly SPY Dividend Payment (Every 3 months)
            if m % 3 == 0 and m > 0:
                cash_dividend = spy_position_size * day_spy * self.spy_div_yield / 4.0
                if cash_dividend > 0:
                    cash += cash_dividend
                    if full_book:
                        self.book.append({
                            "month": m,
                            "trade": "dividend",
                            "price": cash_dividend,
                            "description": "SPY dividend payment",
                        })
                        transaction_month = True

            # Monthly T-Bill Yield Accrual
            tbill_monthly_return = yield3m[m] / 12.0
            cash *= (1.0 + tbill_monthly_return)

            # Semi-annual T-Note Coupon Distribution & Maturities (Every 6 months)
            expired_notes = []
            for idx, (maturity_m, amount, rate) in tnotes.items():
                months_remaining = maturity_m - m
                if months_remaining % 6 == 0 and m > 0:
                    tnote_coupon = amount * (rate / 2.0)
                    cash += tnote_coupon
                    if full_book:
                        self.book.append({
                            "month": m,
                            "trade": "dividend",
                            "price": tnote_coupon,
                            "description": "T-Note semi-annual coupon payment",
                        })
                        transaction_month = True

                if months_remaining <= 0:
                    expired_notes.append(idx)

            for expired_note in expired_notes:
                (mat, amount, rate) = tnotes.pop(expired_note)
                cash += amount
                if full_book:
                    self.book.append({
                        "month": m,
                        "trade": "deposit",
                        "price": amount,
                        "rate": rate,
                        "description": "T-Note maturity reached",
                    })
                    transaction_month = True

            tnote_position = sum(amount for _, (_, amount, _) in tnotes.items())
            current_nav = cash + (spy_position_size * day_spy) + tnote_position
            fixed_income_position = cash + tnote_position

            # Strategic Equity Rebalancing into Fixed Income
            if fixed_income_position / current_nav <= self.ladder_allocation * 0.8:
                if m >= 9 and day_spy >= sma2[m] and day_spy >= sma4[m] and day_spy >= sma9[m]:
                    needed_amount = (current_nav * self.ladder_allocation) - fixed_income_position
                    req_liquidity = self._get_needed_liquidity(monthly_withdrawal, m, tnotes)
                    if cash > req_liquidity:
                        extra_liquidity = cash - req_liquidity
                        amount_to_sell = min(needed_amount, extra_liquidity)
                        selling_position = math.floor(amount_to_sell / day_spy)
                        selling_position = min(selling_position, spy_position_size)
                        if selling_position > 0:
                            st_gain, lt_gain = self.spy_lots.sell(selling_position, day_spy, m)
                            self.realized_gains.append((m, st_gain, lt_gain))
                            if full_book:
                                self.book.append({
                                    "month": m,
                                    "trade": "sell",
                                    "symbol": "SPY",
                                    "size": selling_position,
                                    "price": day_spy,
                                    "short_term_gain": st_gain,
                                    "long_term_gain": lt_gain,
                                    "description": "Portfolio rebalance to replenish fixed income",
                                })
                                transaction_month = True
                            spy_position_size -= selling_position
                            cash += selling_position * day_spy

            # Reserve Deficit Protection (Sell Equity if Cash Runway < Threshold)
            req_liquidity = self._get_needed_liquidity(monthly_withdrawal, m, tnotes)
            div_events_expected = math.floor(req_liquidity / monthly_withdrawal / 3.0)
            expected_dividends = spy_position_size * spy_price * self.spy_div_yield * \
                (div_events_expected / 4.0)
            spending_needs = req_liquidity - expected_dividends
            current_runway = cash - spending_needs

            if current_runway < 0:
                selling_position = math.ceil(-current_runway / day_spy)
                selling_position = min(selling_position, spy_position_size)
                if selling_position > 0:
                    st_gain, lt_gain = self.spy_lots.sell(selling_position, day_spy, m)
                    self.realized_gains.append((m, st_gain, lt_gain))
                    if full_book:
                        self.book.append({
                            "month": m,
                            "trade": "sell",
                            "price": day_spy,
                            "size": selling_position,
                            "short_term_gain": st_gain,
                            "long_term_gain": lt_gain,
                            "description": "Selling equity shares to restore cash runway",
                        })
                        transaction_month = True
                    spy_position_size -= selling_position
                    cash += selling_position * day_spy

            # Surplus Cash Allocation into T-Note Ladder
            if cash >= (2.0 * current_nav * self.ladder_allocation / 5.0):
                tnote_maturity = 24
                if len(tnotes) > 0:
                    furthest_maturity = max(mat for _, (mat, _, _) in tnotes.items()) - m
                    for target_m in [24, 36, 60]:
                        if furthest_maturity < target_m:
                            tnote_maturity = target_m
                            break

                tnote_tranche = current_nav * self.ladder_allocation / 5.0
                if cash >= (tnote_tranche + spending_needs) and tnote_tranche > 0:
                    if full_book:
                        self.book.append({
                            "month": m,
                            "trade": "buy",
                            "symbol": f"T-Note {int(tnote_maturity / 12)} Years",
                            "price": tnote_tranche,
                            "rate": yield5y[m],
                            "description": "T-Note ladder replenishment",
                        })
                        transaction_month = True
                    tnotes[tnote_id] = (m + tnote_maturity, tnote_tranche, yield5y[m])
                    tnote_id += 1
                    cash -= tnote_tranche

            # Discount current reported inflation in the simulated CPI
            monthly_withdrawal *= 1.0 + cpi_pct[m]

            # Mark to Market Log
            if full_book and transaction_month:
                self.book.append({
                    "month": m,
                    "trade": "mtm",
                    "symbol": "Cash",
                    "price": cash,
                    "description": "Cash MTM",
                })
                self.book.append({
                    "month": m,
                    "trade": "mtm",
                    "symbol": "SPY",
                    "price": day_spy,
                    "size": spy_position_size,
                    "total": spy_position_size * day_spy,
                    "description": "SPY MTM",
                })
                for _, (mat, val, r) in tnotes.items():
                    self.book.append({
                        "month": m,
                        "trade": "mtm",
                        "symbol": "T-Note",
                        "maturity_month": mat,
                        "price": val,
                        "rate": r,
                        "description": "T-Note MTM",
                    })

            tnote_position = sum(amount for _, (_, amount, _) in tnotes.items())
            current_nav = cash + (spy_position_size * day_spy) + tnote_position
            # If we ran out of money, stop the loop and let the remaining path
            # vector to be zero-filled.
            if current_nav <= 0:
                break

            return_path[m] = current_nav

        return return_path

    def annual_realized_gains(self) -> Dict[int, tuple[float, float]]:
        """
        Rolls up self.realized_gains (one entry per sale) into calendar-year
        totals, keyed by simulation year (0-indexed: months 0-11 -> year 0).

        Returns:
        --------
        Dict[int, tuple[float, float]]
            Only years with at least one sale are present.
        """
        annual: Dict[int, tuple[float, float]] = {}
        for month, st_gain, lt_gain in self.realized_gains:
            year = month // 12
            t = annual.setdefault(year, (0.0, 0.0))
            annual[year] = (t[0] + st_gain, t[1] + lt_gain)
        return annual

    @classmethod
    def from_json_object(cls, o: dict):
        if o["type"] != "LongSPYWithTreasuryLadders":
            return None

        return LongSPYWithTreasuryLadders(
            float(o["equity_allocation"]),
            float(o["ladder_allocation"]),
            float(o["yearly_spending"]),
            float(o.get("dividend_yield", 0.01)),
        )
