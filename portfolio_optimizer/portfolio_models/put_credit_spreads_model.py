"""
put_credit_spreads_model.py

Implements the mechanics of a path dependent
continous selling of Put Credit Spreads against
a market, volatility time series leveraging the
SVI (stochastic volatility inspired) model to
properly price short and long put options.

See the class documentation for the specific
dynamics of the trading strategy.
"""

import math
from typing import override

import numpy as np
import pandas as pd

from market_modelling import black_scholes as bs
from market_modelling.dsvi import DynamicSVI
from portfolio_models.short_put_model import SPXPutOptionStrategy

class SPXPutCreditSpreadStrategy(SPXPutOptionStrategy):
    """
    Represents a trading strategy involving selling
    SPX Put Credit Spreads (PCS) on top of a fixed income
    overlay with the following operator's procedure:

    1. Systematically sell a specific given OTM delta
       PCS with a given delta spread between the short/long legs.
    2. The position size is governed by a total fraction of the
       notional value of the put position.
    3. When a specific delta is breached in the short put leg on
       the book, close the entire position and redeploy a new spread.
    4. If the given live options book crosses a specific profit % for
       the spreads, close the existing position and redeploy a new
       PCS.
    5. If the 30 IV vs 90 IV time series enter backwardation, stop
       all shorting operations until the market comes back to a state
       of contango and the SPX underlying EMA-20 is crossed by the spot.
    6. Any PCS live in the book will be closed if they last for longer
       than a specific tail expirating to avoid taking unnecessary
       gamma risk.
    """
    def __init__(self, *,
                 distribution,         # Monthly retirement distribution.
                 leverage=1.0,         # Percentage of NAV notional leverage.
                 rf_rate=0.03,         # Risk free rate.
                 dtes=45,              # Spread initial expiration.
                 short_delta=-0.15,    # Short put delta.
                 delta_spread=0.1,     # Delta spread between short and long.
                 profit_target=0.75,   # Target profit close percentage.
                 dtes_to_close=10,     # Minimum DTEs to have a live spread.
                 delta_threshold=-0.4, # Short delta threshold to stop loss.
                 full_book=True):      # Track every option trade.
        super().__init__(rf_rate=rf_rate, full_book=full_book)
        self.log(f"""Initializing Put Credit Spreads portfolio strategy:"
       Monthly Withdrawals: ${distribution:,.2f}
         Notional Leverage:  {leverage:.2f}x of NAV
  Option Spread Expiration:  {dtes:.0f} DTEs
        Short Option Delta:  {short_delta:.2f}
              Delta Spread:  {delta_spread:.2f}
           Take profits at:  {profit_target*100:.2f}% of premium sold
       Maximium Expiration:  {dtes_to_close:.0f} DTEs
     Delta Close Threshold:  {delta_threshold:.2f}""")
        self.monthly_dist = distribution
        self.leverage = leverage
        self.spread_dtes = dtes
        self.short_delta = short_delta
        self.delta_spread = delta_spread
        self.profit_target = profit_target
        self.dtes_to_close = dtes_to_close
        self.delta_threshold = delta_threshold


    @classmethod
    @override
    def from_json_object(cls, o):
        """
        Builds a lambda that given an underlying, monthly volatility and
        3 month forward volatility simulated paths returns an instance of
        this portfolio strategy from a JSON parsed object. The structure
        must have the following shape:
        {
            "distribution"
        }
        """
        if o["type"] != "SPXPutCreditSpreadStrategy":
            return None
        distribution = float(o["distribution"])
        leverage = float(o["leverage"])
        rf_rate = float(o["rf_rate"])
        dtes = float(o["dtes"])
        short_delta = float(o["short_delta"])
        delta_spread = float(o["delta_spread"])
        profit_target = float(o["profit_target"])
        dtes_to_close = float(o["dtes_to_close"])
        delta_threshold = float(o["delta_threshold"])

        return SPXPutCreditSpreadStrategy(
            distribution=distribution,
            leverage=leverage,
            rf_rate=rf_rate,
            dtes=dtes,
            short_delta=short_delta,
            delta_spread=delta_spread,
            profit_target=profit_target,
            dtes_to_close=dtes_to_close,
            delta_threshold=delta_threshold)


    def _buy_to_open_put(self,
                         day,    # Simulation day
                         spot,   # SPX Spot
                         atm_iv, # ATM Implied Volatility
                         delta,  # Put delta
                         dtes,   # Option expiration
                         size):  # Position size
        self.log(f"""Opening a new long put position with the following requirements:
         Simulation Day:  {day}
                   Spot: ${spot:,.2f}
                 ATM IV:  {atm_iv*100.0:.2f}%
                  Delta:  {delta:.2f}
             Expiration:  {dtes:.0f} DTEs
          Position Size:  {size} contracts.""")
        yr_exp = dtes / 365.0
        put_strike, put_price, _, put_iv = self._find_put_strike_by_delta(
            spot, atm_iv, delta, yr_exp)
        return self._write_put_trade_to_book(
            day=day, trade='bto', strike=put_strike, expiration=dtes,
            delta=delta, iv=put_iv, price=put_price, size=size)


    def _sell_to_close_put(
            self,
            day,         # Simulation day.
            strike,      # Put Strike.
            dtes,        # Remaining expiration.
            delta,       # Put Delta.
            iv,          # Put IV.
            price,       # Current put price.
            size,        # Position size.
            orig_price): # Put opening price.
        pnl = (price - orig_price) * size * 100.0

        self.log(f"""Selling to close live options in book:
            Simulation day:  {day}
                Put strike: ${strike:,.2f}
        Initial Expiration:  {dtes:.0f} DTEs
         Current put delta:  {delta:.2f}
            Current put IV:  {iv*100.0:.2f}%
         Current put price: ${price:,.2f}
                     Units:  {size}
            Original price: ${orig_price:,.2f}
              Position PnL: ${pnl:,.2f}""")
        return self._write_put_trade_to_book(
                    day=day, trade='stc', strike=strike,
                    expiration=dtes, delta=delta, iv=iv, price=price,
                    size=size, pnl=pnl)


    def _sell_put_credit_spread(
            self,
            day,          # Simulation day.
            spot,         # SPX Spot.
            atm_iv,       # ATM Implied Volatility.
            nav,          # Portfolio NAV.
            leverage,     # Short put notional leverage.
            short_delta,  # Short put target delta.
            delta_spread, # Short-long delta spread.
            dtes,         # Spread initial expiration.
            profit_pct):  # Take profit at given percentage.
        self.log(f"""Selling put credit spread with the folloing parameters:
            Simulation day:  {day}
            SPX Spot Price: ${spot:,.2f}
                    ATM IV:  {atm_iv*100:.2f}%
             Portfolio NAV: ${nav:,.2f}
         Notional Leverage:  {leverage:.2f}x
           Short Put Delta:  {short_delta:.2f}
   Short-Long Delta Spread:  {delta_spread:.2f}
                Expiration:  {dtes:.0f} DTEs
             Profit target:  {profit_pct*100:.2f}%""")
        short_credit = self._sell_to_open_put(
            day, spot, atm_iv, nav, leverage,
            short_delta, dtes, profit_pct)
        if len(self.live_options_book['sto']) == 0:
            assert short_credit == 0.0
            self.log("Portfolio NAV too low to sell even a single spread at "
                     "the given notional leverage.  Skipping...")
            return 0.0

        position_size = self.live_options_book['sto'][0]["size"]
        long_delta = short_delta + delta_spread
        assert short_delta < long_delta < 0.0
        long_debit = self._buy_to_open_put(
            day, spot, atm_iv, long_delta, dtes, position_size)
        total_credit = short_credit + long_debit
        return total_credit


    def run_simulation(
        self, *,
        spot_spx: list[float],  # Time series for SPX underlying price.
        spot_vix: list[float],  # Time series for the spot VIX.
        vix3m: list[float],     # Time series for the VIX3M.
        svi: DynamicSVI,        # Stochastic Volatility Inspired IV Model.
        initial_nav: float,     # NAV to start the simulation with.
        days: int) -> np.array: # Days to run the simulation
        """Run portfolio simulation (see parent's class docstring)."""
        self.log(f"""Starting simulation, initial parameters:
            Initial NAV: ${initial_nav:,.2f}
            Days to run:  {days}""")

        assert days <= len(spot_spx)

        # Technical Indicators
        ema20 = pd.Series(spot_spx).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = self.leverage
        nav = initial_nav

        cash = self._sell_put_credit_spread(
            0, spot_spx[0], math.sqrt(spot_vix[0]), nav, current_leverage,
            self.short_delta, self.delta_spread, self.spread_dtes,
            self.profit_target)

        self.log(f"""First Put Credit Spread trade:
             Initial cash after trade: ${cash:,.2f}""")

        return_path = np.zeros(days)
        return_path[0] = nav

        for d in range(1, days):
            today_spot = spot_spx[d]
            today_vix = math.sqrt(spot_vix[d])
            today_vix3m = math.sqrt(vix3m[d])
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= self.monthly_dist
                self.log(f"""End of month.
                Subtracted distribution: ${self.monthly_dist:,.2f}
                           cash balance: ${cash:,.2f}""")

            if today_spot > ema20[d]:
                days_above_ema += 1
            else:
                days_above_ema = 0

            nav = nav * (1 + r/252.0)

            # VIX Term Structure Check
            backwardation = today_vix > today_vix3m * 1.05 # Avoid noise

            self.log(f"""
                Simulation at day:  {d}
                         SPX Spot: ${today_spot:,.2f}
                              VIX:  {today_vix*100:.2f}%
                           VIX 3M:  {today_vix3m*100:.2f}%
            VIX in backwardation?:  {backwardation}
                             Cash: ${cash:,.2f}
                              NAV: ${nav:,.2f}
                    Options state:  {state}""")

            buy_to_close = False

            if state == 'cooldown':
                if days_above_ema >= 5 and not backwardation:
                    state = 'wade_in'
                    current_leverage = self.leverage / 2.0

                    # Deploy a given PCS with
                    # the specified DTEs.
                    prev_cash = cash
                    spread_credit = self._sell_put_credit_spread(
                        d, today_spot, today_vix, nav, current_leverage,
                        self.short_delta, self.delta_spread,
                        self.spread_dtes, self.profit_target)
                    cash += spread_credit

                    self.log(f"""New put credit spread trade:
                        Spread credit: ${spread_credit:,.2f}
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

            if state in ['active', 'wade_in']:
                # Get the latest entry in the option ledger.
                if len(self.live_options_book['sto']) == 0:
                    # No live options available, sell a new spread
                    self.log("No live option trades in book, attempting to sell"
                             "new put credit spreads:")
                    cash += self._sell_put_credit_spread(
                        d, today_spot, today_vix, nav, current_leverage, self.short_delta,
                        self.delta_spread, self.spread_dtes, self.profit_target)

                    # Reinvest (or subtract) new cash flows.
                    if abs(cash) > 0.01:
                        prev_nav = nav
                        prev_cash = cash
                        nav = nav + cash
                        cash = 0.0
                        self.log(f"""Reinvesting / withdrawing cash after options trade:
                        Previous NAV: ${prev_nav:,.2f}
                        After-trade NAV: ${nav:,.2f}
                        Previous Cash: ${prev_cash:,.2f}""")

                    continue

                lbe_short = self.live_options_book['sto'][0]
                spread_dtes = lbe_short["expiration"] + lbe_short["day"] - d
                spread_target_profit = lbe_short["target_profit"]
                spread_exp = max(spread_dtes,1)/365.0

                # Get data for the short leg
                short_put_strike = lbe_short["strike"]
                short_put_orig_price = lbe_short["price"]
                short_put_size = lbe_short["size"]
                assert spread_target_profit is not None

                book_short_put_iv = svi.get_iv_curve(
                    today_vix, short_put_strike, today_spot, spread_exp)
                book_short_put_price = bs.option_price(
                    today_spot, short_put_strike, spread_exp, r, book_short_put_iv, False)
                book_short_put_delta = bs.option_delta(
                    today_spot, short_put_strike, spread_exp, r, book_short_put_iv, False)

                # Get data for the long leg
                lbe_long = self.live_options_book['bto'][0]
                long_put_strike = lbe_long["strike"]
                long_put_orig_price = lbe_long["price"]
                long_put_size = lbe_long["size"]
                assert spread_target_profit is not None

                book_long_put_iv = svi.get_iv_curve(
                    today_vix, long_put_strike, today_spot, spread_exp)
                book_long_put_price = bs.option_price(
                    today_spot, long_put_strike, spread_exp, r, book_long_put_iv, False)
                book_long_put_delta = bs.option_delta(
                    today_spot, long_put_strike, spread_exp, r, book_long_put_iv, False)

                spread_book_price = book_short_put_price - book_long_put_price
                spread_orig_price = short_put_orig_price - long_put_orig_price
                spread_target_profit_price = (1.0 - self.profit_target) * spread_orig_price

                # Operator's Decision Tree:
                # Option 1:
                #  If the VIX is in backwardation with VIX3M
                #  immediately close the options book and enter
                #  a cooldown period where no more premium is sold.
                if backwardation:
                    self.log("Entered a state of backwardation between"
                             "VIX and VIX3M.")
                    buy_to_close = True
                    state = 'cooldown'
                    days_above_ema = 0
                # Option 2:
                #  If the short option leg delta has climbed to more than
                #  the specified threshold avoid exposing the portfolio
                #  to more risk and buy to close at a loss.
                #
                #  Similar to the backwardation case, enter a cooldown
                #  period to avoid taking more risk.
                #  TODO: Revise the cooldown rule with CVaR calculations.
                #        It feels overly conservative.
                elif book_short_put_delta <= self.delta_threshold:
                    self.log(f"""Short put leg delta threshold crossed:
                    Current short put delta: {book_short_put_delta:.2f}
                            Delta Threshold: {self.delta_threshold:.2f}""")
                    buy_to_close = True
                    state = 'cooldown'
                    days_above_ema = 0
                # Option 3:
                #  If we have hit the profit target, simply buy to
                #  close. If we were in cooldown period, start to
                #  wade back in using half of the target notional.
                elif spread_book_price <= spread_target_profit_price:
                    self.log(f"""Credit spread target profit reached:
                    Spread book price: ${spread_book_price:,.2f}
                Target price to close: ${spread_target_profit_price:,.2f}""")
                    buy_to_close = True
                    if state == 'wade_in':
                        current_leverage = self.leverage
                        state = 'active'
                # Option 4:
                #  If the options book expiration is now past
                #  the tail expiration limit, close immediately to
                #  avoid more gamma risk exposure.
                elif spread_dtes <= self.dtes_to_close:
                    self.log("Credit spread reached terminal expiration "
                            f"of {spread_dtes} DTEs, closing.")
                    buy_to_close = True

                if buy_to_close:
                    self.log("Closing credit spread.")
                    short_put_close_debit = self._buy_to_close_put(
                        d, short_put_strike, spread_dtes, book_short_put_delta,
                        book_short_put_iv, book_short_put_price, short_put_size,
                        short_put_orig_price)
                    long_put_close_credit = self._sell_to_close_put(
                        d, long_put_strike, spread_dtes, book_long_put_delta,
                        book_long_put_iv, book_long_put_price, long_put_size,
                        long_put_orig_price)
                    spread_close_debit = long_put_close_credit + short_put_close_debit
                    assert spread_close_debit < 0.0

                    prev_cash = cash
                    cash += spread_close_debit

                    self.log(f"""Deducting put credit spread closing cost:
                 Spread closing debit: ${spread_close_debit:,.2f}
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_put_credit_spread(
                            d, today_spot, today_vix, nav, current_leverage, self.short_delta,
                            self.delta_spread, self.spread_dtes, self.profit_target)

            # Reinvest (or subtract) new cash flows.
            if abs(cash) > 0.01:
                prev_nav = nav
                prev_cash = cash
                nav = nav + cash
                cash = 0.0
                self.log(f"""Reinvesting / withdrawing cash after options trade:
                Previous NAV: ${prev_nav:,.2f}
                After-trade NAV: ${nav:,.2f}
                Previous Cash: ${prev_cash:,.2f}""")

            return_path[-1] = nav

        # In order to make this portfolio stitchable, we need to close
        # all live option positions at the end of the simulation.
        if len(self.live_options_book['sto']) == 1:
            assert len(self.live_options_book['bto']) == 1
            self.log("Closing remaining option positions:")

            # Short Leg
            # TODO: Consider integrating all this logic in the private
            #       method _buy_to_close_put
            lbe = self.live_options_book['sto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = spot_vix[-1]
            spx = spot_spx[-1]
            r = self.risk_free_rate
            book_put_iv = svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(spx, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(spx, strike, yr_exp, r, book_put_iv, False)
            short_close_debit = self._buy_to_close_put(
                days - 1, strike, exp, book_put_delta, book_put_iv,
                book_put_price, lbe["size"], lbe["price"])

            # Long leg
            lbe = self.live_options_book['bto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = spot_vix[-1]
            spx = spot_spx[-1]
            r = self.risk_free_rate
            book_put_iv = svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(spx, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(spx, strike, yr_exp, r, book_put_iv, False)
            long_close_credit = self._sell_to_close_put(
                days - 1, strike, exp, book_put_delta, book_put_iv,
                book_put_price, lbe["size"], lbe["price"])

            total_debit = long_close_credit - short_close_debit
            assert total_debit > 0.0

            return_path[-1] += total_debit

        return return_path
