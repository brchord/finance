"""
portfolio_simulation.py

This script provides a basic simulation for Stochastic Volatility
with Correlated Jumps (SVCJ) to be used to simulate equities
markets to later on be integrated into a Monte Carlo simulation
engine designed to stress test multiple trading strategies using
Conditional Value at Risk (CVaR).
"""

import math
import time

import numpy as np
import pandas as pd

import market_modelling.black_scholes as bs

# ==========================================
# 4. Trading simulation (Operator's Manual)
# ==========================================
class ShortSPXPutStrategy:
    """
    Represents a trading strategy involving the systematic
    selling of SPX Put Options overlaid on top of a fixed income
    (risk free rate) portfolio with the following operator rules:

    1. Systematically sell a specific given OTM delta put options
    2. The position size is governed by a total fraction of the
       notional value of the put position.
    3. When a specific delta is breach in the current options book,
       roll the option to the next available expected expiration
       at a credit.
    4. If the given option book crosses a specific profit %, close the
       existing position and redeploy a new short put position.
    5. If the 30 IV vs 90 IV time series enter backwardation, stop
       all shorting operations until the market comes back to a state
       of contango.
    6. Any short put position crossing a specific delta, will be closed
       to avoid accumulating delta and gamma risk.
    """
    def __init__(self,
                 spot_spx,         # Time series for SPX underlying price.
                 spot_vix,         # Time series for the spot VIX.
                 vix3m,            # Time series for the VIX3M.
                 svi,              # Stochastic Volatility Inspired IV Model.
                 rf_rate=0.03,     # Annualized risk free rate.
                 inflation=0.025,  # Annualized inflation rate.
                 delta=-0.15,      # Put Delta to use when shorting.
                 dtes=45,          # Short option expirations.
                 max_dtes=135,     # Option book maximium expiration permitted.
                 take_profit=0.75, # Take profits at percentage of each premium sold.
                 full_book=True):  # Track full options book for debugging.
        print(f"""Initializing short put portfolio strategy:"
          Initial SPX Spot: {spot_spx[0]:,.2f}
               Initial VIX: {math.sqrt(spot_vix[0]):.2f}%
             Initial VIX3M: {math.sqrt(vix3m[0]):.2f}%
            Risk Free Rate: {rf_rate*100.0:.2f}%
            Inflation Rate: {inflation*100.0:.2f}%
              Option Delta: {delta:.2f}
         Option Expiration: {dtes:.0f} DTEs
       Maximium Expiration: {max_dtes:.0f} DTEs
           Take profits at: {take_profit*100:.2f}% of premium sold
        Track Options Book: {full_book}
        """)
        self.spot = spot_spx
        self.vix = spot_vix
        self.vix3m = vix3m
        self.svi = svi
        self.risk_free_rate = rf_rate
        self.inflation = inflation
        self.delta = delta
        self.dtes = dtes
        self.limit_dtes = max_dtes
        self.take_profit = take_profit
        self.full_book = full_book
        self.book = []

    def _find_put_strike_by_price(self, spot, atm_iv, target_price, expiration):
        return self._find_put_strike(
            spot, atm_iv, target_price, expiration, use_delta=False)


    def _find_put_strike_by_delta(self, spot, atm_iv, target_delta, expiration):
        return self._find_put_strike(
            spot, atm_iv, target_delta, expiration, use_delta=True)


    def _find_put_strike(self, spot, atm_iv, target, expiration, use_delta):
        # Binary search to find the exact put strike that yields the target delta
        print(f"""
        Searching put option with parameters:
        Underlying price:  {spot:,.2f}
                  ATM IV:  {atm_iv * 100:.2f}""")
        if use_delta:
            print(f"""
            Target Delta:  {target:.2f}""")
        else:
            print(f"""
            Target Price: ${target:,.2f}""")
        print(f"""
                    DTEs:  {expiration*365:.0f}""")

        low = spot * 0.5
        high = spot
        r = self.risk_free_rate
        for _ in range(40):
            mid = (low + high) / 2.0
            iv = self.svi.get_iv_curve(atm_iv, mid, spot, expiration)
            delta = bs.option_delta(spot, mid, expiration, r, iv, is_call=False)
            price = bs.option_price(spot, mid, expiration, r, iv, is_call=False)

            print(f"""
              Trying strike: ${mid:.2f}
            IV estimated at:  {iv*100:.2f}%
                   Delta at:  {delta:.2f}
                   Price at: ${price:,.2f}""")

            if use_delta:
                abs_current = abs(delta)
            else:
                abs_current = abs(price)

            abs_target = abs(target)
            if abs(abs_current - abs_target) < 0.005:
                break
            if abs_current > abs_target:
                high = mid
            else:
                low = mid
        round_strike = int((low + high) / 20)
        return round_strike * 10, price, delta, iv


    def _roll_current_option(self, cur_day, spot, atm_iv,
                             position_size, orig_price, new_expiration,
                             target_profit_price):
        print(f"""Trying to roll put position with the following requirements:
              Simulation day:  {cur_day}
                        Spot: ${spot:,.2f}
                      ATM IV:  {atm_iv*100.0:.2f}%
      Original Position Size:  {position_size}
          Put original price: ${orig_price:,.2f}
              New Expiration:  {new_expiration:.0f} DTEs
               Target Profit: ${target_profit_price:,.2f}""")

        yearly_expiration = new_expiration/365.0
        strike, price, delta, iv = self._find_put_strike_by_price(
            spot, atm_iv, orig_price, yearly_expiration)
        premium = self._deploy_put(
            cur_day, strike, new_expiration, delta, iv,
            price, position_size, target_profit_price)
        return premium


    def _sell_put(self, cur_day, spot, atm_iv, nav,
                  leverage, delta, expiration, target_profit=0.0):
        print(f"""Trying to sell put with the following requirements:
         Simulation Day:  {cur_day}
                   Spot: ${spot:,.2f}
                 ATM IV:  {atm_iv*100.0:.2f}%
                    NAV: ${nav:,.2f}
               Leverage:  {leverage:.2f}x
                  Delta:  {delta:.2f}
             Expiration:  {expiration:.0f} DTEs
          Target Profit: ${target_profit:,.2f}""")
        yearly_exp = expiration / 365.0
        put_strike, put_price, _, put_iv = self._find_put_strike_by_delta(
            spot, atm_iv, delta, yearly_exp)
        position_size = round(nav * leverage / put_strike / 100)

        if abs(target_profit) < 1e-2:
            target_profit = put_price * (1.0 - self.take_profit)

        premium = self._deploy_put(
            cur_day, put_strike, expiration, delta, put_iv,
            put_price, position_size, target_profit)
        return premium


    def _deploy_put(self, day, strike, expiration, delta,
                    iv, price, size, target_profit):
        premium = size * price * 100
        print(f"""Deploying put option:
               Simulation Day:  {day}
                       Strike: ${strike:,.2f}
                   Expiration:  {expiration:,.0f} DTEs
                        Delta:  {delta:.2f}
                           IV:  {iv*100.0:.2f}%
                        Price: ${price:,.2f}
                Position size:  {size}
                Total premium: ${premium:,.2f}
                Target profit: ${target_profit:,.2f}
                """)

        # Trade day, Put strike, Initial Expiration, Delta, IV, Price, Size, Total Premium
        self.book.append([day, strike, expiration, delta, iv,
                          price, size, premium, target_profit])
        return premium


    def _rebuy_put(self, cur_day, put_strike, dtes,
                   put_delta, put_iv, put_price,
                   position_size, total_cost, final_profit):
        print(f"""Buying to close live options in book:
            Simulation day:  {cur_day}
                Put strike: ${put_strike:,.2f}
        Initial Expiration:  {dtes:.0f} DTEs
         Current put delta:  {put_delta:.2f}
            Current put IV:  {put_iv:.2f}
         Current put price: ${put_price:,.2f}
                     Units:  {position_size}
           Cash withdrawal: ${total_cost:,.2f}""")
        self.book.append([cur_day, put_strike, dtes, put_delta,
                          put_iv, put_price, position_size,
                          total_cost, final_profit])


    def run_simulation(self,
                       nav,                         # Initial portfolio NAV
                       monthly_distribution,        # Monthly withdrawals
                       notional_leverage = 0.5):    # Position size for options underwriting
        """
        Initialize the portfolio simulation with the given starting parameters:

                         nav: Initial portfolio's Net Asset Value
        monthly_distribution: How much money to withdraw monthly.
           notional_leverage: Amount of leverage based on notional value on the overlaid
                              short options position to be deployed.
        """

        print(f"""Initiating path simulation with the following parameters:
               Initial NAV: ${nav:,.2f}
        Monthly withdrawal: ${monthly_distribution:,.2f}
         Notional leverage:  {notional_leverage:.2f}x
        """)

        # Technical Indicators
        ema20 = pd.Series(self.spot).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = notional_leverage
        days = len(self.spot)

        cash = self._sell_put(
            0, self.spot[0], math.sqrt(self.vix[0]), nav,
            current_leverage, self.delta, self.dtes)

        print(f"""First short put trade:
             Initial cash after trade: ${cash:,.2f}""")

        for d in range(1, days):
            spot = self.spot[d]
            spot_vix = math.sqrt(self.vix[d])
            vix3m = math.sqrt(self.vix3m[d])
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= monthly_distribution
                print(f"""End of month.
                Subtracted distribution: ${monthly_distribution:,.2f}
                           cash balance: ${cash:,.2f}
                """)

            if spot > ema20[d]:
                days_above_ema += 1
            else:
                days_above_ema = 0

            nav = nav * (1 + r/365.0)

            # VIX Term Structure Check
            backwardation = spot_vix > vix3m * 1.05 # Avoid noise

            print(f"""
                Simulation at day:  {d}
                         SPX Spot: ${spot:,.2f}
                              VIX:  {spot_vix*100:.2f}%
                           VIX 3M:  {vix3m*100:.2f}%
            VIX in backwardation?:  {backwardation}
                             Cash: ${cash:,.2f}
                              NAV: ${nav:,.2f}
                    Options state:  {state}""")

            buy_to_close = False
            roll_option = False

            # TRIPLE-LOCK COOLDOWN RE-ENTRY
            if state == 'cooldown':
                if days_above_ema >= 5 and not backwardation:
                    state = 'wade_in'
                    current_leverage = notional_leverage / 2.0

                    # Deploy a given Delta Put with
                    # the specified DTEs.
                    prev_cash = cash
                    cash += self._sell_put(
                        d, spot, spot_vix, nav,
                        current_leverage, self.delta, self.dtes)
                    print(f"""New short put trade:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

            if state in ['active', 'wade_in']:
                # Price the latest options in the book
                last_book_entry = self.book[-1]
                put_strike = last_book_entry[1]
                put_orig_price = last_book_entry[5]
                put_size = last_book_entry[6]
                put_dtes = last_book_entry[2] + last_book_entry[0] - d
                put_target_profit = last_book_entry[8]
                put_exp = max(put_dtes,1)/365.0

                book_put_iv = self.svi.get_iv_curve(spot_vix, put_strike, spot, put_exp)
                book_put_price = bs.option_price(spot, put_strike, put_exp, r, book_put_iv, False)
                book_put_delta = bs.option_delta(spot, put_strike, put_exp, r, book_put_iv, False)

                # Operator's Decision Tree:
                # Option 1:
                #  If the VIX is in backwardation with VIX3M
                #  immediately close the options book and enter
                #  a cooldown period where no more premium is sold.
                if backwardation:
                    buy_to_close = True
                    state = 'cooldown'
                    days_above_ema = 0 # HARD EJECT
                # Option 2:
                #  If the live options delta has climbed to more than
                #  -0.50 avoid exposing the portfolio to more risk
                #  and buy to close at a loss.
                #
                #  Similar to the backwardation case, enter a cooldown
                #  period to avoid taking more risk.
                #  TODO: Revise the cooldown rule with CVaR calculations.
                #        It feels overly conservative.
                elif book_put_delta <= -0.50:
                    buy_to_close = True
                    state = 'cooldown'
                    days_above_ema = 0 # PRICE EJECT
                # Option 3:
                #  The option book has now a bag of options whose
                #  delta is unreasonably high (more than 0.35)
                #  so roll at the smallest credit possible to the
                #  next 45 DTEs cycle after its original expiration
                #  do this only to a limit of 120 DTEs.
                elif book_put_delta <= -0.35:
                    roll_option = True
                # Option 4:
                #  If we have hit the profit target, simply buy to
                #  close. If we were in cooldown period, start to
                #  wade back in using half of the target notional.
                elif book_put_price <= put_target_profit:
                    buy_to_close = True
                    if state == 'wade_in':
                        current_leverage = notional_leverage
                        state = 'active'
                # Option 5:
                #  If the options book expiration is now a week or
                #  less, close immediately to avoid more gamma
                #  risk exposure.
                elif put_dtes <= 7:
                    buy_to_close = True # 7-DTE HARD DECK (Gamma Risk Eject)

                if buy_to_close:
                    assert not roll_option
                    print("Trying to close option (not roll)")
                    book_close_cost = book_put_price * put_size * 100
                    profit = (put_orig_price - book_put_price) * put_size * 100
                    self._rebuy_put(d, put_strike, last_book_entry[2],
                                    book_put_delta, book_put_iv, book_put_price,
                                    put_size, -book_close_cost, profit)
                    prev_cash = cash
                    cash -= book_close_cost
                    print(f"""Deducting option book buy to close:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_put(
                            d, spot, spot_vix, nav,
                            current_leverage, self.delta, self.dtes)

                if roll_option:
                    assert not buy_to_close
                    print("Trying to roll current options position")
                    book_close_cost = book_put_price * put_size * 100
                    profit = (put_orig_price - book_put_price) * put_size * 100
                    self._rebuy_put(d, put_strike, last_book_entry[2],
                                    book_put_delta, book_put_iv, book_put_price,
                                    put_size, -book_close_cost, profit)

                    prev_cash = cash
                    cash -= book_close_cost

                    print(f"""Deducting option book buy to close and roll:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    remaining_exp = last_book_entry[2] + last_book_entry[0] - d
                    new_expiration = last_book_entry[2] + self.dtes

                    if new_expiration < self.limit_dtes:
                        cash += self._roll_current_option(
                            d, spot, spot_vix, put_size,
                            book_put_price, new_expiration,
                            put_orig_price * (1.0 - self.take_profit))
                        print(f"""
                     Rolled option remaining: {remaining_exp:.0f} DTEs
                    New put expiration to be: {new_expiration:.0f} DTEs
                            Expiration limit: {self.limit_dtes:.0f} DTEs""")
                    else:
                        print(f"""
                        Current options position exceeded the expiration
                        limit permitted of {self.limit_dtes:.0f} DTEs.
                        Selling new option instead.
                        """)
                        cash += self._sell_put(
                            d, spot, spot_vix, nav,
                            current_leverage, self.delta, self.dtes)


            # Reinvest (or subtract) new cash flows.
            if abs(cash) > 0.01:
                prev_nav = nav
                prev_cash = cash
                nav = nav + cash
                cash = 0.0
                print(f"""Reinvesting / withdrawing cash after options trade:
                Previous NAV: ${prev_nav:,.2f}
                After-trade NAV: ${nav:,.2f}
                Previous Cash: ${prev_cash:,.2f}""")
