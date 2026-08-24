"""
option_models.py

Implements SPX put option models including options pricing
proper volatility surface pricing, trading mechanics which
crystalize into 2 well known investment strategies:

1. Short naked SPX puts over a T-bill overlay.
2. Sell SPX put credit spreads over a T-bill overlay.
"""

import math
from abc import ABC, abstractmethod

import pandas as pd

from market_modelling.dsvi import DynamicSVI
from market_modelling import black_scholes as bs

from portfolio_models.linear_models import InvestmentStrategy

class SPXPutOptionStrategy(InvestmentStrategy, ABC):
    """
    Abstract class that encapsulates common elements on
    short put volatility strategies.  Not meant for direct
    instantiation but useful to share common code between
    selling naked SPX Puts and selling SPX Put Credit
    Spreads.
    """
    _option_trade_name_map = {
        "btc": "Buy to close",
        "bto": "Buy to open",
        "stc": "Sell to close",
        "sto": "Sell to open"
    }

    def __init__(self,
                     *,
                     spot_spx: list[float], # Time series for SPX underlying price.
                     spot_vix: list[float], # Time series for the spot VIX.
                     vix3m: list[float],    # Time series for the VIX3M.
                     svi: DynamicSVI,       # Stochastic Volatility Inspired IV Model.
                     rf_rate=0.03,          # Annualized risk free rate.
                     full_book=True):       # Track full options book for debugging.
        print(f"""Initializing SPX put portfolio strategy:"
            Initial SPX Spot: ${spot_spx[0]:,.2f}
                Initial VIX:  {math.sqrt(spot_vix[0]):.2f}%
                Initial VIX3M:  {math.sqrt(vix3m[0]):.2f}%
            Risk Free Rate:  {rf_rate*100.0:.2f}%
        Track Options Book:  {full_book}
        """)
        self.spot = spot_spx
        self.vix = spot_vix
        self.vix3m = vix3m
        self.svi = svi
        self.risk_free_rate = rf_rate
        self.live_options_book = {
            'sto': [],
            'bto': []
        }

        if full_book:
            self.options_trade_book = []


    def _find_put_strike_by_delta(self, spot: float, atm_iv: float,
                                  target_delta: float, expiration: float):
        return self._find_put_strike(
            spot, atm_iv, target_delta, expiration, use_delta=True)


    def _find_put_strike(self, spot: float, atm_iv: float,
                         target: float, expiration: float, use_delta: bool):
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
                    DTEs:  {expiration*365.0:.0f}""")

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


    def _sell_to_open_put(self, cur_day: int, spot: float, atm_iv: float,
                          nav: float, leverage: float, delta: float,
                          expiration: float, target_profit_pct: float):
        print(f"""Trying to sell put with the following requirements:
         Simulation Day:  {cur_day}
                   Spot: ${spot:,.2f}
                 ATM IV:  {atm_iv*100.0:.2f}%
                    NAV: ${nav:,.2f}
               Leverage:  {leverage:.2f}x
                  Delta:  {delta:.2f}
             Expiration:  {expiration:.0f} DTEs
         Target Profit%:  {target_profit_pct:.2f}%""")
        yearly_exp = expiration / 365.0
        put_strike, put_price, _, put_iv = self._find_put_strike_by_delta(
            spot, atm_iv, delta, yearly_exp)
        position_size = math.floor(nav * leverage / put_strike / 100)
        # NAV is too low to create even a single contract, bail early.
        if position_size == 0:
            print("Option notional too large to sell "
                  "even a single contract, skipping.")
            return 0.0

        target_profit = (1.0 - target_profit_pct) * put_price
        return self._write_put_trade_to_book(
            day=cur_day, trade='sto', strike=put_strike, expiration=expiration,
            delta=delta, iv=put_iv, price=put_price,
            size=position_size, target_profit=target_profit)


    def _write_put_trade_to_book(self, *,
            day: int, trade: str, strike: float, expiration: float,
            delta: float, iv: float, price: float, size: int,
            target_profit=None, pnl=None):
        trade = trade.lower()
        assert trade in SPXPutOptionStrategy._option_trade_name_map
        trade_str = SPXPutOptionStrategy._option_trade_name_map[trade]

        assert size > 0

        if trade in ['sto']:
            assert target_profit

        if trade in ['btc', 'stc']:
            assert pnl

        debit_credit = size * price * 100
        if trade in ['bto', 'btc']:
            debit_credit *= -1.0

        print(f"""Recording option trade in book:
               Simulation Day:  {day}
                        Trade:  {trade_str}
                       Strike: ${strike:,.2f}
                   Expiration:  {expiration:,.0f} DTEs
                        Delta:  {delta:.2f}
                           IV:  {iv*100.0:.2f}%
                        Price: ${price:,.2f}
                Position size:  {size}
           Total debit/credit: ${debit_credit:,.2f}""")
        if target_profit:
            print(f"                Target profit: ${target_profit:,.2f}")
        if pnl:
            print(f"                          PnL: ${pnl:,.2f}")

        # Trade day, Put strike, Initial Expiration, Delta, IV, Price, Size, Total Premium
        book_entry =  {
            "day": day,
            "strike": strike,
            "expiration": expiration,
            "delta": delta,
            "iv": iv,
            "price": price,
            "size": size
        }
        if target_profit:
            book_entry["target_profit"] = target_profit
        if pnl:
            book_entry["pnl"] = pnl

        if trade == 'bto':
            assert len(self.live_options_book['bto']) == 0
            self.live_options_book['bto'].append(book_entry)
        if trade == 'sto':
            assert len(self.live_options_book['sto']) == 0
            self.live_options_book['sto'].append(book_entry)
        if trade == 'btc':
            assert len(self.live_options_book['sto']) == 1
            self.live_options_book['sto'].clear()
        if trade == 'stc':
            assert len(self.live_options_book['bto']) == 1
            self.live_options_book['bto'].clear()

        if self.options_trade_book:
            self.options_trade_book.append(book_entry)

        return debit_credit


    def _buy_to_close_put(self, cur_day: int, put_strike: float, dtes: float,
                          put_delta: float, put_iv: float, put_price: float,
                          position_size: int, orig_price: float):
        pnl = (orig_price - put_price) * position_size * 100

        print(f"""Buying to close live options in book:
            Simulation day:  {cur_day}
                Put strike: ${put_strike:,.2f}
        Initial Expiration:  {dtes:.0f} DTEs
         Current put delta:  {put_delta:.2f}
            Current put IV:  {put_iv:.2f}
         Current put price: ${put_price:,.2f}
                     Units:  {position_size}
            Original price: ${orig_price:,.2f}
              Position PnL: ${pnl:,.2f}""")
        return self._write_put_trade_to_book(
            day=cur_day, trade='btc', strike=put_strike,
            expiration=dtes, delta=put_delta, iv=put_iv, price=put_price,
            size=position_size, pnl=pnl)


    @abstractmethod
    def run_simulation(self, initial_nav, days, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        return super().run_simulation(initial_nav, days, daily_nav)


class ShortSPXPutStrategy(SPXPutOptionStrategy):
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
       of contango and the SPX underlying EMA-20 is crossed by the spot.
    6. Any short put position crossing a specific delta, will be closed
       to avoid accumulating delta and gamma risk.
    """
    def __init__(self,
                 *,
                 spot_spx: list[float], # Time series for SPX underlying price.
                 spot_vix: list[float], # Time series for the spot VIX.
                 vix3m: list[float],    # Time series for the VIX3M.
                 svi: DynamicSVI,       # Stochastic Volatility Inspired IV Model.
                 distribution: float,   # Monthly withdrawals
                 leverage=0.5,          # Short put notional position size
                                        # based on a percentage NAV.
                 rf_rate=0.03,          # Annualized risk free rate.
                 inflation=0.025,       # Annualized inflation rate.
                 delta=-0.15,           # Put Delta to use when shorting.
                 dtes=45,               # Short option expirations.
                 max_dtes=135,          # Option book maximium expiration permitted.
                 take_profit=0.75,      # Take profits at percentage of each premium sold.
                 full_book=True):       # Track full options book for debugging.
        super().__init__(
            spot_spx=spot_spx,
            spot_vix=spot_vix,
            vix3m=vix3m,
            svi=svi,
            rf_rate=rf_rate,
            full_book=full_book)

        print(f"""Initializing short put portfolio strategy:"
       Monthly Withdrawals: ${distribution:,.2f}
         Notional Leverage:  {leverage:.2f}x of NAV
            Inflation Rate:  {inflation*100.0:.2f}%
              Option Delta:  {delta:.2f}
         Option Expiration:  {dtes:.0f} DTEs
       Maximium Expiration:  {max_dtes:.0f} DTEs
           Take profits at:  {take_profit*100:.2f}% of premium sold
        """)
        self.monthly_dist = distribution
        self.leverage = leverage
        self.inflation = inflation
        self.delta = delta
        self.dtes = dtes
        self.limit_dtes = max_dtes
        self.take_profit = take_profit


    def _find_put_strike_by_price(self, spot: float, atm_iv: float,
                                  target_price: float, expiration: float):
        return self._find_put_strike(
            spot, atm_iv, target_price, expiration, use_delta=False)


    def _roll_current_put_position(self, cur_day: int, spot: float, atm_iv: float,
                                   position_size: int, orig_price: float,
                                   new_expiration: float, target_profit_price: float):
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
        premium = self._write_put_trade_to_book(
            day=cur_day, trade='sto', strike=strike, expiration=new_expiration,
            delta=delta, iv=iv, price=price, size=position_size,
            target_profit=target_profit_price)
        return premium


    def run_simulation(self, initial_nav: float, days: int, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        print(f"""Starting simulation, initial parameters:
            Initial NAV: ${initial_nav:,.2f}
            Days to run:  {days}
       Report daily NAV:  {daily_nav}""")

        assert days <= len(self.spot)

        # Technical Indicators
        ema20 = pd.Series(self.spot).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = self.leverage
        nav = initial_nav

        cash = self._sell_to_open_put(
            0, self.spot[0], math.sqrt(self.vix[0]),
            nav, current_leverage, self.delta,
            self.dtes, self.take_profit)

        print(f"""First short put trade:
             Initial cash after trade: ${cash:,.2f}""")

        return_path = [nav]

        for d in range(1, days):
            spot = self.spot[d]
            spot_vix = math.sqrt(self.vix[d])
            vix3m = math.sqrt(self.vix3m[d])
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= self.monthly_dist
                print(f"""End of month.
                Subtracted distribution: ${self.monthly_dist:,.2f}
                           cash balance: ${cash:,.2f}
                """)

            if spot > ema20[d]:
                days_above_ema += 1
            else:
                days_above_ema = 0

            nav = nav * (1 + r/252.0)

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
                    current_leverage = self.leverage / 2.0

                    # Deploy a given Delta Put with
                    # the specified DTEs.
                    prev_cash = cash
                    cash += self._sell_to_open_put(
                        d, spot, spot_vix, nav,
                        current_leverage, self.delta, self.dtes,
                        self.take_profit)
                    print(f"""New short put trade:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

            if state in ['active', 'wade_in']:
                # If there are no live option positions, open one now.
                if len(self.live_options_book['sto']) == 0:
                    print("No current live options in the book, selling new ones")
                    cash += self._sell_to_open_put(
                        d, spot, spot_vix, nav,
                        current_leverage, self.delta,
                        self.dtes, self.take_profit)

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
                    continue

                # Get the latest entry in the option ledger.
                lbe = self.live_options_book['sto'][0]
                put_strike = lbe["strike"]
                put_orig_price = lbe["price"]
                put_size = lbe["size"]
                put_dtes = lbe["expiration"] + lbe["day"] - d
                put_target_profit = lbe["target_profit"]
                assert put_target_profit

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
                        current_leverage = self.leverage
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
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"],
                        book_put_delta, book_put_iv, book_put_price,
                        put_size, put_orig_price)
                    prev_cash = cash
                    cash += book_debit
                    print(f"""Deducting option book buy to close:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_to_open_put(
                            d, spot, spot_vix, nav,
                            current_leverage, self.delta,
                            self.dtes, self.take_profit)

                if roll_option:
                    assert not buy_to_close
                    print("Trying to roll current options position")
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"], book_put_delta,
                        book_put_iv, book_put_price, put_size, put_orig_price)

                    prev_cash = cash
                    cash += book_debit

                    print(f"""Deducting option book buy to close and roll:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    remaining_exp = lbe["expiration"] + lbe["day"] - d
                    new_expiration = lbe["expiration"] + self.dtes

                    if new_expiration < self.limit_dtes:
                        cash += self._roll_current_put_position(
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
                        cash += self._sell_to_open_put(
                            d, spot, spot_vix, nav,
                            current_leverage, self.delta,
                            self.dtes, self.take_profit)


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

            if daily_nav:
                return_path.append(nav)

        # In order to make this portfolio stitchable, we need to close
        # all live option positions at the end of the simulation.
        if len(self.live_options_book['sto']) == 1:
            print("Closing remaining live short options:")
            # TODO: Consider integrating all this logic in the private
            #       method _buy_to_close_put
            lbe = self.live_options_book['sto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = self.vix[-1]
            spx = self.spot[-1]
            r = self.risk_free_rate
            book_put_iv = self.svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(spot, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(spot, strike, yr_exp, r, book_put_iv, False)
            debit = self._buy_to_close_put(days - 1, strike, exp, book_put_delta,
                                           book_put_iv, book_put_price,
                                           lbe["size"], lbe["price"])
            if daily_nav:
                return_path[-1] += debit
            else:
                nav += debit

        return return_path if daily_nav else nav


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
    def __init__(self, *, spot_spx, spot_vix, vix3m, svi,
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
        super().__init__(spot_spx=spot_spx, spot_vix=spot_vix, vix3m=vix3m,
                         svi=svi, rf_rate=rf_rate, full_book=full_book)
        print(f"""Initializing Put Credit Spreads portfolio strategy:"
       Monthly Withdrawals: ${distribution:,.2f}
         Notional Leverage:  {leverage:.2f}x of NAV
  Option Spread Expiration:  {dtes:.0f} DTEs
        Short Option Delta:  {short_delta:.2f}
              Delta Spread:  {delta_spread:.2f}
           Take profits at:  {profit_target*100:.2f}% of premium sold
       Maximium Expiration:  {dtes_to_close:.0f} DTEs
     Delta Close Threshold:  {delta_threshold:.2f}
        """)
        self.monthly_dist = distribution
        self.leverage = leverage
        self.spread_dtes = dtes
        self.short_delta = short_delta
        self.delta_spread = delta_spread
        self.profit_target = profit_target
        self.dtes_to_close = dtes_to_close
        self.delta_threshold = delta_threshold


    def _buy_to_open_put(self,
                         day,    # Simulation day
                         spot,   # SPX Spot
                         atm_iv, # ATM Implied Volatility
                         delta,  # Put delta
                         dtes,   # Option expiration
                         size):  # Position size
        print(f"""Opening a new long put position with the following requirements:
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

        print(f"""Selling to close live options in book:
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
        print(f"""Selling put credit spread with the folloing parameters:
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
            print("Portfolio NAV too low to sell even a single spread at "
                  "the given notional leverage.  Skipping...")
            return 0.0

        position_size = self.live_options_book['sto'][0]["size"]
        long_delta = short_delta + delta_spread
        assert short_delta < long_delta < 0.0
        long_debit = self._buy_to_open_put(
            day, spot, atm_iv, long_delta, dtes, position_size)
        total_credit = short_credit + long_debit
        return total_credit


    def run_simulation(self, initial_nav: float, days: int, daily_nav=False):
        """Run portfolio simulation (see parent's class docstring)."""
        print(f"""Starting simulation, initial parameters:
            Initial NAV: ${initial_nav:,.2f}
            Days to run:  {days}
       Report daily NAV:  {daily_nav}""")

        assert days <= len(self.spot)

        # Technical Indicators
        ema20 = pd.Series(self.spot).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = self.leverage
        nav = initial_nav

        cash = self._sell_put_credit_spread(
            0, self.spot[0], math.sqrt(self.vix[0]), nav, current_leverage,
            self.short_delta, self.delta_spread, self.spread_dtes,
            self.profit_target)

        print(f"""First Put Credit Spread trade:
             Initial cash after trade: ${cash:,.2f}""")

        return_path = [nav]

        for d in range(1, days):
            spot = self.spot[d]
            spot_vix = math.sqrt(self.vix[d])
            vix3m = math.sqrt(self.vix3m[d])
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= self.monthly_dist
                print(f"""End of month.
                Subtracted distribution: ${self.monthly_dist:,.2f}
                           cash balance: ${cash:,.2f}
                """)

            if spot > ema20[d]:
                days_above_ema += 1
            else:
                days_above_ema = 0

            nav = nav * (1 + r/252.0)

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

            if state == 'cooldown':
                if days_above_ema >= 5 and not backwardation:
                    state = 'wade_in'
                    current_leverage = self.leverage / 2.0

                    # Deploy a given PCS with
                    # the specified DTEs.
                    prev_cash = cash
                    spread_credit = self._sell_put_credit_spread(
                        d, spot, spot_vix, nav, current_leverage,
                        self.short_delta, self.delta_spread,
                        self.spread_dtes, self.profit_target)
                    cash += spread_credit

                    print(f"""New put credit spread trade:
                        Spread credit: ${spread_credit:,.2f}
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

            if state in ['active', 'wade_in']:
                # Get the latest entry in the option ledger.
                if len(self.live_options_book['sto']) == 0:
                    # No live options available, sell a new spread
                    print("No live option trades in book, attempting to sell"
                          "new put credit spreads:")
                    cash += self._sell_put_credit_spread(
                        d, spot, spot_vix, nav, current_leverage, self.short_delta,
                        self.delta_spread, self.spread_dtes, self.profit_target)

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

                book_short_put_iv = self.svi.get_iv_curve(
                    spot_vix, short_put_strike, spot, spread_exp)
                book_short_put_price = bs.option_price(
                    spot, short_put_strike, spread_exp, r, book_short_put_iv, False)
                book_short_put_delta = bs.option_delta(
                    spot, short_put_strike, spread_exp, r, book_short_put_iv, False)

                # Get data for the long leg
                lbe_long = self.live_options_book['bto'][0]
                long_put_strike = lbe_long["strike"]
                long_put_orig_price = lbe_long["price"]
                long_put_size = lbe_long["size"]
                assert spread_target_profit is not None

                book_long_put_iv = self.svi.get_iv_curve(
                    spot_vix, long_put_strike, spot, spread_exp)
                book_long_put_price = bs.option_price(
                    spot, long_put_strike, spread_exp, r, book_long_put_iv, False)
                book_long_put_delta = bs.option_delta(
                    spot, long_put_strike, spread_exp, r, book_long_put_iv, False)

                spread_book_price = book_short_put_price - book_long_put_price
                spread_orig_price = short_put_orig_price - long_put_orig_price
                spread_target_profit_price = (1.0 - self.profit_target) * spread_orig_price

                # Operator's Decision Tree:
                # Option 1:
                #  If the VIX is in backwardation with VIX3M
                #  immediately close the options book and enter
                #  a cooldown period where no more premium is sold.
                if backwardation:
                    print("Entered a state of backwardation between"
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
                    print(f"""Short put leg delta threshold crossed:
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
                    print(f"""Credit spread target profit reached:
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
                    print("Credit spread reached terminal expiration "
                          f"of {spread_dtes} DTEs, closing.")
                    buy_to_close = True

                if buy_to_close:
                    print("Closing credit spread.")
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

                    print(f"""Deducting put credit spread closing cost:
                 Spread closing debit: ${spread_close_debit:,.2f}
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_put_credit_spread(
                            d, spot, spot_vix, nav, current_leverage, self.short_delta,
                            self.delta_spread, self.spread_dtes, self.profit_target)

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

            if daily_nav:
                return_path.append(nav)

        # In order to make this portfolio stitchable, we need to close
        # all live option positions at the end of the simulation.
        if len(self.live_options_book['sto']) == 1:
            assert len(self.live_options_book['bto']) == 1
            print("Closing remaining option positions:")

            # Short Leg
            # TODO: Consider integrating all this logic in the private
            #       method _buy_to_close_put
            lbe = self.live_options_book['sto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = self.vix[-1]
            spx = self.spot[-1]
            r = self.risk_free_rate
            book_put_iv = self.svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(spot, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(spot, strike, yr_exp, r, book_put_iv, False)
            short_close_debit = self._buy_to_close_put(
                days - 1, strike, exp, book_put_delta, book_put_iv,
                book_put_price, lbe["size"], lbe["price"])

            # Long leg
            lbe = self.live_options_book['bto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = self.vix[-1]
            spx = self.spot[-1]
            r = self.risk_free_rate
            book_put_iv = self.svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(spot, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(spot, strike, yr_exp, r, book_put_iv, False)
            long_close_credit = self._sell_to_close_put(
                days - 1, strike, exp, book_put_delta, book_put_iv,
                book_put_price, lbe["size"], lbe["price"])

            total_debit = long_close_credit - short_close_debit
            assert total_debit > 0.0

            if daily_nav:
                return_path[-1] += total_debit
            else:
                nav += total_debit

        return return_path if daily_nav else nav
