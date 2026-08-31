"""
short_put_model.py

Implements shorting naked SPX puts over a T-bill overlay
using option modelling for pricing proper volatility
surface pricing, trading mechanics

See the class documentation for the specific
dynamics of the trading strategy.
"""

import math
from abc import ABC, abstractmethod
from typing import override

import numpy as np
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


    def __init__(
            self, *,
            rf_rate=0.03,     # Annualized risk free rate.
            full_book=False): # Track full options book for debugging.
        super().__init__()
        self.log(f"""Initializing SPX put portfolio strategy:"
            Risk Free Rate:  {rf_rate*100.0:.2f}%
        Track Options Book:  {full_book}""")
        self.risk_free_rate = rf_rate
        self.live_options_book = {
            'sto': [],
            'bto': []
        }

        self.options_trade_book = None
        self.svi = None

        if full_book:
            self.options_trade_book = []


    def _find_put_strike_by_delta(self, spot: float, atm_iv: float,
                                  target_delta: float, expiration: float):
        return self._find_put_strike(
            spot, atm_iv, target_delta, expiration, use_delta=True)


    def _find_put_strike(self, spot: float, atm_iv: float,
                         target: float, expiration: float, use_delta: bool):
        # Binary search to find the exact put strike that yields the target delta
        self.log(f"""Searching put option with parameters:
        Underlying price:  {spot:,.2f}
                  ATM IV:  {atm_iv * 100:.2f}""")
        if use_delta:
            self.log(f" Target Delta:  {target:.2f}")
        else:
            self.log(f" Target Price: ${target:,.2f}")
        self.log(f" DTEs:  {expiration*365.0:.0f}")

        low = spot * 0.5
        high = spot
        r = self.risk_free_rate
        for _ in range(40):
            mid = (low + high) / 2.0
            iv = self.svi.get_iv_curve(atm_iv, mid, spot, expiration)
            delta = bs.option_delta(spot, mid, expiration, r, iv, is_call=False)
            price = bs.option_price(spot, mid, expiration, r, iv, is_call=False)

            self.log(f"""
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
        self.log(f"""Trying to sell put with the following requirements:
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
            self.log("Option notional too large to sell "
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

        self.log(f"""Recording option trade in book:
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
            self.log(f"                Target profit: ${target_profit:,.2f}")
        if pnl:
            self.log(f"                          PnL: ${pnl:,.2f}")

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

        self.log(f"""Buying to close live options in book:
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
    def run_simulation(
        self, *,
        spot_spx: list[float],  # Time series for SPX underlying price.
        spot_vix: list[float],  # Time series for the spot VIX.
        vix3m: list[float],     # Time series for the VIX3M.
        svi: DynamicSVI,        # Stochastic Volatility Inspired IV Model.
        initial_nav: float,     # NAV to start the simulation with.
        days: int) -> np.array: # Days to run the simulation
        """Run portfolio simulation (see parent's class docstring)."""
        self.svi = svi
        return super().run_simulation(
           spot_spx=spot_spx, spot_vix=spot_vix, vix3m=vix3m,
           svi=svi, initial_nav=initial_nav, days=days)


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
                 distribution: float,   # Monthly withdrawals
                 leverage=0.5,          # Short put notional position size
                                        # based on a percentage NAV.
                 rf_rate=0.03,          # Annualized risk free rate.
                 inflation=0.025,       # Annualized inflation rate.
                 delta=-0.15,           # Put Delta to use when shorting.
                 dtes=45,               # Short option expirations.
                 max_dtes=135,          # Option book maximium expiration permitted.
                 take_profit=0.75,      # Take profits at percentage of each premium sold.
                 full_book=False):       # Track full options book for debugging.
        super().__init__(
            rf_rate=rf_rate,
            full_book=full_book)

        self.log(f"""Initializing short put portfolio strategy:"
       Monthly Withdrawals: ${distribution:,.2f}
         Notional Leverage:  {leverage:.2f}x of NAV
            Inflation Rate:  {inflation*100.0:.2f}%
              Option Delta:  {delta:.2f}
         Option Expiration:  {dtes:.0f} DTEs
       Maximium Expiration:  {max_dtes:.0f} DTEs
           Take profits at:  {take_profit*100:.2f}% of premium sold""")
        self.monthly_dist = distribution
        self.leverage = leverage
        self.inflation = inflation
        self.delta = delta
        self.dtes = dtes
        self.limit_dtes = max_dtes
        self.take_profit = take_profit


    @classmethod
    @override
    def from_json_object(cls, o):
        """
        Builds an instance of this portfolio strategy from a JSON parsed
        object. The structure must have the following shape:
        {
            "type": "SPXPutOptionStrategy",
            "rf_rate": risk_free_rate
            "distribution": monthly_withdrawal,
            "leverage": notional_leverage,
            "inflation": inflation_rate,
            "delta": short_put_delta,
            "dtes": days_to_expiration,
            "max_dtes": max_expiration_to_hold,
            "take_profit": % to take profit [0, 1] float.
        }
        """
        if o["type"] != "SPXPutOptionStrategy":
            return None
        rf_rate = float(o["rf_rate"])
        distribution = float(o["distribution"])
        leverage = float(o["leverage"])
        inflation = float(o["inflation"])
        delta = float(o["delta"])
        dtes = float(o["dtes"])
        max_dtes = float(o["max_dtes"])
        take_profit = float(o["take_profit"])
        return ShortSPXPutStrategy(
            rf_rate=rf_rate,
            distribution=distribution,
            leverage=leverage,
            inflation=inflation,
            delta=delta,
            dtes=dtes,
            max_dtes=max_dtes,
            take_profit=take_profit)


    def _find_put_strike_by_price(self, spot: float, atm_iv: float,
                                  target_price: float, expiration: float):
        return self._find_put_strike(
            spot, atm_iv, target_price, expiration, use_delta=False)


    def _roll_current_put_position(self, cur_day: int, spot: float, atm_iv: float,
                                   position_size: int, orig_price: float,
                                   new_expiration: float, target_profit_price: float):
        self.log(f"""Trying to roll put position with the following requirements:
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

        cash = self._sell_to_open_put(
            0, spot_spx[0], math.sqrt(spot_vix[0]),
            nav, current_leverage, self.delta,
            self.dtes, self.take_profit)

        self.log(f"""First short put trade:
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
                        d, today_spot, today_vix, nav,
                        current_leverage, self.delta, self.dtes,
                        self.take_profit)
                    self.log(f"""New short put trade:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

            if state in ['active', 'wade_in']:
                # If there are no live option positions, open one now.
                if len(self.live_options_book['sto']) == 0:
                    self.log("No current live options in the book, selling new ones")
                    cash += self._sell_to_open_put(
                        d, today_spot, today_vix, nav,
                        current_leverage, self.delta,
                        self.dtes, self.take_profit)

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

                # Get the latest entry in the option ledger.
                lbe = self.live_options_book['sto'][0]
                put_strike = lbe["strike"]
                put_orig_price = lbe["price"]
                put_size = lbe["size"]
                put_dtes = lbe["expiration"] + lbe["day"] - d
                put_target_profit = lbe["target_profit"]
                assert put_target_profit

                put_exp = max(put_dtes,1)/365.0

                book_put_iv = svi.get_iv_curve(
                    today_vix, put_strike, today_spot, put_exp)
                book_put_price = bs.option_price(
                    today_spot, put_strike, put_exp, r, book_put_iv, False)
                book_put_delta = bs.option_delta(
                    today_spot, put_strike, put_exp, r, book_put_iv, False)

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
                    self.log("Trying to close option (not roll)")
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"],
                        book_put_delta, book_put_iv, book_put_price,
                        put_size, put_orig_price)
                    prev_cash = cash
                    cash += book_debit
                    self.log(f"""Deducting option book buy to close:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_to_open_put(
                            d, today_spot, today_vix, nav,
                            current_leverage, self.delta,
                            self.dtes, self.take_profit)

                if roll_option:
                    assert not buy_to_close
                    self.log("Trying to roll current options position")
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"], book_put_delta,
                        book_put_iv, book_put_price, put_size, put_orig_price)

                    prev_cash = cash
                    cash += book_debit

                    self.log(f"""Deducting option book buy to close and roll:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    remaining_exp = lbe["expiration"] + lbe["day"] - d
                    new_expiration = lbe["expiration"] + self.dtes

                    if new_expiration < self.limit_dtes:
                        cash += self._roll_current_put_position(
                            d, today_spot, today_vix, put_size,
                            book_put_price, new_expiration,
                            put_orig_price * (1.0 - self.take_profit))
                        self.log(f"""
                     Rolled option remaining: {remaining_exp:.0f} DTEs
                    New put expiration to be: {new_expiration:.0f} DTEs
                            Expiration limit: {self.limit_dtes:.0f} DTEs""")
                    else:
                        self.log(f"""
                        Current options position exceeded the expiration
                        limit permitted of {self.limit_dtes:.0f} DTEs.
                        Selling new option instead.""")
                        cash += self._sell_to_open_put(
                            d, today_spot, today_vix, nav,
                            current_leverage, self.delta,
                            self.dtes, self.take_profit)


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
            self.log("Closing remaining live short options:")
            # TODO: Consider integrating all this logic in the private
            #       method _buy_to_close_put
            lbe = self.live_options_book['sto'][0]
            strike = lbe["strike"]
            exp = lbe["expiration"] + lbe["day"] + 1 - days
            yr_exp = exp/365.0
            vix = self.vix[-1]
            spx = self.spot[-1]
            r = self.risk_free_rate
            book_put_iv = svi.get_iv_curve(vix, strike, spx, yr_exp)
            book_put_price = bs.option_price(today_spot, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(today_spot, strike, yr_exp, r, book_put_iv, False)
            debit = self._buy_to_close_put(days - 1, strike, exp, book_put_delta,
                                           book_put_iv, book_put_price,
                                           lbe["size"], lbe["price"])
            return_path[-1] += debit

        return return_path
