"""
short_put_model.py

Implements shorting naked SPX puts over a T-bill overlay
using option modelling for pricing proper volatility
surface pricing, trading mechanics

See the class documentation for the specific
dynamics of the trading strategy.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd

from market_modelling.dsvi import DynamicSVI
from market_modelling import black_scholes as bs

from portfolio_models.linear_models import InvestmentStrategy

logger = logging.getLogger(__name__)

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


    def __init__(self, *,
                 rf_rate=0.03): # Annualized risk free rate.
        super().__init__()

        logging.debug("Initializing SPX put portfolio strategy:")
        logging.debug("Risk Free Rate: %2f", rf_rate*100.0)

        self.svi = None
        self.risk_free_rate = rf_rate
        self.live_options_book = {
            'sto': [],
            'bto': []
        }
        self.track_book = False


    def _find_put_strike_by_delta(self, spot: float, atm_iv: float,
                                  target_delta: float, expiration: float):
        return self._find_put_strike(
            spot, atm_iv, target_delta, expiration, use_delta=True)


    def _find_put_strike(self, spot: float, atm_iv: float,
                         target: float, expiration: float, use_delta: bool):
        # Binary search to find the exact put strike that yields the target delta
        logging.debug("Searching put option with parameters:")
        logging.debug("Underlying price:  %.2f", spot)
        logging.debug("ATM IV:  %.2f%%", atm_iv)

        if use_delta:
            logging.debug("Target Delta:  %.2f", target)
        else:
            logging.debug("Target Price: $%.2f", target)
        logging.debug("DTEs: %.0f", expiration*365.0)

        low = spot * 0.5
        high = spot
        r = self.risk_free_rate
        for _ in range(40):
            mid = (low + high) / 2.0
            iv = self.svi.get_iv_curve(atm_iv, mid, spot, expiration)
            delta = bs.option_delta(spot, mid, expiration, r, iv, is_call=False)
            price = bs.option_price(spot, mid, expiration, r, iv, is_call=False)

            logging.debug("Trying strike: $%.2f", mid)
            logging.debug("IV estimated at:  %.2f%%", iv * 100.0)
            logging.debug("Delta at:  %.2f", delta)
            logging.debug("Price at: $%.2f", price)

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
        logging.debug("Trying to sell put with the following requirements:")
        logging.debug("Simulation Day:  %d", cur_day)
        logging.debug("Spot: $%.2f", spot)
        logging.debug("ATM IV:  %.2f%%", atm_iv)
        logging.debug("NAV: $%.2f", nav)
        logging.debug("Leverage:  %.2fx", leverage)
        logging.debug("Delta:  %.2f", delta)
        logging.debug("Expiration:  %.0f", expiration)
        logging.debug("Target Profit:  %.2f%%", target_profit_pct)

        yearly_exp = expiration / 365.0
        put_strike, put_price, _, put_iv = self._find_put_strike_by_delta(
            spot, atm_iv, delta, yearly_exp)
        position_size = math.floor(nav * leverage / put_strike / 100)
        # NAV is too low to create even a single contract, bail early.
        if position_size == 0:
            logging.debug("Option notional too large to sell "
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

        logging.debug("Recording option trade in book:")
        logging.debug("Simulation Day:  %d", day)
        logging.debug("Trade:  %s", trade_str)
        logging.debug("Strike: $%.2f", strike)
        logging.debug("Expiration:  %.0f DTEs", expiration)
        logging.debug("Delta:  %.2f", delta)
        logging.debug("IV:  %.2f%%", iv)
        logging.debug("Price: $%.2f", price)
        logging.debug("Position size:  %d", size)
        logging.debug("Total debit/credit: $%.2f", debit_credit)

        if target_profit:
            logging.debug("Target profit: $%.2f", target_profit)
        if pnl:
            logging.debug("PnL: $%.2f", pnl)

        # Trade day, Put strike, Initial Expiration, Delta, IV, Price, Size, Total Premium
        book_entry =  {
            "day": day,
            "trade": trade_str,
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

        if self.track_book:
            self.book.append(book_entry)

        return debit_credit


    def _buy_to_close_put(self, cur_day: int, put_strike: float, dtes: float,
                          put_delta: float, put_iv: float, put_price: float,
                          position_size: int, orig_price: float):
        pnl = (orig_price - put_price) * position_size * 100

        logging.debug("Buying to close live options in book:")
        logging.debug("Simulation day:  %d", cur_day)
        logging.debug("Put strike: $%.2f", put_strike)
        logging.debug("Initial Expiration:  %.0f DTEs", dtes)
        logging.debug("Current put delta:  %.2f", put_delta)
        logging.debug("Current put IV:  %.2f", put_iv)
        logging.debug("Current put price: $%.2f", put_price)
        logging.debug("Units:  %d", position_size)
        logging.debug("Original price: $%.2f", orig_price)
        logging.debug("Position PnL: $%.2f", pnl)

        return self._write_put_trade_to_book(
            day=cur_day, trade='btc', strike=put_strike,
            expiration=dtes, delta=put_delta, iv=put_iv, price=put_price,
            size=position_size, pnl=pnl)


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
        """Run portfolio simulation (see parent's class docstring)."""
        return super().run_simulation(
           spot_spx=spot_spx, spot_vix=spot_vix, vix3m=vix3m,
           svi=svi, initial_nav=initial_nav, days=days, full_book=full_book)


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
                 take_profit=0.75):     # Take profits at percentage of each premium sold.
        super().__init__(rf_rate=rf_rate)

        logging.debug("Initializing short put portfolio strategy:")
        logging.debug("Monthly Withdrawals: $%.2f", distribution)
        logging.debug("Notional Leverage:  %.2fx of NAV", leverage)
        logging.debug("Inflation Rate:  %.2f%%", rf_rate)
        logging.debug("Option Delta:  %.2f", delta)
        logging.debug("Option Expiration:  %.0f DTEs", dtes)
        logging.debug("Maximium Expiration:  %.0f DTEs", max_dtes)
        logging.debug("Take profits at:  %.2f%% of premium sold", take_profit)

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
        delta = float(o["delta"])
        distribution = float(o["distribution"])
        dtes = float(o["dtes"])
        inflation = float(o["inflation"])
        leverage = float(o["leverage"])
        max_dtes = float(o["max_dtes"])
        rf_rate = float(o["rf_rate"])
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
        logging.debug("Trying to roll put position with the following requirements:")
        logging.debug("Simulation day:  %d", cur_day)
        logging.debug("Spot: $%.2f", spot)
        logging.debug("ATM IV:  %.2f%%", atm_iv)
        logging.debug("Original Position Size:  %d", position_size)
        logging.debug("Put original price: $%.2f", orig_price)
        logging.debug("New Expiration:  %.0f DTEs", new_expiration)
        logging.debug("Target Profit: $%.2f", target_profit_price)

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
        spot_spx: list[float],        # Time series for SPX underlying price.
        spot_vix: list[float],        # Time series for the spot VIX.
        vix3m: list[float],           # Time series for the VIX3M.
        svi: DynamicSVI,              # Stochastic Volatility Inspired IV Model.
        initial_nav: float,           # NAV to start the simulation with.
        days: int,                    # Days to run the simulation
        full_book=False) -> np.array: # Track full options book for debugging.
        """Run portfolio simulation (see parent's class docstring)."""
        logging.debug("Starting simulation, initial parameters:")
        logging.debug("Initial NAV: $%.2f", initial_nav)
        logging.debug("Days to run:  %d", days)

        self.track_book = full_book
        self.svi = svi
        assert days <= len(spot_spx)

        # Technical Indicators
        ema20 = pd.Series(spot_spx).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = self.leverage
        nav = initial_nav

        cash = self._sell_to_open_put(
            0, spot_spx[0], spot_vix[0],
            nav, current_leverage, self.delta,
            self.dtes, self.take_profit)

        logging.debug("First short put trade:")
        logging.debug("Initial cash after trade: $%.2f", cash)

        return_path = np.zeros(days)
        return_path[0] = nav

        for d in range(1, days):
            today_spot = spot_spx[d]
            today_vix = spot_vix[d]
            today_vix3m = vix3m[d]
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= self.monthly_dist
                logging.debug("End of month.")
                logging.debug("Subtracted distribution: $%.2f", self.monthly_dist)
                logging.debug("cash balance: $%.2f", cash)
                if full_book:
                    self.book.append({
                        "day": d,
                        "trade": "Withdrawal",
                        "price": self.monthly_dist
                    })

            if today_spot > ema20[d]:
                days_above_ema += 1
            else:
                days_above_ema = 0

            nav = nav * (1 + r/252.0)

            # VIX Term Structure Check
            backwardation = today_vix > today_vix3m

            logging.debug("Simulation at day:  %d", d)
            logging.debug("SPX Spot: $%.2f", today_spot)
            logging.debug("VIX:  %.2f%%", today_vix)
            logging.debug("VIX 3M:  %.2f%%", today_vix3m)
            logging.debug("VIX in backwardation?: %s", str(backwardation))
            logging.debug("SPX EMA-20: $%.2f", ema20[d])
            logging.debug("Cash: $%.2f", cash)
            logging.debug("NAV: $%.2f", nav)
            logging.debug("Options state: %s", state)

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
                    logging.debug("New short put trade:")
                    logging.debug("Cash before trade: $%.2f", prev_cash)
                    logging.debug("Cash after trade: $%.2f", cash)

            if state in ['active', 'wade_in']:
                # If there are no live option positions, open one now.
                if len(self.live_options_book['sto']) == 0:
                    logging.debug("No current live options in the book, selling new ones")
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
                        logging.debug("Reinvesting / withdrawing cash after options trade:")
                        logging.debug("Previous NAV: $%.2f", prev_nav)
                        logging.debug("After-trade NAV: $%.2f", nav)
                        logging.debug("Previous Cash: $%.2f", prev_cash)

                    return_path[d] = nav
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
                    logging.debug("Trying to close option (not roll)")
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"],
                        book_put_delta, book_put_iv, book_put_price,
                        put_size, put_orig_price)
                    prev_cash = cash
                    cash += book_debit
                    logging.debug("Deducting option book buy to close:")
                    logging.debug("Cash before trade: $%.2f", prev_cash)
                    logging.debug("Cash after trade: $%.2f", cash)
                    if state != 'cooldown':
                        premium = self._sell_to_open_put(
                            d, today_spot, today_vix, nav,
                            current_leverage, self.delta,
                            self.dtes, self.take_profit)
                        cash += premium

                if roll_option:
                    assert not buy_to_close
                    logging.debug("Trying to roll current options position")
                    book_debit = self._buy_to_close_put(
                        d, put_strike, lbe["expiration"], book_put_delta,
                        book_put_iv, book_put_price, put_size, put_orig_price)

                    prev_cash = cash
                    cash += book_debit

                    logging.debug("Deducting option book buy to close and roll:")
                    logging.debug("Cash before trade: $%.2f", prev_cash)
                    logging.debug("Cash after trade: $%.2f", cash)

                    remaining_exp = lbe["expiration"] + lbe["day"] - d
                    new_expiration = lbe["expiration"] + self.dtes

                    if new_expiration < self.limit_dtes:
                        cash += self._roll_current_put_position(
                            d, today_spot, today_vix, put_size,
                            book_put_price, new_expiration,
                            put_orig_price * (1.0 - self.take_profit))
                        logging.debug("Rolled option remaining: %.0f DTEs", remaining_exp)
                        logging.debug("New put expiration to be: %.0f DTEs", new_expiration)
                        logging.debug("Expiration limit: %.0f DTEs", self.limit_dtes)
                    else:
                        logging.debug("Current options position exceeded the expiration")
                        logging.debug("limit permitted of %0f DTEs.", self.limit_dtes)
                        logging.debug("Selling new option instead.")
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
                logging.debug("Reinvesting / withdrawing cash after options trade:")
                logging.debug("Previous NAV: $%.2f", prev_nav)
                logging.debug("After-trade NAV: $%.2f", nav)
                logging.debug("Previous Cash: $%.2f", prev_cash)

            return_path[d] = nav

        # In order to make this portfolio stitchable, we need to close
        # all live option positions at the end of the simulation.
        if len(self.live_options_book['sto']) == 1:
            logging.debug("Closing remaining live short options:")
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
            book_put_price = bs.option_price(today_spot, strike, yr_exp, r, book_put_iv, False)
            book_put_delta = bs.option_delta(today_spot, strike, yr_exp, r, book_put_iv, False)
            debit = self._buy_to_close_put(days - 1, strike, exp, book_put_delta,
                                           book_put_iv, book_put_price,
                                           lbe["size"], lbe["price"])
            return_path[-1] += debit

        return return_path
