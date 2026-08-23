"""
portfolio_models.py

Contains a set of different models that represent well known
investment strategies:

1. Fixed Income.
2. Long SP500.
3. Short SPX options against a T-Bills overlay
4. A single strategy built out of a linear combination
   of the ones above.
"""

import math
from abc import ABC, abstractmethod

import pandas as pd

from market_modelling import black_scholes as bs
from market_modelling.dsvi import DynamicSVI

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
        position_size = round(nav * leverage / put_strike / 100)
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
        print(f"""Buying to close live options in book:
            Simulation day:  {cur_day}
                Put strike: ${put_strike:,.2f}
        Initial Expiration:  {dtes:.0f} DTEs
         Current put delta:  {put_delta:.2f}
            Current put IV:  {put_iv:.2f}
         Current put price: ${put_price:,.2f}
                     Units:  {position_size}""")
        pnl = (orig_price - put_price) * position_size * 100
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
       of contango.
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
