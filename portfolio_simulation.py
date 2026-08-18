import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize

# ==========================================
# 1. BLACK-SCHOLES GREEKS & PRICING
# ==========================================
def norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, r, sigma, is_call=False):
    print(f"""
    Pricing option:
    Spot: ${S:,.2f}
    Strike: ${K:,.2f}
    Expiration: {T * 365.0} DTEs
    Risk Free Rate: {r * 100:.2f}%
    Implied Volatility: {sigma * 100:.2f}%
    Is Call?: {is_call}""")
    
    T = max(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call: return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else: return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S, K, T, r, sigma, is_call=False):
    T = max(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call: return norm_cdf(d1)
    else: return norm_cdf(d1) - 1.0

class DynamicSVI:
    def __init__(self, strikes_market, iv_market, spot_price, T):
        """
        Initializes Dynamic Surface Volatility Interpolation
        with data extracted from a real observed market options chain.
        strikes_market: list of observed strike prices from real market data
        iv_market: corresponding implied volatility for the above strikes
        spot_price: current underlying spot price
        T: option maturity in years
        """
        self.T = T
        
        k_market = np.log(strikes_market / spot_price)
        self.a0, self.b, self.rho, self.m, self.sigma = self._fit_svi(k_market, iv_market, T)
        
        # Precompute the shape constant C (contribution of shape to ATM variance)
        self.shape_constant = self.b * (-self.rho * self.m + np.sqrt(self.m**2 + self.sigma**2))

    
    def _fit_svi(self, k_market, iv_market, T, initial_guess=None):
        w_market = (iv_market ** 2) * T
        
        def svi_total_variance(params, k):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def objective(params):
            a, b, rho, m, sigma = params
            if b < 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e6
            w_fit = svi_total_variance(params, k_market)
            return np.sum((w_fit - w_market) ** 2)
    
        if initial_guess is None:
            initial_guess = [w_market[len(w_market)//2], 0.1, -0.4, 0.0, 0.1]
            
        bounds = [
            (-np.inf, np.inf), 
            (0.0, np.inf),     
            (-0.999, 0.999),   
            (-np.inf, np.inf), 
            (1e-4, np.inf)     
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        return result.x

    def get_iv_curve(self, simulated_atm_iv, strikes, forward, current_T):
        """
        Extrapolates the full OTM IV curve given a simulated ATM IV and the 
        current (potentially shrunk) time-to-expiry current_T.
        """
        # 1. Convert simulated ATM IV to total variance for the current T
        w_atm_target = (simulated_atm_iv ** 2) * current_T
        
        # 2. Dynamically adjust 'a' to match the simulated ATM variance
        a_t = w_atm_target - self.shape_constant
        
        # 3. Compute log-moneyness for target strikes
        k = np.log(strikes / forward)
        
        # 4. Evaluate Raw SVI total variance
        w_t = a_t + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2))
        
        # 5. Convert back to implied volatility using the current T
        iv_curve = np.sqrt(np.maximum(w_t, 1e-8) / current_T)
        return iv_curve

# ==========================================
# 3. Stochastic Volatility with Correlated Jumps
#    Time Series Simulation
# ==========================================
class SVCJSimulation:
    """
    Stochastic Volatility with Correlated Jumps (SVCJ)
    Instituitonal parameters for S&P 500.
    """
    def __init__(self, 
                 mu = 0.08,      # Equity drift 
                 kappa = 4.5,    # VIX mean reversion speed
                 theta = 0.04,   # Long-term variance
                 sigma_v = 0.4,  # Volatility of volatility
                 rho = -0.65,    # Price/vol correlation (leverage effect)
                 lambda_j = 1.5, # Expected jumps per year
                 mu_j = -0.05,   # Mean price jump size
                 sigma_j = 0.06, # Volatility of price jump
                 mu_v = 0.08):   # Mean variance jump size
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.mu_v = mu_v
        
    def generate_path(self, S0, V0, days=252):
        V0 = V0 * V0
        dt = 1.0 / 252.0
        S = np.zeros(days); S[0] = S0
        V = np.zeros(days); V[0] = V0
        
        for t in range(1, days):
            Z1 = np.random.standard_normal()
            Z2 = self.rho * Z1 + math.sqrt(1 - self.rho**2) * np.random.standard_normal()
            
            # Poisson Jump
            N = np.random.poisson(self.lambda_j * dt)
            J_S = 0; J_V = 0
            if N > 0:
                Z3 = np.random.standard_normal()
                J_S = self.mu_j + self.sigma_j * Z3
                # Variance jumps are positive and exponentially distributed
                J_V = np.random.exponential(self.mu_v)
                
            # Variance process (Euler-Maruyama, ensuring V > 0)
            v_prev = max(V[t-1], 1e-6)
            V[t] = v_prev + self.kappa * (self.theta - v_prev) * dt + self.sigma_v * math.sqrt(v_prev * dt) * Z2 + J_V
            V[t] = max(V[t], 7.225e-3)
            
            # Price process
            S[t] = S[t-1] * np.exp((self.mu - 0.5 * v_prev) * dt + math.sqrt(v_prev * dt) * Z1 + J_S)
            
        return S, V

    def derive_vix3m(self, vix_path):
        """
        Derives VIX3M path directly from a pre-calculated VIX path using affine mapping.
        """
        # Helper functions for affine coefficients A(tau) and B(tau)
        def get_A_B(tau):
            B = (1.0 - np.exp(-self.kappa * tau)) / (self.kappa * tau)
            A = self.theta * (1.0 - B)
            return A, B
    
        A_30, B_30 = get_A_B(30/365.0)
        A_90, B_90 = get_A_B(90/365.0)
        
        # Compute linear mapping coefficients alpha and beta
        beta = B_90 / B_30
        alpha = A_90 - beta * A_30
        
        # Map VIX^2 to VIX3M^2 via linear transformation, then take the square root
        vix3m_path = np.sqrt(alpha + beta * (vix_path ** 2))
        
        return vix3m_path        

# ==========================================
# 4. THE TRADING ENGINE (Operator's Manual)
# ==========================================
class ShortSPXPutStrategy:
    def __init__(self,
                 spot_spx,        # Time series for SPX underlying price.
                 spot_vix,        # Time series for the spot VIX.
                 vix3m,           # Time series for the VIX3M.
                 svi,             # Stochastic Volatility Inspired IV Model.
                 rf_rate=0.03,    # Annualized risk free rate.
                 inflation=0.025, # Annualized inflation rate.
                 full_book=True): # Track full options book for debugging.
        print(f"""Initializing short put portfolio strategy:"
          Initial SPX Spot: {spot_spx[0]:,.2f}
               Initial VIX: {math.sqrt(spot_vix[0]):.2f}%
             Initial VIX3M: {math.sqrt(vix3m[0]):.2f}%
            Risk Free Rate: {rf_rate*100.0:.2f}%
            Inflation Rate: {inflation*100.0:.2f}%
        Track Options Book: {full_book}
        """)
        self.spot = spot_spx
        self.vix = spot_vix
        self.vix3m = vix3m
        self.svi = svi
        self.risk_free_rate = rf_rate
        self.inflation = inflation
        self.full_book = full_book
        self.book = []

    def _find_put_strike(self, spot, atm_iv, target_delta, expiration):
        # Binary search to find the exact put strike that yields the target delta
        print(f"""
        Searching put option with parameters:
        Underlying price: {spot:,.2f}
                  ATM IV: {atm_iv * 100:.2f}
            Target Delta: {target_delta:.2f}
                    DTEs: {expiration*365:.0f}
        """)
        low = spot * 0.7
        high = spot
        r = self.risk_free_rate
        for _ in range(40):
            mid = (low + high) / 2.0
            iv = self.svi.get_iv_curve(atm_iv, mid, spot, expiration)
            delta = bs_delta(spot, mid, expiration, r, iv, is_call=False)
            print(f"""Trying strike: ${mid:.2f} 
            IV estimated at: {iv*100:.2f}%
            Delta at: {delta:.2f}
            """)
            abs_delta = abs(delta)
            abs_target_delta = abs(target_delta)
            if abs(abs_target_delta - abs_delta) < 0.005:
                break
            if abs_delta > abs_target_delta: high = mid
            else: low = mid
        round_strike = int((low + high) / 20)
        return round_strike * 10

    def _sell_put(self, cur_day, spot, atm_iv, nav, leverage, delta, expiration):
        print(f"""Trying to sell put with the following requirements:
         Simulation Day:  {cur_day}
                   Spot: ${spot:,.2f}
                 ATM IV:  {atm_iv:.2f}
                    NAV: ${nav:,.2f}
               Leverage:  {leverage}x
                  Delta:  {delta}
             Expiration:  {expiration} DTEs
        """)
        r = self.risk_free_rate
        yearly_exp = expiration / 365.0
        put_strike = self._find_put_strike(spot, atm_iv, delta, yearly_exp)
        put_iv = self.svi.get_iv_curve(atm_iv, put_strike, spot, yearly_exp)
        put_price = bs_price(spot, put_strike, yearly_exp, r, put_iv, False)
        position_size = round(nav * leverage / put_strike / 100)
        premium = position_size * put_price * 100
        print(f"""Deploying put option:
               Strike:  {put_strike:,.2f}
                Price:  {put_price:,.2f}
        Position size:  {position_size}
        Total premium: ${premium:,.2f}
        """)
        # Trade day, Put strike, Initial Expiration, Delta, IV, Price, Size, Total Premium
        self.book.append(
            [cur_day, put_strike, expiration, delta,
             put_iv, put_price, position_size, premium])
        return premium

    def _rebuy_put(self, cur_day, put_strike, dtes,
                   put_delta, put_iv, put_price,
                   position_size, total_cost):
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
                          put_iv, put_price, position_size, total_cost])

    def run_simulation(self,
                       nav,                         # Initial portfolio NAV
                       monthly_distribution,        # Monthly withdrawals needed to retirement living      
                       notional_leverage = 0.5):    # Position size for options underwriting
        print(f"""Initiating path simulation with the following parameters:
               Initial NAV: ${nav:,.2f}
        Monthly withdrawal: ${monthly_distribution:,.2f}
         Notional leverage:  {notional_leverage}x
        """)

        # Technical Indicators
        ema20 = pd.Series(self.spot).ewm(span=20, adjust=False).mean().values

        state = 'active'
        days_above_ema = 0
        current_leverage = notional_leverage
        days = len(self.spot)

        cash = self._sell_put(
            0, self.spot[0], math.sqrt(self.vix[0]), nav,
            current_leverage, -0.15, 45)
        print(f"""First short put trade:
             Initial cash after trade: ${cash:,.2f}""")

        for d in range(1, days):
            spot = self.spot[d]
            spot_vix = math.sqrt(self.vix[d])
            vix3m = math.sqrt(self.vix3m[d])
            dt = 365.0 / 252.0
            r = self.risk_free_rate

            if d % 21 == 0:
                cash -= monthly_distribution
                print(f"""End of month.
                Subtracted distribution: ${monthly_distribution:,.2f}
                           cash balance: ${cash:,.2f}
                """)
            
            if spot > ema20[d]: days_above_ema += 1
            else: days_above_ema = 0

            cur_bond_nav = nav * np.exp(r * (d/252.0))
            
            # VIX Term Structure Check
            backwardation = spot_vix > vix3m * 1.05 # Avoid noise
    
            print(f"""Simulation at day {d}
                         SPX Spot: ${spot:,.2f}
                              VIX:  {spot_vix*100:.2f}%
                           VIX 3M:  {vix3m*100:.2f}%
            VIX in backwardation?:  {backwardation}
                             Cash: ${cash:,.2f}
                        Bonds NAV: ${cur_bond_nav:,.2f}
                    Options state:  {state}
                  """)
            
            # TRIPLE-LOCK COOLDOWN RE-ENTRY
            if state == 'cooldown':
                if days_above_ema >= 5 and not backwardation:
                    state = 'wade_in'
                    current_leverage = notional_leverage / 2.0 
                    # Deploy a Delta -0.15 Put with 45 DTEs.
                    prev_cash = cash
                    cash += self._sell_put(
                        d, spot, spot_vix, cur_bond_nav,
                        current_leverage, -0.15, 45)
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
                put_exp = max(put_dtes,1)/365.0

                book_put_iv = self.svi.get_iv_curve(spot_vix, put_strike, spot, put_exp)
                book_put_price = bs_price(spot, put_strike, put_exp, r, book_put_iv, False)
                book_put_delta = bs_delta(spot, put_strike, put_exp, r, book_put_iv, False)
                
                buy_to_close = False
                # OPERATOR'S DECISION TREE
                if backwardation:
                    buy_to_close = True; state = 'cooldown'; days_above_ema = 0 # HARD EJECT
                elif book_put_delta <= -0.50:
                    buy_to_close = True; state = 'cooldown'; days_above_ema = 0 # PRICE EJECT
                elif book_put_delta <= -0.35:
                    buy_to_close = True # GRACEFUL DEFENSE
                    # TODO: Roll logic is incomplete, need to finish
                    #       closing the book puts and selling new ones
                    #       45 DTEs beyond the original option expiration.
                elif book_put_price <= 0.5 * put_orig_price:
                    buy_to_close = True # TAKE PROFIT
                    if state == 'wade_in': current_leverage = notional_leverage; state = 'active'
                elif put_dtes <= 7:
                    buy_to_close = True # 7-DTE HARD DECK (Gamma Risk Eject)
                    
                if buy_to_close:
                    book_close_cost = book_put_price * put_size * 100;
                    self._rebuy_put(d, put_strike, last_book_entry[2],
                                    book_put_delta, book_put_iv, book_put_price,
                                    put_size, -book_close_cost)
                    prev_cash = cash
                    cash -= book_close_cost
                    print(f"""Deducting option book buy to close:
                    Cash before trade: ${prev_cash:,.2f}
                     Cash after trade: ${cash:,.2f}""")

                    if state != 'cooldown':
                        cash += self._sell_put(d, spot, spot_vix, nav,
                                               current_leverage, -0.15, 45)

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


# --- Real options chain data from Aug 15th 2026
spx_chain_data = """
6900	.2283
7200	.1907
7375	.1698
7475	.1586
7550	.1506
7625	.1433
7700	.1367
7750	.1328
7825	.1280
7875	.1256
7925	.1238
8000	.1213
8050	.1210
8100	.1214
8300	.1330
"""
spx_chain_dtes = 44.0
spot_spx = 7786.00
spot_vix = 0.1425
strikes = []
ivs = []

for row in spx_chain_data.strip().split('\n'):
    strike, iv = row.split('\t')
    strikes.append(float(strike))
    ivs.append(float(iv))

svi = DynamicSVI(np.array(strikes), np.array(ivs), spot_spx, spx_chain_dtes / 365)
svcj = SVCJSimulation()
spx, vix = svcj.generate_path(spot_spx, spot_vix)
vix3m = svcj.derive_vix3m(vix)
trade_strategy = ShortSPXPutStrategy(spx, vix, vix3m, svi)
trade_strategy.run_simulation(nav=8000000, monthly_distribution=12500, notional_leverage=0.5)

''' 
Further implementation roadmap:

1. Finish the rolling trade logic to properly simulate the delta 0.35
   roll at a credit scenario.
2. Spot check the IV of further expiration options and confirm they make sense.
3. Implement a stopgap condition to avoid rolling indefinitely.
4. Implement a stitchable segment simulation architecture
   4.1 First, make sure every individual simulation records their terminal NAV and IV and all the remaining positions are closed so
       the portfolio is easy to carry forward with these 2 parameters as new conditions for a subsequent simulation.
5. Compute the CVaR:
   5.1 Sort all the terminal returns in ascending order, and compute the average returns up to the P percentile
   5.2 We're interested in -15% 99-CVaR and -7% 90 CVaR (confirm this again with the LLM)
6. Plot the return distributions
7. Consider alternative portfolios
   7.1 15% SPY and 85% T-Bills + Short SPX puts
   7.2 100% T-bills and 0.75x notional SPX 5-15 delta credit spreads
   7.3 More short put / spreads cases varying the notional.
   7.3 Covered calls? 
8. Figure out how to discount the inflation
'''                 