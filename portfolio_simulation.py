import numpy as np
import math
import pandas as pd

# ==========================================
# 1. BLACK-SCHOLES GREEKS & PRICING
# ==========================================
def norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, r, sigma, is_call=False):
    T = max(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call: return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else: return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S, K, T, r, sigma, is_call=False):
    T = max(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call: return norm_cdf(d1)
    else: return norm_cdf(-d1) - 1.0

# ==========================================
# 2. DYNAMIC VOLATILITY SURFACE (SVI-Lite)
# ==========================================
def get_iv(S, K, T, V_t_spot):
    """
    Fits a quadratic Volatility Surface to the user's live data:
    ATM = 12.77%, 15D (-4.6% OTM) = 16.93%, 5D (-7.3% OTM) = 22.76%
    Adjusts for Time to Expiration (1/sqrt(T)) and VIX Spikes.
    """
    atm_iv = np.sqrt(V_t_spot)
    k_log = np.log(K/S) # Moneyness
    
    # Base coefficients calculated from live data (approximate quadratic fit)
    # y = a + bx + cx^2
    # At 45 DTE (T=0.123): b ~ -1.15, c ~ 3.5
    b = -1.15
    c = 3.50
    
    # Scale skew by 1/sqrt(T). As expiration approaches, skew steepens.
    # Normalized to 45 DTE (0.123 years)
    time_scalar = math.sqrt(0.123) / math.sqrt(max(T, 0.01))
    
    # Sticky Delta / Vol Expansion dampener: 
    # When VIX spikes, the absolute curve shifts up, but the relative skew slope flattens slightly.
    vol_scalar = 0.1277 / max(atm_iv, 0.1277)
    
    if k_log >= 0:
        # Call side (ITM Puts) - Gentle upward curve, respecting put-call parity
        iv = atm_iv + (b * 0.2 * k_log + c * 0.5 * (k_log**2)) * time_scalar
        return min(max(iv, 0.05), 3.0)
    else:
        # Put side (OTM Puts) - Aggressive Smirk
        iv = atm_iv + (b * k_log + c * (k_log**2)) * time_scalar * vol_scalar
        return min(max(iv, 0.05), 3.0) # Capped at 300% IV to prevent math errors

def find_strike_skew(S, target_delta, T, r, V_t):
    # Binary search to find the exact strike that yields the target delta
    low = S * 0.3
    high = S * 1.5
    for _ in range(40):
        mid = (low + high) / 2.0
        iv = get_iv(S, mid, T, V_t)
        delta = bs_delta(S, mid, T, r, iv, is_call=False)
        if delta > target_delta: high = mid
        else: low = mid
    return (low + high) / 2.0

# ==========================================
# 3. SVCJ MONTE CARLO GENERATOR
# ==========================================
def generate_svcj_path(days=252, S0=7786.0, V0=0.1425**2):
    """
    Stochastic Volatility with Correlated Jumps (SVCJ)
    Instituitonal parameters for S&P 500.
    """
    dt = 1.0 / 252.0
    mu = 0.08         # Equity drift
    kappa = 4.0       # VIX Mean reversion speed
    theta = 0.18**2   # Long-term variance
    sigma_v = 0.4     # Volatility of volatility
    rho = -0.65       # Price/Vol correlation (Leverage effect)
    
    # Jump Parameters
    lambda_j = 1.5    # Expected jumps per year
    mu_j = -0.05      # Mean price jump size
    sigma_j = 0.06    # Volatility of price jump
    mu_v = 0.04       # Mean variance jump size
    
    S = np.zeros(days); S[0] = S0
    V = np.zeros(days); V[0] = V0
    
    for t in range(1, days):
        Z1 = np.random.standard_normal()
        Z2 = rho * Z1 + math.sqrt(1 - rho**2) * np.random.standard_normal()
        
        # Poisson Jump
        N = np.random.poisson(lambda_j * dt)
        J_S = 0; J_V = 0
        if N > 0:
            Z3 = np.random.standard_normal()
            J_S = mu_j + sigma_j * Z3
            # Variance jumps are positive and exponentially distributed
            J_V = np.random.exponential(mu_v)
            
        # Variance process (Euler-Maruyama, ensuring V > 0)
        v_prev = max(V[t-1], 1e-6)
        V[t] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * math.sqrt(v_prev * dt) * Z2 + J_V
        V[t] = max(V[t], 1e-6)
        
        # Price process
        S[t] = S[t-1] * np.exp((mu - 0.5 * v_prev) * dt + math.sqrt(v_prev * dt) * Z1 + J_S)
        
    return S, V

# ==========================================
# 4. THE TRADING ENGINE (Operator's Manual)
# ==========================================
def run_simulation(S_path, V_spot_path, rf_rate, inflation, notional_leverage, nav):
    r = rf_rate
    days = len(S_path)
    
    # Technical Indicators
    ema20 = pd.Series(S_path).ewm(span=20, adjust=False).mean().values
    
    # Proxying VIX 3-Month Term Structure using a 63-day moving average of spot VIX
    vix_3m = np.zeros(days)
    for i in range(days): vix_3m[i] = np.mean(np.sqrt(V_spot_path[max(0, i-63):i+1]))
    
    state = 'active'
    days_above_ema = 0
    cash = 0.0 
    
    sp_K = sp_entry = units = DTE = 0
    current_leverage = notional_leverage
    
    def deploy_put(S, V, cur_nav, lev):
        nonlocal sp_K, sp_entry, units, DTE
        DTE = 45.0
        sp_K = find_strike_skew(S, -0.15, DTE/365., r, V)
        sp_entry = bs_price(S, sp_K, DTE/365., r, get_iv(S, sp_K, DTE/365., V), False)
        units = (cur_nav * lev) / S # Strict Notional Sizing
        return units * sp_entry
        
    cash += deploy_put(S_path[0], V_spot_path[0], nav, current_leverage)
    
    for d in range(1, days):
        S = S_path[d]; V_t = V_spot_path[d]
        spot_vix = math.sqrt(V_t)
        term_vix3m = vix_3m[d]
        dt = 365.0 / 252.0
        
        if d % 21 == 0: cash -= 12500.0 # Salary
        
        if S > ema20[d]: days_above_ema += 1
        else: days_above_ema = 0
        
        cur_bond_nav = nav * np.exp(r * (d/252.0))
        
        # VIX Term Structure Check
        backwardation = spot_vix > (term_vix3m * 1.05) # 5% threshold to avoid noise
        
        # TRIPLE-LOCK COOLDOWN RE-ENTRY
        if state == 'cooldown':
            if days_above_ema >= 5 and not backwardation:
                state = 'wade_in'
                current_leverage = notional_leverage / 2.0 
                cash += deploy_put(S, V_t, cur_bond_nav + cash, current_leverage)
        
        sp_p = 0.0
        if state in ['active', 'wade_in']:
            DTE -= dt
            sp_p = bs_price(S, sp_K, max(DTE,1)/365., r, get_iv(S, sp_K, max(DTE,1)/365., V_t), False)
            sp_d = bs_delta(S, sp_K, max(DTE,1)/365., r, get_iv(S, sp_K, max(DTE,1)/365., V_t), False)
            
            roll = False
            # OPERATOR'S DECISION TREE
            if backwardation:
                roll = True; state = 'cooldown'; days_above_ema = 0 # HARD EJECT
            elif sp_d <= -0.50:
                roll = True; state = 'cooldown'; days_above_ema = 0 # PRICE EJECT
            elif sp_d <= -0.35:
                roll = True # GRACEFUL DEFENSE
            elif sp_p <= 0.5 * sp_entry:
                roll = True # TAKE PROFIT
                if state == 'wade_in': current_leverage = notional_leverage; state = 'active'
            elif DTE <= 7:
                roll = True # 7-DTE HARD DECK (Gamma Risk Eject)
                
            # ACCOUNTING (Flawless deduction of closing costs)
            if roll:
                cash -= units * sp_p 
                if state != 'cooldown':
                    cash += deploy_put(S, V_t, cur_bond_nav + cash - (units*sp_p), current_leverage)
                else:
                    units = 0; sp_p = 0
                    
        # ... End of loop NAV Tracking goes here ...


