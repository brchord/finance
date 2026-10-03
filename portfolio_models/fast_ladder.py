"""
fast_ladder.py

Numba port of LongSPYWithTreasuryLadders.run_simulation (linear_models.py),
restricted to the Monte Carlo use case (full_book=False): no bookkeeping
lists, no pandas objects, no transaction book.

This module does not replace or modify LongSPYWithTreasuryLadders, which
remains the reference implementation -- see tests/test_fast_ladder.py for
the parity tests that pin this port to it.

Structural simplifications applied here:

- TaxLotTracker reduces to two scalars. `buy()` is only ever called once,
  at month 0, so the single lot's cost basis is `spy_price` for the entire
  simulation and its holding period is simply `month >= 12`. We don't even
  need a running share count for the lot itself: `spy_position_size` (which
  the reference implementation already tracks) *is* the lot's share count.
- The T-Note ladder becomes fixed-capacity arrays (maturity, amount,
  rate) instead of a dict, kept in purchase order like the dict's
  insertion order. 64 is more than the worst case (an initial 3 plus at
  most one new tranche per month, each expiring within 60 months).
- Tax law flattens to per-year bracket arrays, computed by *calling* the
  real TaxRegimeScenario.resolve() once per year at cpi_relative=1.0 (see
  `flatten_tax_regime`) rather than reimplementing each regime's logic.
  Only the CPI scaling step (which depends on each path's own simulated
  CPI) happens inside the kernel.
- The reference implementation calls `_get_needed_liquidity` twice per
  month (once in the rebalancing block, once in the reserve-deficit block)
  against an identical ladder state, so it always returns the same value
  both times; this port computes it once and reuses it. This is a
  numerically-exact deduplication, not a behavior change.

Everything else is preserved exactly. Comments below reference the
corresponding line numbers in linear_models.py so a diff against the
reference is easy to audit.

Bit-for-bit, not just "close": outputs must equal the reference's exactly
(tests/test_fast_ladder.py compares with assert_array_equal). Rounding
differences of one unit in the last place are not harmless here: the
strategy branches on thresholds (share counts via ceil/floor, rebalancing
and runway comparisons), so on rare paths a last-digit difference flips a
decision and the path diverges by percent. An earlier version that was
only "close" diverged visibly on 9 of 600,000 paths of a real config.
Three things the reference does implicitly are therefore reproduced
exactly:

- Moving averages: the reference uses pandas' rolling mean, an online
  Kahan-compensated algorithm whose results differ in the last digit from
  a direct average of the same window. _pandas_rolling_mean is a port of
  pandas' kernel, verified equal to pandas on ~10M values.
- The T-note total: the reference uses Python's builtin sum(), which (on
  Python >= 3.12) is compensated while every item is an exact float and
  switches to plain addition at the first numpy float64. The three
  starting notes hold Python floats; every note bought later holds a
  numpy float64 (its amount derives from day_spy). _python_sum reproduces
  this, with a per-note flag recording which kind each amount is.
- Addition order: notes are kept in purchase order, and each month's
  coupons are all added before any matured principal, as the reference's
  two loops do. Floating-point addition isn't associative.

(The `expected_dividends` calculation once used month-0 `spy_price`
instead of the current month's `day_spy` -- a bug in the reference
implementation, fixed there and here together; see git history for the
fix commit.)
"""

import math
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

from tax_models.regimes import TaxRegimeScenario

N_ORDINARY_BRACKETS = 7
N_LTCG_BRACKETS = 3
T_NOTE_SLOTS = 64
LONG_TERM_HOLDING_MONTHS = 12


@njit(cache=True)
def _pandas_rolling_mean(values, window):
    """
    pd.Series(values).rolling(window=window, min_periods=1).mean(), bit for
    bit: a port of pandas' roll_mean kernel (Kahan-compensated running sum
    with separate add/remove compensation, plus its sign and
    repeated-value corrections). A direct average of each window differs
    from pandas in the last digit on most values. Verified identical to
    pandas 3.0.6 on ~10M values, including repeated values, mixed signs,
    signed zeros and magnitudes from 1e-8 to 1e10.
    """
    n = values.shape[0]
    out = np.empty(n)
    nobs = 0
    neg_ct = 0
    sum_x = 0.0
    comp_add = 0.0
    comp_rem = 0.0
    same_run = 0
    prev_value = values[0] if n > 0 else 0.0
    for i in range(n):
        if i >= window:
            val = values[i - window]
            if val == val:
                nobs -= 1
                y = -val - comp_rem
                t = sum_x + y
                comp_rem = t - sum_x - y
                sum_x = t
                if math.copysign(1.0, val) < 0.0:
                    neg_ct -= 1
        val = values[i]
        if val == val:
            nobs += 1
            y = val - comp_add
            t = sum_x + y
            comp_add = t - sum_x - y
            sum_x = t
            if math.copysign(1.0, val) < 0.0:
                neg_ct += 1
            if val == prev_value:
                same_run += 1
            else:
                same_run = 1
            prev_value = val
        if nobs > 0:
            result = sum_x / nobs
            if same_run >= nobs:
                result = prev_value
            elif neg_ct == 0 and result < 0.0:
                result = 0.0
            elif neg_ct == nobs and result > 0.0:
                result = 0.0
            out[i] = result
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _cs_to_double(hi, lo):
    if lo != 0.0 and math.isfinite(lo):
        return hi + lo
    return hi


@njit(cache=True)
def _python_sum(values, is_py_float, n):
    """
    Python's builtin sum() over values[:n], where is_py_float[i] says
    whether that item would be an exact Python float (True) or a numpy
    float64 (False) in the reference. CPython (>= 3.12) adds exact floats
    with Neumaier compensation, but at the first item that isn't an exact
    float it folds the compensation into the running total and continues
    with plain addition for the rest. Verified identical to sum() on
    100,000 mixed float/np.float64 lists.
    """
    hi = 0.0
    lo = 0.0
    compensated = True
    acc = 0.0
    for i in range(n):
        x = values[i]
        if compensated and is_py_float[i]:
            t = hi + x
            if abs(hi) >= abs(x):
                lo += (hi - t) + x
            else:
                lo += (x - t) + hi
            hi = t
        elif compensated:
            acc = _cs_to_double(hi, lo) + x
            compensated = False
        else:
            acc = acc + x
    if compensated:
        return _cs_to_double(hi, lo)
    return acc


def flatten_tax_regime(
    tax_regime: Optional[TaxRegimeScenario], max_years: int
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, float, float]:
    """
    Precomputes, for each simulated year 0..max_years-1, the unscaled
    (cpi_relative_to_start=1.0) ordinary/LTCG bracket tables that
    `tax_regime` resolves to -- by calling the regime's own `resolve()`,
    never reimplementing its logic. CPI scaling is applied per-path,
    per-year inside the Numba kernel instead, since it depends on each
    path's own simulated CPI trajectory.

    If `tax_regime` is None, returns has_tax=False and zero-filled arrays
    (unused by the kernel in that case).

    Raises ValueError if a regime's bracket table ever has a different
    number of brackets than the current regimes do (would require
    widening N_ORDINARY_BRACKETS/N_LTCG_BRACKETS and this function), or if
    NIIT parameters vary by year (the kernel takes them as plain scalars).
    """
    if tax_regime is None:
        return (
            False,
            np.zeros((1, N_ORDINARY_BRACKETS)),
            np.zeros((1, N_ORDINARY_BRACKETS)),
            np.zeros(1),
            np.zeros((1, N_LTCG_BRACKETS)),
            np.zeros((1, N_LTCG_BRACKETS)),
            0.0, 0.0,
        )

    max_years = max(max_years, 1)
    ord_rates = np.zeros((max_years, N_ORDINARY_BRACKETS))
    ord_floors = np.zeros((max_years, N_ORDINARY_BRACKETS))
    ord_deduction = np.zeros(max_years)
    ltcg_rates = np.zeros((max_years, N_LTCG_BRACKETS))
    ltcg_floors = np.zeros((max_years, N_LTCG_BRACKETS))

    niit_rate = None
    niit_threshold = None

    for year in range(max_years):
        law = tax_regime.resolve(year, 1.0)

        if len(law.ordinary.brackets) != N_ORDINARY_BRACKETS:
            raise ValueError(
                f"Regime '{tax_regime.name}' year {year} resolved to "
                f"{len(law.ordinary.brackets)} ordinary brackets, expected "
                f"{N_ORDINARY_BRACKETS}; fast_ladder.flatten_tax_regime "
                "needs updating to match.")
        if len(law.ltcg.brackets) != N_LTCG_BRACKETS:
            raise ValueError(
                f"Regime '{tax_regime.name}' year {year} resolved to "
                f"{len(law.ltcg.brackets)} LTCG brackets, expected "
                f"{N_LTCG_BRACKETS}; fast_ladder.flatten_tax_regime needs "
                "updating to match.")

        for i, (rate, floor) in enumerate(law.ordinary.brackets):
            ord_rates[year, i] = rate
            ord_floors[year, i] = floor
        ord_deduction[year] = law.ordinary.standard_deduction

        for i, (rate, floor) in enumerate(law.ltcg.brackets):
            ltcg_rates[year, i] = rate
            ltcg_floors[year, i] = floor

        if niit_rate is None:
            niit_rate = law.niit.rate
            niit_threshold = law.niit.threshold_single
        elif (law.niit.rate != niit_rate
              or law.niit.threshold_single != niit_threshold):
            raise ValueError(
                f"Regime '{tax_regime.name}' varies NIIT parameters by "
                "year; fast_ladder's kernel assumes NIIT rate/threshold "
                "are constant across the simulation horizon.")

    assert niit_rate is not None and niit_threshold is not None
    return (True, ord_rates, ord_floors, ord_deduction,
            ltcg_rates, ltcg_floors, niit_rate, niit_threshold)


@njit(cache=True)
def _bracket_tax(taxable_income, rates, floors):
    """Mirrors BracketTable.tax_on (tax_models/regimes.py)."""
    n = rates.shape[0]
    if taxable_income <= 0.0:
        return 0.0
    tax = 0.0
    for i in range(n):
        floor = floors[i]
        if i + 1 < n:
            top = min(taxable_income, floors[i + 1])
        else:
            top = taxable_income
        if top <= floor:
            break
        tax += (top - floor) * rates[i]
    return tax


@njit(cache=True)
def _compute_tax(ordinary_income, preferential_income,
                  ord_rates_y, ord_floors_y, ord_deduction_y,
                  ltcg_rates_y, ltcg_floors_y,
                  niit_rate, niit_threshold, cpi_relative_to_start):
    """Mirrors AnnualTaxLaw.compute_tax (tax_models/regimes.py), with the
    bracket floors and standard deduction CPI-scaled here instead of ahead
    of time (see flatten_tax_regime)."""
    deduction = ord_deduction_y * cpi_relative_to_start
    total_income = ordinary_income + preferential_income

    ordinary_taxable = max(0.0, ordinary_income - deduction)
    unused_deduction = max(0.0, deduction - ordinary_income)
    preferential_taxable = max(0.0, preferential_income - unused_deduction)

    n_ord = ord_rates_y.shape[0]
    scaled_ord_floors = np.empty(n_ord)
    for i in range(n_ord):
        scaled_ord_floors[i] = ord_floors_y[i] * cpi_relative_to_start
    ordinary_tax = _bracket_tax(ordinary_taxable, ord_rates_y, scaled_ord_floors)

    n_ltcg = ltcg_rates_y.shape[0]
    scaled_ltcg_floors = np.empty(n_ltcg)
    for i in range(n_ltcg):
        scaled_ltcg_floors[i] = ltcg_floors_y[i] * cpi_relative_to_start

    stack_floor = ordinary_taxable
    stack_ceiling = ordinary_taxable + preferential_taxable
    preferential_tax = (
        _bracket_tax(stack_ceiling, ltcg_rates_y, scaled_ltcg_floors)
        - _bracket_tax(stack_floor, ltcg_rates_y, scaled_ltcg_floors)
    )

    niit_tax = niit_rate * max(0.0, total_income - niit_threshold)
    return ordinary_tax + preferential_tax + niit_tax


@njit(cache=True)
def run_simulation_fast(
    spx, cpi, yield3m, yield5y,
    initial_nav, months,
    equity_allocation, ladder_allocation, yearly_spending, spy_div_yield,
    has_tax,
    ord_rates, ord_floors, ord_deduction,
    ltcg_rates, ltcg_floors,
    niit_rate, niit_threshold,
):
    """
    Numba port of LongSPYWithTreasuryLadders.run_simulation, full_book=False
    only. Returns (nav_path, ruin_month): nav_path matches the reference's
    return value exactly (zero-filled from the ruin month onward); ruin_month
    is -1 if the path never ruins, else the month index where NAV hit <= 0.

    tax regime arrays come from flatten_tax_regime(); pass has_tax=False and
    any correctly-shaped zero arrays when the strategy has no tax_regime.
    """
    nav_path = np.zeros(months)

    monthly_withdrawal = yearly_spending / 12.0
    spx_prices = spx / 10.0
    spy_price = spx_prices[0]

    # Single tax lot for the whole simulation (linear_models.py:235-237):
    # spy_position_size IS the lot's share count, and its cost basis is
    # spy_price for the entire run since no further buys ever occur.
    spy_position_size = int(math.ceil(
        initial_nav * equity_allocation / spy_price))

    tnote_amount = initial_nav * ladder_allocation / 5.0

    # The ladder in purchase order (the reference dict's insertion order);
    # entries [0, n_notes) are live. t_py marks amounts that are exact
    # Python floats in the reference -- the three starting notes, derived
    # from initial_nav and the allocation (Python numbers from the config)
    # -- as opposed to numpy float64s, which every later purchase is. See
    # _python_sum.
    t_maturity = np.zeros(T_NOTE_SLOTS, dtype=np.int64)
    t_amount = np.zeros(T_NOTE_SLOTS)
    t_rate = np.zeros(T_NOTE_SLOTS)
    t_py = np.zeros(T_NOTE_SLOTS, dtype=np.bool_)
    t_expired = np.zeros(T_NOTE_SLOTS, dtype=np.bool_)

    t_maturity[0], t_amount[0], t_rate[0], t_py[0] = (
        24, tnote_amount, yield5y[0], True)
    t_maturity[1], t_amount[1], t_rate[1], t_py[1] = (
        36, tnote_amount * 2.0, yield5y[0], True)
    t_maturity[2], t_amount[2], t_rate[2], t_py[2] = (
        60, tnote_amount, yield5y[0], True)
    n_notes = 3

    cash = (initial_nav - (tnote_amount * 4.0)
            - (spy_price * spy_position_size))

    # Rolling means and the shifted CPI pct change
    # (linear_models.py:296-305), reproducing pandas exactly.
    sma2 = _pandas_rolling_mean(spx_prices, 2)
    sma4 = _pandas_rolling_mean(spx_prices, 4)
    sma9 = _pandas_rolling_mean(spx_prices, 9)

    cpi_pct = np.zeros(months)
    for m in range(months - 1):
        cpi_pct[m] = cpi[m + 1] / cpi[m] - 1.0
    # cpi_pct[months - 1] stays 0.0, matching the reference's explicit
    # override of the otherwise-undefined last shifted value.

    ruined = False
    ruin_month = -1

    ytd_ordinary_income = 0.0
    ytd_preferential_income = 0.0
    pending_tax_due = 0.0

    for m in range(months):
        day_spy = spx_prices[m]

        # Settle last year's tax bill (linear_models.py:327-365).
        if pending_tax_due > 0.0:
            tax_shortfall = pending_tax_due - cash
            if tax_shortfall > 0.0:
                selling_position = int(math.ceil(tax_shortfall / day_spy))
                selling_position = min(selling_position, spy_position_size)
                if selling_position > 0:
                    gain = selling_position * (day_spy - spy_price)
                    if m >= LONG_TERM_HOLDING_MONTHS:
                        ytd_preferential_income += gain
                    else:
                        ytd_ordinary_income += gain
                    spy_position_size -= selling_position
                    cash += selling_position * day_spy

            payment = min(pending_tax_due, cash)
            cash -= payment
            pending_tax_due = 0.0

        # Monthly withdrawal (linear_models.py:367-368).
        cash -= monthly_withdrawal

        # Quarterly SPY dividend (linear_models.py:379-384).
        if m % 3 == 0 and m > 0:
            cash_dividend = spy_position_size * day_spy * spy_div_yield / 4.0
            if cash_dividend > 0.0:
                cash += cash_dividend
                ytd_preferential_income += cash_dividend

        # Monthly T-Bill yield accrual (linear_models.py:396-400).
        tbill_interest = cash * (yield3m[m] / 12.0)
        cash += tbill_interest
        ytd_ordinary_income += tbill_interest

        # Semi-annual T-Note coupons and maturities (linear_models.py:
        # 404-435): two passes in purchase order, like the reference --
        # every coupon into cash first, then every matured principal.
        # Interleaving them would change the order cash is summed in.
        for idx in range(n_notes):
            months_remaining = t_maturity[idx] - m
            if months_remaining % 6 == 0 and m > 0:
                coupon = t_amount[idx] * (t_rate[idx] / 2.0)
                cash += coupon
                ytd_ordinary_income += coupon
            t_expired[idx] = months_remaining <= 0

        kept = 0
        for idx in range(n_notes):
            if t_expired[idx]:
                cash += t_amount[idx]
            else:
                t_maturity[kept] = t_maturity[idx]
                t_amount[kept] = t_amount[idx]
                t_rate[kept] = t_rate[idx]
                t_py[kept] = t_py[idx]
                kept += 1
        n_notes = kept

        tnote_position = _python_sum(t_amount, t_py, n_notes)

        current_nav = cash + spy_position_size * day_spy + tnote_position
        fixed_income_position = cash + tnote_position

        # _get_needed_liquidity is called twice per month against an
        # identical ladder state in the reference (linear_models.py:
        # 449-450 and 481-482); computed once here and reused (see module
        # docstring).
        min_maturity = -1
        for idx in range(n_notes):
            if min_maturity == -1 or t_maturity[idx] < min_maturity:
                min_maturity = t_maturity[idx]
        if min_maturity == -1:
            req_liquidity = monthly_withdrawal * 12.0
        else:
            req_liquidity = monthly_withdrawal * max(1, min_maturity - m)

        # Strategic Equity Rebalancing into Fixed Income
        # (linear_models.py:443-477).
        if (current_nav > 0.0
                and fixed_income_position / current_nav
                <= ladder_allocation * 0.8):
            if (m >= 9 and day_spy >= sma2[m] and day_spy >= sma4[m]
                    and day_spy >= sma9[m]):
                needed_amount = (current_nav * ladder_allocation
                                 - fixed_income_position)
                if cash > req_liquidity:
                    extra_liquidity = cash - req_liquidity
                    amount_to_sell = min(needed_amount, extra_liquidity)
                    selling_position = int(
                        math.floor(amount_to_sell / day_spy))
                    selling_position = min(
                        selling_position, spy_position_size)
                    if selling_position > 0:
                        gain = selling_position * (day_spy - spy_price)
                        if m >= LONG_TERM_HOLDING_MONTHS:
                            ytd_preferential_income += gain
                        else:
                            ytd_ordinary_income += gain
                        spy_position_size -= selling_position
                        cash += selling_position * day_spy

        # Reserve Deficit Protection (linear_models.py:479-513).
        div_events_expected = math.floor(
            req_liquidity / monthly_withdrawal / 3.0)
        expected_dividends = (spy_position_size * day_spy * spy_div_yield
                              * (div_events_expected / 4.0))
        spending_needs = req_liquidity - expected_dividends
        current_runway = cash - spending_needs

        if current_runway < 0.0:
            selling_position = int(math.ceil(-current_runway / day_spy))
            selling_position = min(selling_position, spy_position_size)
            if selling_position > 0:
                gain = selling_position * (day_spy - spy_price)
                if m >= LONG_TERM_HOLDING_MONTHS:
                    ytd_preferential_income += gain
                else:
                    ytd_ordinary_income += gain
                spy_position_size -= selling_position
                cash += selling_position * day_spy

        # Surplus Cash Allocation into T-Note Ladder (linear_models.py:
        # 516-544). `current_nav` and `fixed_income_position` here are
        # deliberately the stale values computed at the top of the month
        # (before this month's rebalancing/reserve sells) -- the reference
        # never recomputes them until after this block, so neither do we.
        if cash >= (2.0 * current_nav * ladder_allocation / 5.0):
            tnote_maturity = 24
            if n_notes > 0:
                max_maturity = t_maturity[0]
                for idx in range(1, n_notes):
                    if t_maturity[idx] > max_maturity:
                        max_maturity = t_maturity[idx]
                furthest_maturity = max_maturity - m
                for target_m in (24, 36, 60):
                    if furthest_maturity < target_m:
                        tnote_maturity = target_m
                        break

            tnote_tranche = current_nav * ladder_allocation / 5.0
            if (cash >= (tnote_tranche + spending_needs)
                    and tnote_tranche > 0.0):
                if n_notes == T_NOTE_SLOTS:
                    raise RuntimeError(
                        "fast_ladder: T-Note ladder exceeded its fixed "
                        "64-note capacity")
                t_maturity[n_notes] = m + tnote_maturity
                t_amount[n_notes] = tnote_tranche
                t_rate[n_notes] = yield5y[m]
                t_py[n_notes] = False  # derives from day_spy: np.float64
                n_notes += 1
                cash -= tnote_tranche

        # Discount current reported inflation (linear_models.py:547).
        monthly_withdrawal *= 1.0 + cpi_pct[m]

        # Year-End Tax Liability Computation (linear_models.py:582-597).
        if has_tax and (m + 1) % 12 == 0:
            year = m // 12
            cpi_relative_to_start = cpi[m] / cpi[0]
            pending_tax_due = _compute_tax(
                ytd_ordinary_income, ytd_preferential_income,
                ord_rates[year], ord_floors[year], ord_deduction[year],
                ltcg_rates[year], ltcg_floors[year],
                niit_rate, niit_threshold, cpi_relative_to_start,
            )
            ytd_ordinary_income = 0.0
            ytd_preferential_income = 0.0

        tnote_position = _python_sum(t_amount, t_py, n_notes)
        current_nav = cash + spy_position_size * day_spy + tnote_position

        if current_nav <= 0.0:
            ruined = True
            ruin_month = m
            break

        nav_path[m] = current_nav

    if not ruined and pending_tax_due > 0.0:
        payment = min(pending_tax_due, cash)
        cash -= payment
        nav_path[months - 1] -= payment

    return nav_path, ruin_month


@njit(parallel=True, cache=True)
def run_simulation_fast_batch(
    spx_paths, cpi_paths, yield3m_paths, yield5y_paths,
    initial_nav, months,
    equity_allocation, ladder_allocation, yearly_spending, spy_div_yield,
    has_tax,
    ord_rates, ord_floors, ord_deduction,
    ltcg_rates, ltcg_floors,
    niit_rate, niit_threshold,
):
    """
    Parallel batch wrapper over run_simulation_fast: `*_paths` are 2-D
    (num_paths, months) arrays, as returned by PathSimulator.simulate_paths.
    Returns (final_navs, ruin_months, annual_navs): one entry per path,
    and annual_navs[i, y] is path i's NAV at the end of year y
    (annual_navs[i, 0] is initial_nav; see annual_snapshot_months).
    """
    num_paths = spx_paths.shape[0]
    final_navs = np.empty(num_paths)
    ruin_months = np.empty(num_paths, dtype=np.int64)
    snapshot_months = annual_snapshot_months(months)
    annual_navs = np.empty((num_paths, snapshot_months.shape[0] + 1))

    for i in prange(num_paths):
        nav_path, ruin_month = run_simulation_fast(
            spx_paths[i], cpi_paths[i], yield3m_paths[i], yield5y_paths[i],
            initial_nav, months,
            equity_allocation, ladder_allocation, yearly_spending,
            spy_div_yield, has_tax,
            ord_rates, ord_floors, ord_deduction,
            ltcg_rates, ltcg_floors,
            niit_rate, niit_threshold,
        )
        final_navs[i] = nav_path[-1]
        ruin_months[i] = ruin_month
        annual_navs[i, 0] = initial_nav
        for y in range(snapshot_months.shape[0]):
            annual_navs[i, y + 1] = nav_path[snapshot_months[y]]

    return final_navs, ruin_months, annual_navs


@njit(cache=True)
def annual_snapshot_months(months):
    """
    Month indices of each year-end within a simulation of `months` months:
    nav_path[m] is the NAV at the end of month m, so year y (1-based) ends
    at index 12 * y - 1. A trailing partial year has no snapshot.
    """
    return np.arange(1, months // 12 + 1) * 12 - 1
