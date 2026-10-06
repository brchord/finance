"""
History: how the maximum sustainable spending moves from review to
review, and a side-by-side comparison of two reviews.
"""

import pandas as pd
import streamlit as st

from planner import charts, decision, store, ui

st.title("History")
reviews = store.list_reviews()
if not reviews:
    st.info("Create a review in the Explorer first.")
    st.stop()

real = ui.real_dollars()
# Each review is judged under its own retirement age and household.
crits = {r.id: ui.criteria(r) for r in reviews}
ceiling = crits[reviews[0].id].ruin_ceiling


def frontier_of(review: store.Review) -> decision.Frontier:
    cells = ui.model_cells(ui.merged_cells(review), decision.DECISION_MODEL,
                           review.profile.tax_regime)
    return decision.frontier(cells, crits[review.id])


frontiers = {r.id: frontier_of(r) for r in reviews}

rows = []
for r in reviews:
    f = frontiers[r.id]
    best = f.best_cell.cell if f.best_cell else None
    rows.append({
        "Date": r.date,
        "Review": r.label,
        "NAV": r.profile.initial_nav,
        "Max spending": f.best_spending,
        "First failing": f.next_failing_spending,
        "Estimate": f.estimate,
        "Allocation": best.allocation if best else None,
        "Lifetime ruin": crits[r.id].ruin(best) * 100 if best else None,
        "Years in ruin": (crits[r.id].years_in_ruin(best)
                          if best else None),
        "Assumptions": ui.assumptions_caption(r.profile),
    })
money = st.column_config.NumberColumn(format="$%,.0f")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={
                 "NAV": money, "Max spending": money,
                 "First failing": money, "Estimate": money,
                 "Lifetime ruin": st.column_config.NumberColumn(
                     format="%.2f%%"),
                 "Years in ruin": st.column_config.NumberColumn(
                     format="%.3f"),
             })
st.caption(f"Decision model {decision.DECISION_MODEL}, "
           f"{ceiling:.0%} lifetime ruin ceiling, each review under its own "
           "life expectancy settings. Allocation, lifetime ruin and years "
           "in ruin are for the best cell at the max spending.")

theme = ui.theme()
dated = [r for r in reversed(reviews)
         if frontiers[r.id].best_spending is not None]
if len(dated) >= 2:
    st.plotly_chart(charts.spending_timeline(
        [r.date for r in dated],
        [frontiers[r.id].best_spending for r in dated],
        [r.label for r in dated], theme), theme="streamlit")

if len(reviews) < 2:
    st.stop()

st.subheader("Compare two reviews")
labels = {r.id: f"{r.label} · {r.date}" for r in reviews}
ids = [r.id for r in reviews]
c1, c2 = st.columns(2)
current_id = c1.selectbox("Current", ids, index=0,
                          format_func=labels.__getitem__)
previous_id = c2.selectbox("Previous", ids, index=1,
                           format_func=labels.__getitem__)
current = next(r for r in reviews if r.id == current_id)
previous = next(r for r in reviews if r.id == previous_id)
fc, fp = frontiers[current.id], frontiers[previous.id]


def describe(f: decision.Frontier, review: store.Review) -> dict:
    best = f.best_cell.cell if f.best_cell else None
    p = review.profile
    h = review.household
    crit = crits[review.id]
    r = real and best is not None and best.p10_real_return is not None
    suffix = "" if best is None or r == real else " (nominal)"
    return {
        "NAV": ui.money(p.initial_nav),
        "Retirement age": f"{p.retirement_age:g}",
        "Terminal age": f"{p.terminal_age:g}",
        "Tax regime": p.tax_regime,
        "Starting CAPE": f"{p.initial_cape:g}",
        "Long-run CAPE": f"{p.target_cape:g}",
        "Earnings growth": f"{p.annual_earnings_growth:.2%}",
        "Buyback yield": f"{p.annual_buyback_yield:.2%}",
        "Dividend yield": f"{p.dividend_yield:.2%}",
        "Plan for": "one person" if h.partner is None else "a couple",
        "Modal age at death": f"{h.person.modal_age:g}",
        "Mortality spread": f"{h.person.dispersion:g}",
        "Partner (age diff / modal / spread)": (
            "—" if h.partner is None else
            f"{h.partner_age_offset:+g} / {h.partner.modal_age:g} / "
            f"{h.partner.dispersion:g}"),
        "Max spending": ui.money(f.best_spending),
        "Crossing estimate": ui.money(f.estimate),
        "Best allocation": best.allocation if best else "—",
        "Lifetime ruin": ui.pct(crit.ruin(best)) if best else "—",
        "Years in ruin": ui.years(crit.years_in_ruin(best)) if best else "—",
        f"P10 return ({ui.dollars_label(real)})":
            ui.total_return(best.pct_return(10, r)) + suffix
            if best else "—",
        f"P50 return ({ui.dollars_label(real)})":
            ui.total_return(best.pct_return(50, r)) + suffix
            if best else "—",
    }


a, b = describe(fc, current), describe(fp, previous)
diff = pd.DataFrame({"Previous": b, "Current": a})
diff["Changed"] = ["•" if a[k] != b[k] else "" for k in a]
# Tall enough for every row (35px each plus the header).
st.dataframe(diff, width="stretch", height=35 * (len(diff) + 1) + 3)

curves = []
if fc.best_cell:
    curves.append((f"{current.label} ({fc.best_cell.cell.allocation} at "
                   f"{ui.money(fc.best_spending)})", fc.best_cell.cell,
                   theme.series_1))
if fp.best_cell:
    curves.append((f"{previous.label} ({fp.best_cell.cell.allocation} at "
                   f"{ui.money(fp.best_spending)})", fp.best_cell.cell,
                   theme.series_2))
if curves and current.profile.retirement_age == previous.profile.retirement_age:
    st.markdown("**Survival of each review's best cell**")
    st.plotly_chart(charts.survival_curves(
        curves, current.profile.retirement_age, theme),
        theme="streamlit")
    st.caption("A curve shifting up after a good stretch is the signal to "
               "consider raising spending.")
