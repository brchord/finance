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

crit = ui.criteria()
real = ui.real_dollars()


def frontier_of(review: store.Review) -> decision.Frontier:
    cells = ui.model_cells(ui.merged_cells(review), decision.DECISION_MODEL,
                           review.profile.tax_regime)
    return decision.frontier(cells, crit)


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
        "P(ruin)": best.ruin_rate * 100 if best else None,
        "ES10 age": (decision.month_to_age(best.ruin_month_es10,
                                           r.profile.retirement_age)
                     if best else None),
        "Assumptions": ui.assumptions_caption(r.profile),
    })
money = st.column_config.NumberColumn(format="$%,.0f")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={
                 "NAV": money, "Max spending": money,
                 "First failing": money, "Estimate": money,
                 "P(ruin)": st.column_config.NumberColumn(format="%.2f%%"),
                 "ES10 age": st.column_config.NumberColumn(format="%.1f"),
             })
st.caption(f"Decision model {decision.DECISION_MODEL}, "
           f"{crit.ruin_ceiling:.0%} ruin ceiling. Allocation, P(ruin) "
           "and ES10 are for the best cell at the max spending.")

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
        "Max spending": ui.money(f.best_spending),
        "Crossing estimate": ui.money(f.estimate),
        "Best allocation": best.allocation if best else "—",
        "P(ruin)": ui.pct(best.ruin_rate) if best else "—",
        "ES10 age": (ui.age(best.ruin_month_es10, p.retirement_age)
                     if best else "—"),
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
        curves, current.profile.retirement_age, crit.ruin_ceiling, theme),
        theme="streamlit")
    st.caption("A curve shifting up after a good stretch is the signal to "
               "consider raising spending.")
