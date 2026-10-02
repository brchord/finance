"""
Cell detail: one (spending, allocation) under the decision model next to
the reference model.
"""

import streamlit as st

from planner import charts, decision, ui

review = ui.selected_review()
st.title("Cell detail")
if review is None:
    st.info("Create a review in the Explorer first.")
    st.stop()

merged = ui.merged_cells(review)
tax_regime = review.profile.tax_regime
cells = ui.model_cells(merged, decision.DECISION_MODEL, tax_regime)
if not cells:
    st.info("This review has no results yet. Run a sweep in the Explorer.")
    st.stop()

crit = ui.criteria()
ret_age = review.profile.retirement_age
ref_model = ui.reference_model()
by_key = {(c.spending, c.equity): c for c in cells}
ref_by_key = {(c.spending, c.equity): c
              for c in ui.model_cells(merged, ref_model, tax_regime)}

# Default: the cell picked in the Explorer, else the frontier winner.
chosen = st.session_state.get("cell")
if chosen not in by_key:
    best = decision.frontier(cells, crit).best_cell
    chosen = ((best.cell.spending, best.cell.equity) if best
              else min(by_key))

levels = sorted({k[0] for k in by_key})
c1, c2 = st.columns(2)
spending = c1.selectbox("Spending", levels, index=levels.index(chosen[0]),
                        format_func=ui.money)
equities = sorted(e for s, e in by_key if s == spending)
equity = c2.selectbox(
    "Allocation", equities,
    index=equities.index(chosen[1]) if chosen[1] in equities else 0,
    format_func=lambda e: by_key[(spending, e)].allocation)
st.session_state["cell"] = (spending, equity)

cell = by_key[(spending, equity)]
ref = ref_by_key.get((spending, equity))
st.caption(f"**{review.label}** · {ui.profile_caption(review.profile)} · "
           f"{cell.total_paths:,} paths")


def kpis(title: str, c: decision.Cell):
    st.markdown(f"**{title}**")
    k = st.columns(5)
    passes = crit.passes(c)
    k[0].metric("P(ruin)", ui.pct(c.ruin_rate),
                delta="passes" if passes else "fails ceiling",
                delta_color="normal" if passes else "inverse",
                delta_arrow="off")
    k[1].metric("ES10 ruin age", ui.age(c.ruin_month_es10, ret_age))
    k[2].metric("ES5 ruin age", ui.age(c.ruin_month_es5, ret_age))
    k[3].metric("P10 return", ui.total_return(c.p10_return))
    k[4].metric("P50 return", ui.total_return(c.p50_return))
    st.caption(f"{c.ruin_count:,} ruined paths · earliest ruin at age "
               f"{ui.age(c.ruin_month_min, ret_age)} (a single path; noisy)"
               f" · P5 return {ui.total_return(c.p5_return)}"
               f" · P25 return {ui.total_return(c.p25_return)}")


kpis(f"{decision.DECISION_MODEL} (decision)", cell)
if ref is not None:
    kpis(f"{ref_model} (reference)", ref)
else:
    st.caption(f"No {ref_model} result for this cell: it wasn't part of "
               "the runs. Run a sweep with it selected as the reference.")

theme = ui.theme()
st.subheader("Survival")
curves = [(decision.DECISION_MODEL, cell, theme.series_1)]
if ref is not None:
    curves.append((f"{ref_model} (reference)", ref, theme.series_2))
st.plotly_chart(charts.survival_curves(curves, ret_age, crit.ruin_ceiling,
                                       theme), theme="streamlit")

st.subheader("Age at ruin")
if cell.ruin_count == 0:
    st.write("No path is ruined within the horizon.")
else:
    st.plotly_chart(charts.ruin_age_histogram(cell, ret_age,
                                              theme.series_1, theme),
                    theme="streamlit")
    st.caption(decision.DECISION_MODEL)
