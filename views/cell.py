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

crit = ui.criteria(review)
ret_age = review.profile.retirement_age
terminal = review.profile.terminal_age
household = review.household
real = ui.real_dollars()
ages = decision.ruin_ages(ret_age, terminal)
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
    # Results from before real returns existed fall back to nominal.
    r = real and c.p10_real_return is not None
    k = st.columns(5)
    passes = crit.passes(c)
    k[0].metric("Lifetime ruin", ui.pct(crit.ruin(c)),
                delta="passes" if passes else "fails ceiling",
                delta_color="normal" if passes else "inverse",
                delta_arrow="off",
                help="P(ruined while still alive). The ceiling applies "
                     "to this.")
    k[1].metric("Years in ruin", ui.years(crit.years_in_ruin(c)),
                help="Expected years lived after the money runs out, "
                     "averaged over all paths (0 where it never does).")
    k[2].metric(f"Ruin by {terminal:g}", ui.pct(c.ruin_rate),
                help="Share of all paths ruined by the end of the "
                     "horizon, whether or not you'd still be alive.")
    k[3].metric("P10 return", ui.total_return(c.pct_return(10, r)),
                help=f"Total return over the horizon, "
                     f"{ui.dollars_label(r)}.")
    k[4].metric("P50 return", ui.total_return(c.pct_return(50, r)),
                help=f"Total return over the horizon, "
                     f"{ui.dollars_label(r)}.")
    if ages:
        k = st.columns(5)
        for col, a in zip(k, ages):
            col.metric(f"P(ruin) before {a}",
                       ui.pct(decision.ruin_prob_before(c, a, ret_age)),
                       help="Share of all paths ruined before this age.")
    note = "" if r == real else " (this run predates real returns)"
    st.caption(f"{c.ruin_count:,} ruined paths · earliest ruin at age "
               f"{ui.age(c.ruin_month_min, ret_age)} (a single path; noisy)"
               f" · ES10 / ES5 ruin age "
               f"{ui.age(c.ruin_month_es10, ret_age)} / "
               f"{ui.age(c.ruin_month_es5, ret_age)} (among ruined paths "
               f"only)"
               f" · P5 return {ui.total_return(c.pct_return(5, r))}"
               f" · P25 return {ui.total_return(c.pct_return(25, r))}"
               f" · returns in {ui.dollars_label(r)}{note}")


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
st.plotly_chart(charts.survival_curves(
    curves, ret_age, theme,
    alive=lambda a: household.p_alive(a, ret_age)),
    theme="streamlit")
st.caption("Solvency by age, with P(alive) for comparison: a ruin counts "
           "toward lifetime ruin in proportion to the chance you're alive "
           "when it happens.")

st.subheader("NAV percentiles by age")
fan_models = [(decision.DECISION_MODEL, cell)]
if ref is not None:
    fan_models.append((f"{ref_model} (reference)", ref))
if cell.nav_bands is None:
    st.write("This cell's run predates per-year NAV percentiles; re-run "
             "the sweep to see them.")
else:
    log_scale = st.toggle(
        "Log scale", value=False,
        help="Shows the downside bands in more detail. Years where a "
             "percentile is $0 (ruined) drop off the chart.")
    tabs = st.tabs([name for name, _ in fan_models])
    for tab, (name, c) in zip(tabs, fan_models):
        with tab:
            r = real and c.real_nav_bands is not None
            bands = c.bands(r)
            if bands is None:
                st.write("Not available for this run.")
                continue
            if r != real:
                st.caption("This run predates real NAV percentiles; "
                           "showing nominal dollars.")
            st.plotly_chart(charts.fan_chart(
                bands, ret_age, review.profile.initial_nav, theme,
                log_scale, y_title=f"NAV ({ui.dollars_label(r)})"),
                theme="streamlit")
    st.caption(("Today's dollars: each path deflated by its own simulated "
                "inflation." if real else
                "Nominal dollars of each year: not adjusted for inflation.")
               + " Only the median and below are shown: upside is captured "
               "by recalibrating at the next review.")

st.subheader("Age at ruin")
if cell.ruin_count == 0:
    st.write("No path is ruined within the horizon.")
else:
    st.plotly_chart(charts.ruin_age_histogram(cell, ret_age,
                                              theme.series_1, theme),
                    theme="streamlit")
    st.caption(decision.DECISION_MODEL)
