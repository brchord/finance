"""
Explorer: run sweeps for the selected review and find the maximum
sustainable spending and the best allocation at it.
"""

import datetime as dt

import pandas as pd
import streamlit as st

import job_files
from planner import charts, decision, store, ui

SWEEP_KEYS = ("sp_floor", "sp_ceil", "sp_step", "eq_floor", "eq_ceil",
              "eq_step", "paths")
PENDING_SWEEP = "_pending_sweep"
SWEEP_REVIEW = "_sweep_review_id"
ACTIVE_RUNS = "_active_run_ids"


def sweep_defaults(review: store.Review) -> dict:
    "Prefill: the review's latest run, else the latest run of any review."
    for r in [review] + [r for r in store.list_reviews() if r != review]:
        runs = store.list_runs(r)
        if runs:
            sw = runs[0].sweep
            return dict(sp_floor=sw.spending_floor, sp_ceil=sw.spending_ceil,
                        sp_step=sw.spending_step,
                        eq_floor=round(sw.equity_floor * 100),
                        eq_ceil=round(sw.equity_ceil * 100),
                        eq_step=round(sw.equity_step * 100),
                        paths=sw.total_paths)
    return dict(sp_floor=80_000.0, sp_ceil=120_000.0, sp_step=5_000.0,
                eq_floor=30, eq_ceil=80, eq_step=10, paths=50_000)


def init_sweep_state(review: store.Review):
    s = st.session_state
    if PENDING_SWEEP in s:
        s.update(s.pop(PENDING_SWEEP))
    elif s.get(SWEEP_REVIEW) != review.id or any(
            k not in s for k in SWEEP_KEYS):
        s.update(sweep_defaults(review))
    s[SWEEP_REVIEW] = review.id


def steps(low: float, high: float, step: float) -> int:
    return int(round((high - low) / step)) + 1 if step > 0 else 0


def seconds_per_cell_path(review: store.Review):
    "Throughput of the review's latest succeeded run, for an ETA."
    for run in store.list_runs(review):
        if run.state == job_files.SUCCEEDED and run.status:
            start = dt.datetime.fromisoformat(run.status["started_at"])
            end = dt.datetime.fromisoformat(run.status["updated_at"])
            work = run.status["cells_total"] * run.config["total_paths"]
            if work:
                return (end - start).total_seconds() / work
    return None


def sweep_form(review: store.Review):
    s = st.session_state
    st.markdown("**Spending ($/year)**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("From", min_value=0.0, step=1_000.0, format="%.0f",
                    key="sp_floor")
    c2.number_input("To", min_value=0.0, step=1_000.0, format="%.0f",
                    key="sp_ceil")
    c3.number_input("Step", min_value=500.0, step=500.0, format="%.0f",
                    key="sp_step")
    st.markdown("**Equity allocation (%)**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("From", min_value=0, max_value=100, step=5,
                    key="eq_floor")
    c2.number_input("To", min_value=0, max_value=100, step=5, key="eq_ceil")
    c3.number_input("Step", min_value=1, max_value=100, step=1,
                    key="eq_step")
    st.number_input("Paths per cell", min_value=100, step=10_000,
                    key="paths")

    if s.sp_ceil < s.sp_floor or s.eq_ceil < s.eq_floor:
        st.error("Each range's 'To' must be at least its 'From'.")
        return
    models = (decision.DECISION_MODEL, ui.reference_model())
    n_cells = (steps(s.sp_floor, s.sp_ceil, s.sp_step)
               * steps(s.eq_floor, s.eq_ceil, s.eq_step) * len(models))
    eta = seconds_per_cell_path(review)
    caption = f"{n_cells} cells ({len(models)} models)"
    if eta is not None:
        caption += f" · roughly {eta * n_cells * s.paths:,.0f} s"
    st.caption(caption)

    if st.button("Run", type="primary"):
        sweep = store.Sweep(
            spending_floor=s.sp_floor, spending_ceil=s.sp_ceil,
            spending_step=s.sp_step, equity_floor=s.eq_floor / 100,
            equity_ceil=s.eq_ceil / 100, equity_step=s.eq_step / 100,
            total_paths=int(s.paths), models=models)
        run = store.launch(review, sweep)
        if run.state == job_files.SUCCEEDED:
            st.toast("This sweep already ran in this review; its results "
                     "are already included.")
        else:
            st.rerun()


@st.fragment(run_every=2)
def active_runs(review: store.Review):
    s = st.session_state
    runs = [r for r in store.list_runs(review) if r.active]
    ids = {r.id for r in runs}
    finished = s.get(ACTIVE_RUNS, set()) - ids
    s[ACTIVE_RUNS] = ids
    if finished:
        st.rerun(scope="app")
    for run in runs:
        sw = run.sweep
        c1, c2 = st.columns([5, 1])
        label = (f"{ui.md_money(sw.spending_floor)}–{ui.md_money(sw.spending_ceil)}"
                 f" · {sw.equity_floor:.0%}–{sw.equity_ceil:.0%} equity · "
                 f"{sw.total_paths:,} paths")
        if run.status:
            label += (f" — {run.status['cells_done']}/"
                      f"{run.status['cells_total'] or '?'} cells")
        c1.progress(run.progress, text=label)
        if c2.button("Cancel", key=f"cancel_{run.id}",
                     disabled=run.status is None):
            store.cancel(run)
            st.rerun(scope="app")


def open_ended(f: decision.Frontier) -> str:
    "Max spending, with a '+' when no simulated level above it fails."
    if f.best_spending is None:
        return "none"
    return ui.money(f.best_spending) + (
        "+" if f.next_failing_spending is None else "")


def headline(f: decision.Frontier, ref_f: decision.Frontier,
             crit: decision.Criteria, review: store.Review):
    c1, c2, c3 = st.columns([2, 2, 1.4])
    if f.best_spending is None:
        c1.metric("Max sustainable spending", "none")
        c1.caption(f"No simulated spending level passes the "
                   f"{crit.ruin_ceiling:.0%} ceiling; refine downward.")
    else:
        best = f.best_cell
        c1.metric("Max sustainable spending", open_ended(f),
                  help="Highest simulated spending level where some "
                       "allocation passes the ruin ceiling.")
        if f.next_failing_spending is None:
            c1.caption("Every simulated level passes: extend the sweep "
                       "upward.")
        else:
            c1.caption(f"{ui.md_money(f.next_failing_spending)} fails · "
                       f"crossing estimated at {ui.md_money(f.estimate)}")
        c2.metric(f"Best allocation at {ui.md_money(f.best_spending)}",
                  best.cell.allocation if best else "—")
        if best:
            c2.caption(
                f"ruin {ui.pct(best.cell.ruin_rate)} · ES10 age "
                f"{ui.age(best.cell.ruin_month_es10, review.profile.retirement_age)}"
                f" · {best.reason}")
    c3.metric("Reference model", open_ended(ref_f),
              help=f"Max sustainable spending under {ui.reference_model()},"
                   " for context only.")
    if f.near_ceiling:
        st.warning("The deciding cell's ruin rate is within two standard "
                   "errors of the ceiling: it could pass or fail on a "
                   "re-run. More paths would settle it.", icon="⚠️")
    if f.non_monotonic:
        st.info("A lower spending level fails while a higher one passes: "
                "the frontier is inside the simulation noise here.",
                icon="ℹ️")


# Status palette (dataviz reference): good / critical. The marks carry the
# status in color; the row tints are faint enough for normal text ink to
# stay readable on both light and dark surfaces.
PASS_COLOR, FAIL_COLOR = "#0ca30c", "#d03b3b"
PASS_TINT = "background-color: rgba(12, 163, 12, 0.10)"
FAIL_TINT = "background-color: rgba(208, 59, 59, 0.10)"


def ranked_table(level_cells, ref_by_key, crit, review):
    ret_age = review.profile.retirement_age
    ranked = decision.rank(level_cells, crit)
    rows = []
    for r in ranked:
        c = r.cell
        ref = ref_by_key.get((c.spending, c.equity))
        rows.append({
            "Rank": r.rank,
            "Allocation": c.allocation,
            "Passes": "✓" if r.passes else "✗",
            "P(ruin)": c.ruin_rate * 100,
            # Half-width of the 95% confidence interval of P(ruin).
            "±95%": 1.96 * c.ruin_rate_se * 100,
            "Paths": c.total_paths,
            "ES10 age": decision.month_to_age(c.ruin_month_es10, ret_age),
            "P10 return": c.p10_return * 100,
            "P25 return": c.p25_return * 100,
            "P50 return": c.p50_return * 100,
            "Ref. P(ruin)": ref.ruin_rate * 100 if ref else None,
            "Decided by": r.decided_by,
        })
    df = pd.DataFrame(rows)
    passes = [r.passes for r in ranked]
    styled = (df.style
              .apply(lambda row: [PASS_TINT if passes[row.name]
                                  else FAIL_TINT] * len(row), axis=1)
              .map(lambda v: f"color: {PASS_COLOR if v == '✓' else FAIL_COLOR};"
                             " font-weight: bold", subset=["Passes"]))
    event = st.dataframe(
        styled, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="rank_table",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Passes": st.column_config.TextColumn(width="small"),
            "P(ruin)": st.column_config.NumberColumn(format="%.2f%%"),
            "±95%": st.column_config.NumberColumn(
                format="±%.2f%%",
                help="95% confidence interval of P(ruin), from the number "
                     "of paths: the true ruin rate is likely within this "
                     "margin of the estimate."),
            "Paths": st.column_config.NumberColumn(
                format="localized",
                help="Paths simulated for this cell. Cells merged from "
                     "different runs can differ."),
            "Ref. P(ruin)": st.column_config.NumberColumn(
                format="%.2f%%", help=ui.reference_model()),
            "ES10 age": st.column_config.NumberColumn(
                format="%.1f",
                help="Mean age at ruin of the earliest 10% of ruined paths."
                     " Empty when no path is ruined."),
            "P10 return": st.column_config.NumberColumn(format="%+.0f%%"),
            "P25 return": st.column_config.NumberColumn(format="%+.0f%%"),
            "P50 return": st.column_config.NumberColumn(format="%+.0f%%"),
            "Decided by": st.column_config.TextColumn(
                help="The ranking step that put this allocation ahead of "
                     "the ones below it. Steps run in order: ruin rate, "
                     "ES10 age, P10 return, P50 return; every step before "
                     "the deciding one was a tie within tolerance."),
        })
    selected = event.selection.rows
    return ranked[selected[0]].cell if selected else None


def refinement(cells, crit, review):
    runs = [r for r in store.list_runs(review)
            if r.state == job_files.SUCCEEDED]
    if not runs:
        return
    sp_step = min(r.sweep.spending_step for r in runs)
    eq_step = min(r.sweep.equity_step for r in runs)
    proposal = decision.propose_refinement(cells, crit, sp_step, eq_step)
    with st.expander("Refine around the frontier", expanded=False):
        if proposal is None:
            st.write("The frontier is already bracketed at the finest "
                     "spending resolution ($500).")
            return
        st.write(ui.esc(proposal.rationale))
        st.write(
            f"Spending {ui.md_money(proposal.spending_floor)}–"
            f"{ui.md_money(proposal.spending_ceil)} by "
            f"{ui.md_money(proposal.spending_step)}; equity "
            f"{proposal.equity_floor:.0%}–{proposal.equity_ceil:.0%} by "
            f"{proposal.equity_step:.0%}.")
        if st.button("Load into the sweep form"):
            st.session_state[PENDING_SWEEP] = dict(
                sp_floor=float(proposal.spending_floor),
                sp_ceil=float(proposal.spending_ceil),
                sp_step=float(proposal.spending_step),
                eq_floor=round(proposal.equity_floor * 100),
                eq_ceil=round(proposal.equity_ceil * 100),
                eq_step=round(proposal.equity_step * 100))
            st.rerun()


def runs_list(review: store.Review):
    runs = store.list_runs(review)
    with st.expander(f"Runs in this review ({len(runs)})"):
        if not runs:
            st.write("None yet.")
        for run in runs:
            sw = run.sweep
            st.markdown(
                f"**{run.state}** · started {run.started_at or '—'} · "
                f"{ui.md_money(sw.spending_floor)}–{ui.md_money(sw.spending_ceil)}"
                f" by {ui.md_money(sw.spending_step)} · "
                f"{sw.equity_floor:.0%}–{sw.equity_ceil:.0%} by "
                f"{sw.equity_step:.0%} · {sw.total_paths:,} paths · "
                f"`{run.id}`")
            if run.error:
                st.error(run.error)
                if run.log_path.exists():
                    st.code("\n".join(
                        run.log_path.read_text().splitlines()[-15:]))


# --------------------------------------------------------------------------

review = ui.selected_review()
st.title("Explorer")
if review is None:
    st.info("Create a review to get started: it holds the baseline NAV and "
            "horizon that its runs share.")
    if st.button("New review", type="primary"):
        ui.new_review_dialog(None)
    st.stop()

st.caption(f"**{review.label}** · {ui.profile_caption(review.profile)}")
init_sweep_state(review)
merged = ui.merged_cells(review)
has_results = bool(merged)

with st.expander("New sweep", expanded=not has_results):
    sweep_form(review)
if any(r.active for r in store.list_runs(review)):
    active_runs(review)
else:
    st.session_state[ACTIVE_RUNS] = set()

if not has_results:
    st.stop()

crit = ui.criteria()
tax_regime = review.profile.tax_regime
cells = ui.model_cells(merged, decision.DECISION_MODEL, tax_regime)
ref_cells = ui.model_cells(merged, ui.reference_model(), tax_regime)
if not cells:
    st.warning(f"No results for {decision.DECISION_MODEL} in this review.")
    st.stop()

f = decision.frontier(cells, crit)
ref_f = decision.frontier(ref_cells, crit)
headline(f, ref_f, crit, review)

theme = ui.theme()
show_ref = st.toggle("Show the reference model", value=False,
                     disabled=not ref_cells)
y_max = max(c.ruin_rate for c in cells + (ref_cells if show_ref else []))
# Capped at 3x the ceiling: the decision happens near it, and high-spending
# levels far above it would otherwise squash that region.
y_max = min(max(y_max, crit.ruin_ceiling), 3 * crit.ruin_ceiling) * 1.08
equities = [c.equity for c in merged.values()]
equity_range = (min(equities), max(equities))
fig = charts.ruin_vs_spending(
    cells, crit.ruin_ceiling, theme, best_spending=f.best_spending,
    y_max=y_max, equity_range=equity_range)
if show_ref and ref_cells:
    c1, c2 = st.columns(2)
    c1.markdown(f"**P(ruin) · {decision.DECISION_MODEL}**")
    c1.plotly_chart(fig, theme="streamlit")
    c2.markdown(f"**P(ruin) · {ui.reference_model()}** (reference)")
    c2.plotly_chart(charts.ruin_vs_spending(
        ref_cells, crit.ruin_ceiling, theme, y_max=y_max,
        equity_range=equity_range), theme="streamlit")
else:
    st.markdown(f"**P(ruin) · {decision.DECISION_MODEL}**")
    st.plotly_chart(fig, theme="streamlit")

levels = sorted({c.spending for c in cells})
default_level = f.best_spending if f.best_spending is not None else levels[0]
level = st.selectbox("Rank allocations at spending", levels,
                     index=levels.index(default_level),
                     format_func=ui.money, key=f"level_{review.id}")
level_cells = [c for c in cells if c.spending == level]
ref_by_key = {(c.spending, c.equity): c for c in ref_cells}
chosen = ranked_table(level_cells, ref_by_key, crit, review)
if chosen is not None:
    if st.button(f"Open {chosen.allocation} at {ui.md_money(chosen.spending)} "
                 "in Cell detail"):
        st.session_state["cell"] = (chosen.spending, chosen.equity)
        st.switch_page("views/cell.py")
else:
    st.caption("Select a row to open it in Cell detail.")

refinement(cells, crit, review)
runs_list(review)
