"""
planner/ui.py

Streamlit plumbing shared by the pages: the sidebar (review selector,
decision controls), cached loading of results, and formatting.
"""

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

import job_files
from planner import charts, decision, store

PENDING_REVIEW = "_pending_review_id"
REAL, NOMINAL = "Today's $", "Nominal $"
DEFAULT_CEILING_PCT = decision.Criteria().ruin_ceiling * 100
DEFAULT_TIE_PCT = decision.Criteria().tie_share * 100


def theme() -> charts.Theme:
    try:
        dark = st.context.theme.type == "dark"
    except AttributeError:
        dark = False
    return charts.theme_for(dark)


# --------------------------------------------------------------------------
# Loading


@st.cache_data(show_spinner=False, max_entries=256)
def _run_cells(results_path: str, mtime: float, run_id: str,
               finished_at: str) -> List[decision.Cell]:
    # mtime is only part of the cache key: a rewritten file is reloaded.
    results = job_files.read_json(Path(results_path))
    if results is None:
        return []
    return decision.cells_from_results(results, run_id, finished_at)


def merged_cells(review: store.Review) -> Dict[decision.CellKey,
                                               decision.Cell]:
    "Cells of all succeeded runs of the review, merged."
    cells: List[decision.Cell] = []
    for run in store.list_runs(review):
        if run.state == job_files.SUCCEEDED and run.results_path.exists():
            cells.extend(_run_cells(
                str(run.results_path), run.results_path.stat().st_mtime,
                run.id, run.finished_at or ""))
    return decision.merge_cells(cells)


def model_cells(merged: Dict[decision.CellKey, decision.Cell], model: str,
                tax_regime: str) -> List[decision.Cell]:
    return [c for c in merged.values()
            if c.model == model and c.tax_regime == tax_regime]


# --------------------------------------------------------------------------
# Sidebar


def criteria(review: store.Review) -> decision.Criteria:
    "The sidebar's decision settings, for review's profile and household."
    s = st.session_state
    return decision.Criteria(
        ruin_ceiling=s.get("ceiling_pct", DEFAULT_CEILING_PCT) / 100,
        tie_share=s.get("tie_pct", DEFAULT_TIE_PCT) / 100,
        p10_tolerance=s.get("p10_tol_pct", 10.0) / 100,
        retirement_age=review.profile.retirement_age,
        household=review.household)


def real_dollars() -> bool:
    "Whether returns and NAVs are shown in today's dollars."
    return st.session_state.get("dollars", REAL) == REAL


def reference_model() -> str:
    return st.session_state.get("reference_model",
                                decision.REFERENCE_MODELS[0])


def selected_review() -> Optional[store.Review]:
    review_id = st.session_state.get("review_id")
    for review in store.list_reviews():
        if review.id == review_id:
            return review
    return None


@st.dialog("New review")
def new_review_dialog(latest: Optional[store.Review]):
    st.caption("A review is a NAV snapshot. All of its runs share this "
               "profile; start a new review when the NAV changes.")
    p = latest.profile if latest else None
    with st.form("new_review"):
        label = st.text_input("Label", placeholder="e.g. Q4 2026")
        date = st.date_input("Review date", value=dt.date.today())
        nav = st.number_input(
            "Baseline NAV ($)", min_value=0.0, step=10_000.0, format="%.0f",
            value=float(p.initial_nav) if p else 1_000_000.0)
        retirement_age = st.number_input(
            "Retirement age", min_value=18.0, max_value=100.0, step=1.0,
            value=float(p.retirement_age) if p else 65.0)
        terminal_age = st.number_input(
            "Terminal age (end of the simulation)", min_value=19.0,
            max_value=120.0, step=1.0,
            value=float(p.terminal_age) if p else 100.0)
        with st.expander("Market assumptions"):
            st.caption("Used by the CAPE-drag models (the decision model "
                       "and HybridValuationVAR); see doc/assumptions.md. "
                       "Set the starting CAPE to the current value.")
            d = store.Profile(initial_nav=0, retirement_age=0,
                              years_to_simulate=0)
            a = p or d
            c1, c2 = st.columns(2)
            initial_cape = c1.number_input(
                "Starting CAPE", min_value=5.0, max_value=80.0, step=0.5,
                value=float(a.initial_cape),
                help=f"Shiller CAPE today. Engine default "
                     f"{d.initial_cape:g}.")
            target_cape = c2.number_input(
                "Long-run CAPE", min_value=5.0, max_value=80.0, step=0.5,
                value=float(a.target_cape),
                help="Where valuations revert to. ~16 full-history median, "
                     "~20 post-1950, ~25-27 post-1990.")
            c1, c2, c3 = st.columns(3)
            growth = c1.number_input(
                "Real earnings growth (%)", min_value=-5.0, max_value=10.0,
                step=0.1, value=a.annual_earnings_growth * 100,
                format="%.2f")
            buybacks = c2.number_input(
                "Net buyback yield (%)", min_value=0.0, max_value=10.0,
                step=0.1, value=a.annual_buyback_yield * 100,
                format="%.2f",
                help="Added to per-share price growth. 0 reproduces the "
                     "original model; US net buybacks have run ~1-2%.")
            dividends = c3.number_input(
                "Dividend yield (%)", min_value=0.0, max_value=10.0,
                step=0.1, value=a.dividend_yield * 100, format="%.2f")
        with st.expander("Advanced"):
            tax_regime = st.text_input(
                "Tax regime", value=p.tax_regime if p else
                decision.DEFAULT_TAX_REGIME)
        submitted = st.form_submit_button("Create review", type="primary")
    if submitted:
        if terminal_age <= retirement_age:
            st.error("Terminal age must be after the retirement age.")
            return
        profile = store.Profile(
            initial_nav=nav, retirement_age=retirement_age,
            years_to_simulate=terminal_age - retirement_age,
            tax_regime=tax_regime, initial_cape=initial_cape,
            target_cape=target_cape, annual_earnings_growth=growth / 100,
            annual_buyback_yield=buybacks / 100,
            dividend_yield=dividends / 100)
        review = store.create_review(
            label or date.isoformat(), date, profile,
            household=latest.household if latest else decision.Household())
        st.session_state[PENDING_REVIEW] = review.id
        st.rerun()


def sidebar():
    """
    Renders the sidebar on every page. Widget state lives under the keys
    read by criteria(), reference_model() and selected_review().
    """
    s = st.session_state
    if PENDING_REVIEW in s:
        s["review_id"] = s.pop(PENDING_REVIEW)

    reviews = store.list_reviews()
    with st.sidebar:
        st.subheader("Review")
        if reviews:
            ids = [r.id for r in reviews]
            if s.get("review_id") not in ids:
                s["review_id"] = ids[0]
            labels = {r.id: f"{r.label} · {r.date}" for r in reviews}
            st.selectbox("Review", ids, key="review_id",
                         format_func=labels.__getitem__,
                         label_visibility="collapsed")
        else:
            st.caption("No reviews yet.")
        if st.button("New review", width="stretch"):
            new_review_dialog(reviews[0] if reviews else None)

        st.subheader("Display")
        st.radio(
            "Dollars", [REAL, NOMINAL], key="dollars", horizontal=True,
            help="Today's dollars deflate each path by its own simulated "
                 "inflation; nominal dollars are those of the year they "
                 "occur in. Over a long horizon nominal figures look far "
                 "larger. Results from runs before real figures existed "
                 "fall back to nominal.")

        st.subheader("Decision")
        st.number_input("Lifetime ruin ceiling (%)", min_value=0.1,
                        max_value=50.0, step=0.25,
                        value=DEFAULT_CEILING_PCT, key="ceiling_pct",
                        help="A cell passes if the decision model's "
                             "lifetime ruin probability (ruined while "
                             "still alive) is at or below this. It is "
                             "lower than ruin by the end of the horizon, "
                             "so a given ceiling is looser than the same "
                             "ceiling on that.")
        st.selectbox(
            "Reference model", decision.REFERENCE_MODELS,
            key="reference_model",
            help=f"Shown for context. Decisions always use "
                 f"{decision.DECISION_MODEL}. New runs simulate the "
                 f"selected reference model.")
        review = selected_review()
        if review is not None:
            with st.expander("Life expectancy"):
                household_form(review)
        with st.expander("Ranking tolerances"):
            st.number_input(
                "Ruin tie (% of the ceiling)", min_value=0.0, max_value=100.0,
                step=5.0, value=DEFAULT_TIE_PCT, key="tie_pct",
                help="Two allocations tie on lifetime ruin when they are "
                     "within this share of the ceiling (0.2pp at 20% of a "
                     "1% ceiling), and on years in ruin within that times "
                     f"{decision.YEARS_PER_RUIN:g} years. A difference "
                     "within the simulation's noise (2 standard errors) "
                     "is always a tie.")
            st.number_input("P10 tie (% of terminal wealth)",
                            min_value=0.0, step=1.0, value=10.0,
                            key="p10_tol_pct")


def household_form(review: store.Review):
    """
    Edits the selected review's household (saved to review.json). Keys
    are per review so switching reviews shows that review's values.
    """
    h = review.household
    default = decision.Life()
    k = review.id
    ret_age = review.profile.retirement_age
    st.caption(
        "Ruin counts by the chance someone is still alive then. Defaults "
        f"fit US male mortality ({default.modal_age:g} / "
        f"{default.dispersion:g}; see the design doc). Saved with the "
        "review; no re-run needed.")
    with st.form(f"household_{k}", border=False):
        couple = st.radio("Plan for", ["One person", "A couple"],
                          index=0 if h.partner is None else 1,
                          horizontal=True, key=f"hh_couple_{k}")
        c1, c2 = st.columns(2)
        modal = c1.number_input(
            "Most likely age at death", min_value=60.0, max_value=110.0,
            step=0.5, value=float(h.person.modal_age), key=f"hh_m_{k}",
            help="Gompertz modal age. Raise it for good health or long "
                 "family lifespans.")
        spread = c2.number_input(
            "Spread (years)", min_value=4.0, max_value=20.0, step=0.5,
            value=float(h.person.dispersion), key=f"hh_b_{k}",
            help="How spread out ages at death are around the modal age.")
        partner = h.partner or h.person
        st.markdown("**Partner** (used when planning for a couple)")
        c1, c2, c3 = st.columns(3)
        offset = c1.number_input(
            "Age difference", min_value=-40.0, max_value=40.0, step=1.0,
            value=float(h.partner_age_offset), key=f"hh_off_{k}",
            help="Partner's age minus yours.")
        p_modal = c2.number_input(
            "Modal age", min_value=60.0, max_value=110.0, step=0.5,
            value=float(partner.modal_age), key=f"hh_pm_{k}")
        p_spread = c3.number_input(
            "Spread", min_value=4.0, max_value=20.0, step=0.5,
            value=float(partner.dispersion), key=f"hh_pb_{k}")
        if st.form_submit_button("Save"):
            store.update_household(review, decision.Household(
                person=decision.Life(modal, spread),
                partner=(decision.Life(p_modal, p_spread)
                         if couple == "A couple" else None),
                partner_age_offset=offset))
            st.rerun()
    life = h.person.life_expectancy(ret_age)
    st.caption(f"Life expectancy from {ret_age:g}: {life:.1f} · alive at "
               f"85: {h.p_alive(85, ret_age):.0%} · at 95: "
               f"{h.p_alive(95, ret_age):.0%}"
               + (" (either partner)" if h.partner is not None else ""))


# --------------------------------------------------------------------------
# Formatting


def money(x: Optional[float]) -> str:
    return "—" if x is None else f"${x:,.0f}"


def esc(text: str) -> str:
    """
    Escapes dollar signs for Streamlit markdown (captions, labels, buttons,
    st.write), which otherwise reads "$...$" as LaTeX.
    """
    return text.replace("$", "\\$")


def md_money(x: Optional[float]) -> str:
    "money() for markdown contexts."
    return esc(money(x))


def pct(x: Optional[float], digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}%}"


def years(x: float) -> str:
    "Expected years in ruin, e.g. 0.08 yr."
    return f"{x:.2f} yr"


def age(month: Optional[float], retirement_age: float) -> str:
    a = decision.month_to_age(month, retirement_age)
    return "no ruin" if a is None else f"{a:.1f}"


def total_return(r: Optional[float]) -> str:
    "Total return over the horizon, e.g. +5% or +1,430%."
    return "—" if r is None else f"{r:+,.0%}"


def dollars_label(real: bool) -> str:
    return "today's $" if real else "nominal $"


def assumptions_caption(profile: store.Profile) -> str:
    return (f"CAPE {profile.initial_cape:g} → {profile.target_cape:g} · "
            f"earnings growth {profile.annual_earnings_growth:.1%} · "
            f"buybacks {profile.annual_buyback_yield:.1%} · "
            f"dividends {profile.dividend_yield:.1%}")


def profile_caption(profile: store.Profile) -> str:
    return (f"NAV {md_money(profile.initial_nav)} · retire at "
            f"{profile.retirement_age:g} · horizon to age "
            f"{profile.terminal_age:g} · tax regime {profile.tax_regime}"
            f" · {assumptions_caption(profile)}")
