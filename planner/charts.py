"""
planner/charts.py

Plotly figures for the UI. Colors follow the dataviz reference palette:
categorical slots 1/2 (blue/orange) for decision vs reference model, and a
one-hue blue ordinal ramp for equity allocations (lighter = less equity),
with separately chosen steps for dark mode.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from planner import decision


@dataclass(frozen=True)
class Theme:
    series_1: str
    series_2: str
    ramp: Sequence[str]      # ordinal, low -> high equity
    muted: str
    grid: str
    # Fan chart fills, outer (P5-P10) to inner (P25-P50), and median line.
    bands: Sequence[str] = ()
    band_line: str = ""


LIGHT = Theme(
    series_1="#2a78d6", series_2="#eb6834",
    # Blue steps 250 -> 700: the light end keeps >= 2:1 on a white surface.
    ramp=("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
          "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    muted="#52514e", grid="#e5e4e0",
    bands=("#cde2fb", "#9ec5f4", "#6da7ec"), band_line="#184f95")
DARK = Theme(
    series_1="#3987e5", series_2="#d95926",
    # Blue steps 600 -> 150: the dark end keeps >= 2:1 on a dark surface.
    ramp=("#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
          "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"),
    muted="#c3c2b7", grid="#383835",
    bands=("#0d366b", "#184f95", "#256abf"), band_line="#9ec5f4")


def theme_for(dark: bool) -> Theme:
    return DARK if dark else LIGHT


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}"
                         for x, y in zip(ca, cb))


def equity_color(equity: float, theme: Theme,
                 equity_range: Tuple[float, float] = (0.0, 1.0)) -> str:
    """
    Position of the equity weight within equity_range on the ramp. Pass a
    range that is stable for the data at hand (e.g. every allocation in
    the review), not just the allocations currently shown, so an
    allocation keeps its color when the view changes.
    """
    low, high = equity_range
    t = (equity - low) / (high - low) if high > low else 0.5
    pos = min(max(t, 0.0), 1.0) * (len(theme.ramp) - 1)
    i = min(int(pos), len(theme.ramp) - 2)
    return _mix(theme.ramp[i], theme.ramp[i + 1], pos - i)


def _base_layout(fig: go.Figure, theme: Theme, height: int = 380):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, linecolor=theme.grid)
    fig.update_yaxes(gridcolor=theme.grid, gridwidth=1, zeroline=False)


def _ceiling_line(fig: go.Figure, y: float, label: str, theme: Theme,
                  **kwargs):
    fig.add_hline(y=y, line=dict(color=theme.muted, width=1, dash="dash"),
                  annotation_text=label, annotation_position="top left",
                  annotation_font_color=theme.muted, **kwargs)


def ruin_vs_spending(cells: Sequence[decision.Cell], ceiling: float,
                     theme: Theme,
                     best_spending: Optional[float] = None,
                     y_max: Optional[float] = None,
                     equity_range: Tuple[float, float] = (0.0, 1.0)
                     ) -> go.Figure:
    "P(ruin) against spending, one line per equity allocation."
    by_equity: Dict[float, List[decision.Cell]] = {}
    for c in cells:
        by_equity.setdefault(c.equity, []).append(c)

    fig = go.Figure()
    for equity in sorted(by_equity):
        line = sorted(by_equity[equity], key=lambda c: c.spending)
        color = equity_color(equity, theme, equity_range)
        fig.add_trace(go.Scatter(
            x=[c.spending for c in line],
            y=[c.ruin_rate for c in line],
            name=line[0].allocation,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
            customdata=[[c.ruin_count, c.total_paths] for c in line],
            hovertemplate=("%{y:.2%} (%{customdata[0]:,} of "
                           "%{customdata[1]:,} paths)")))
    _base_layout(fig, theme)
    _ceiling_line(fig, ceiling, f"{ceiling:.0%} ceiling", theme)
    if best_spending is not None:
        fig.add_vline(x=best_spending,
                      line=dict(color=theme.muted, width=1))
    fig.update_layout(legend_title_text="Equity/FI  ")
    fig.update_xaxes(title_text="Yearly spending", tickprefix="$",
                     tickformat=",.0f")
    fig.update_yaxes(title_text="P(ruin)", tickformat=".0%",
                     rangemode="tozero",
                     range=[0, y_max] if y_max is not None else None)
    return fig


def survival_curves(curves: Sequence[tuple], retirement_age: float,
                    ceiling: float, theme: Theme) -> go.Figure:
    """
    P(solvent) by age. curves: (name, cell, color) tuples, drawn in order.
    """
    fig = go.Figure()
    for name, cell, color in curves:
        survival = cell.survival()
        fig.add_trace(go.Scatter(
            x=[retirement_age + (m + 1) / 12 for m in range(len(survival))],
            y=survival, name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="%{y:.2%}"))
    _base_layout(fig, theme)
    _ceiling_line(fig, 1 - ceiling, f"{1 - ceiling:.0%} solvent", theme)
    fig.update_xaxes(title_text="Age", hoverformat=".1f")
    fig.update_yaxes(title_text="P(still solvent)", tickformat=".0%")
    return fig


def ruin_age_histogram(cell: decision.Cell, retirement_age: float,
                       color: str, theme: Theme) -> go.Figure:
    """
    Ruined paths per year of age, from the first year with a ruined path
    to the last: the ruin-free years before (usually most of the horizon)
    would only squash the interesting part against one edge.
    """
    hist = cell.ruin_histogram
    years = (len(hist) + 11) // 12
    counts = [sum(hist[y * 12:(y + 1) * 12]) for y in range(years)]
    ages = [retirement_age + y for y in range(years)]
    nonzero = [y for y, n in enumerate(counts) if n > 0]
    if nonzero:
        first, last = nonzero[0], nonzero[-1] + 1
        counts, ages = counts[first:last], ages[first:last]
    fig = go.Figure(go.Bar(
        x=ages, y=counts, marker=dict(color=color, cornerradius=4),
        hovertemplate="Age %{x}: %{y:,} paths<extra></extra>"))
    _base_layout(fig, theme, height=300)
    fig.update_layout(bargap=0.15, hovermode="closest")
    fig.update_xaxes(title_text="Age at ruin")
    fig.update_yaxes(title_text="Ruined paths", tickformat=",d")
    return fig


def spending_timeline(dates: Sequence[str], values: Sequence[Optional[float]],
                      labels: Sequence[str], theme: Theme) -> go.Figure:
    "Max sustainable spending per review."
    fig = go.Figure(go.Scatter(
        x=list(dates), y=list(values), mode="lines+markers",
        text=list(labels), line=dict(color=theme.series_1, width=2),
        marker=dict(color=theme.series_1, size=9),
        hovertemplate="%{text}: $%{y:,.0f}<extra></extra>"))
    _base_layout(fig, theme, height=320)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Review date")
    fig.update_yaxes(title_text="Max sustainable spending", tickprefix="$",
                     tickformat=",.0f")
    return fig


def fan_chart(bands: dict, retirement_age: float, initial_nav: float,
              theme: Theme, log_scale: bool = False) -> go.Figure:
    """
    Per-year NAV percentile bands (MonteCarloCLI aggregated "nav_bands"):
    nested P5-P10, P10-P25 and P25-P50 fills and the median line.
    """
    ages = [retirement_age + y for y in bands["years"]]
    fig = go.Figure()
    layers = [("p5", "p10", "P5–P10"), ("p10", "p25", "P10–P25"),
              ("p25", "p50", "P25–P50")]
    for (low, high, name), fill in zip(layers, theme.bands):
        fig.add_trace(go.Scatter(
            x=ages, y=bands[low], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=ages, y=bands[high], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill, name=name, hoverinfo="skip"))
    for key, label in (("p5", "P5"), ("p10", "P10"), ("p25", "P25")):
        # Invisible traces so the unified hover lists every percentile.
        fig.add_trace(go.Scatter(
            x=ages, y=bands[key], mode="lines", line=dict(width=0),
            showlegend=False, name=label, hovertemplate="$%{y:,.0f}"))
    fig.add_trace(go.Scatter(
        x=ages, y=bands["p50"], mode="lines", name="Median",
        line=dict(color=theme.band_line, width=2),
        hovertemplate="$%{y:,.0f}"))
    _base_layout(fig, theme)
    fig.add_hline(y=initial_nav, line=dict(color=theme.muted, width=1),
                  annotation_text="starting NAV",
                  annotation_position="bottom right",
                  annotation_font_color=theme.muted)
    fig.update_xaxes(title_text="Age")
    fig.update_yaxes(title_text="NAV (real $)", tickprefix="$",
                     tickformat="~s", type="log" if log_scale else "linear",
                     # Linear starts at $0 so the distance to ruin shows.
                     rangemode="normal" if log_scale else "tozero")
    return fig
