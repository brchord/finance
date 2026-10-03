import pytest

pytest.importorskip("plotly")
from planner import charts, decision  # noqa: E402


def cell_with_histogram(histogram):
    return decision.Cell(
        model=decision.DECISION_MODEL, spending=90_000.0, equity=0.6,
        tax_regime=decision.DEFAULT_TAX_REGIME, total_paths=1_000,
        ruin_count=sum(histogram), ruin_month_min=None, ruin_month_es5=None,
        ruin_month_es10=None, p5_return=0.0, p10_return=0.0,
        p25_return=0.0, p50_return=0.0, ruin_histogram=tuple(histogram))


def test_ruin_histogram_skips_ruin_free_years():
    # 5 years; ruin only in years 2 and 4 (ages 62 and 64).
    hist = [0] * 60
    hist[24] = 3
    hist[50] = 1
    fig = charts.ruin_age_histogram(cell_with_histogram(hist), 60,
                                    "#000000", charts.LIGHT)
    bar = fig.data[0]
    # Leading/trailing ruin-free years dropped; the gap year is kept so the
    # age axis stays continuous.
    assert list(bar.x) == [62, 63, 64]
    assert list(bar.y) == [3, 0, 1]


def test_equity_color_spans_the_given_range():
    low = charts.equity_color(0.3, charts.LIGHT, (0.3, 0.8))
    high = charts.equity_color(0.8, charts.LIGHT, (0.3, 0.8))
    assert (low, high) == (charts.LIGHT.ramp[0], charts.LIGHT.ramp[-1])
