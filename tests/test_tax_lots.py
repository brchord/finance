import pytest

from portfolio_models.linear_models import TaxLotTracker


def test_fifo_split_between_short_and_long_term():
    t = TaxLotTracker()
    t.buy(10, 100.0, month=0)
    t.buy(10, 110.0, month=6)
    st, lt = t.sell(15, 120.0, month=12)
    assert lt == pytest.approx(200.0)   # 10 shares held 12 months
    assert st == pytest.approx(50.0)    # 5 shares held 6 months
    assert t.total_shares_held() == pytest.approx(5.0)
    assert t.average_cost_basis() == pytest.approx(110.0)
    assert t.realized_long_term_gain == pytest.approx(200.0)
    assert t.realized_short_term_gain == pytest.approx(50.0)


def test_eleven_months_is_short_term():
    t = TaxLotTracker()
    t.buy(1, 100.0, month=0)
    st, lt = t.sell(1, 150.0, month=11)
    assert (st, lt) == (pytest.approx(50.0), 0.0)


def test_losses_are_negative_gains():
    t = TaxLotTracker()
    t.buy(2, 100.0, month=0)
    st, _ = t.sell(2, 90.0, month=1)
    assert st == pytest.approx(-20.0)


def test_non_positive_buy_is_noop():
    t = TaxLotTracker()
    t.buy(0, 100.0, month=0)
    t.buy(-1, 100.0, month=0)
    assert t.total_shares_held() == 0.0
    assert t.average_cost_basis() == 0.0


def test_overselling_raises():
    t = TaxLotTracker()
    t.buy(1, 100.0, month=0)
    with pytest.raises(RuntimeError):
        t.sell(2, 100.0, month=1)
