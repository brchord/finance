import pytest

from tax_models.regimes import (
    BASELINE_2026_SINGLE,
    PRE_TCJA_STYLE_SINGLE,
    BracketTable,
    CurrentLawIndexed,
    HistoricalAverageDrift,
    RegimeSwitchAtYear,
    build_tax_regime,
)

SIMPLE = BracketTable(brackets=((0.10, 0), (0.20, 10_000)),
                      standard_deduction=1_000)


class TestBracketTable:
    @pytest.mark.parametrize("income, expected", [
        (-500, 0.0),
        (0, 0.0),
        (5_000, 500.0),
        (10_000, 1_000.0),
        (15_000, 2_000.0),
    ])
    def test_tax_on(self, income, expected):
        assert SIMPLE.tax_on(income) == pytest.approx(expected)

    def test_scaled_scales_thresholds_and_deduction_not_rates(self):
        scaled = SIMPLE.scaled(2.0)
        assert scaled.brackets == ((0.10, 0), (0.20, 20_000))
        assert scaled.standard_deduction == 2_000

    def test_scaled_is_immutable_copy(self):
        SIMPLE.scaled(3.0)
        assert SIMPLE.brackets[1][1] == 10_000


class TestComputeTax:
    law = BASELINE_2026_SINGLE

    def test_ordinary_only(self):
        # taxable 33_900: 10% of 12_400 + 12% of 21_500
        assert self.law.compute_tax(50_000, 0) == pytest.approx(3_820.0)

    def test_unused_deduction_shelters_preferential_income(self):
        # 60_000 pref - 16_100 deduction = 43_900 < 0% LTCG ceiling
        assert self.law.compute_tax(0, 60_000) == pytest.approx(0.0)

    def test_preferential_income_stacks_on_ordinary(self):
        # ordinary: 1_240 + 12% of 31_500 = 5_020
        # pref stacks 43_900 -> 83_900, 15% above 49_450 = 5_167.5
        assert self.law.compute_tax(60_000, 40_000) == pytest.approx(10_187.5)

    def test_niit_applies_above_threshold(self):
        # ltcg: 15% of (283_900 - 49_450); niit: 3.8% of 100_000
        assert self.law.compute_tax(0, 300_000) == pytest.approx(38_967.5)

    def test_no_income_no_tax(self):
        assert self.law.compute_tax(0, 0) == 0.0


class TestScenarios:
    def test_current_law_indexes_brackets_but_not_niit(self):
        law = CurrentLawIndexed(
            "x", BASELINE_2026_SINGLE).resolve(5, 2.0)
        assert law.ordinary.standard_deduction == pytest.approx(32_200)
        assert law.niit.threshold_single == 200_000

    def test_historical_drift_endpoints_and_hold(self):
        s = HistoricalAverageDrift("x", BASELINE_2026_SINGLE, 0.396, 15)
        assert s.resolve(0, 1.0).ordinary.brackets[-1][0] == pytest.approx(0.37)
        assert s.resolve(15, 1.0).ordinary.brackets[-1][0] == pytest.approx(
            0.396)
        assert s.resolve(40, 1.0).ordinary.brackets[-1][0] == pytest.approx(
            0.396)

    def test_historical_drift_midpoint(self):
        s = HistoricalAverageDrift("x", BASELINE_2026_SINGLE, 0.396, 10)
        assert s.resolve(5, 1.0).ordinary.brackets[-1][0] == pytest.approx(
            0.383)

    def test_regime_switch(self):
        s = RegimeSwitchAtYear("x", BASELINE_2026_SINGLE, 10,
                               PRE_TCJA_STYLE_SINGLE)
        assert s.resolve(9, 1.0).ordinary.standard_deduction == 16_100
        assert s.resolve(10, 1.0).ordinary.standard_deduction == 8_500
        assert s.resolve(10, 2.0).ordinary.standard_deduction == 17_000


class TestBuildTaxRegime:
    @pytest.mark.parametrize("name", [None, "none"])
    def test_untaxed(self, name):
        assert build_tax_regime(name) is None

    @pytest.mark.parametrize("name", [
        "current_law_indexed", "historical_average_drift",
        "pre_tcja_reversion"])
    def test_named_regimes(self, name):
        assert build_tax_regime(name).name == name

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tax_regime"):
            build_tax_regime("bogus")
