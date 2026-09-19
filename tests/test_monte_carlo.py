import json
import os
from pathlib import Path

import numpy as np
import pytest

import monte_carlo as mc

GOLDEN = Path(__file__).parent / "golden" / "monte_carlo_agg.json"

SWEEP = dict(
    yearly_spending_floor=80_000, yearly_spending_ceil=90_000,
    spend_increments=10_000, equity_floor=0.5, equity_ceil=0.6,
    weight_increments=0.1, initial_nav=1_000_000, years_to_simulate=10,
    retirement_age=60, total_paths=20, workers=2,
    tax_regimes=["none", "current_law_indexed"], master_seed=123)
ALL_MODELS = [m.name() for m in mc.MonteCarloCLI.SUPPORTED_MODELS]


def run_cli(tmp_path, market_cache, tag="run", **overrides):
    config = {**SWEEP, "models": ALL_MODELS, **overrides}
    config_file = tmp_path / f"{tag}.json"
    config_file.write_text(json.dumps(config))
    cli = mc.MonteCarloCLI(str(config_file), str(market_cache))
    cli.run()
    cli.aggregate()
    return cli


@pytest.fixture(scope="module")
def cli(tmp_path_factory, market_cache):
    return run_cli(tmp_path_factory.mktemp("mc"), market_cache)


def comparable(raw):
    """Raw results minus wall-clock timings."""
    return {k: v for k, v in raw.items() if k != "perf_data"}


class TestMCConfig:
    def make(self, **kw):
        defaults = dict(
            yearly_spending_floor=100_000, yearly_spending_ceil=120_000,
            spend_increments=10_000, starting_equity=0.5, ending_equity=0.6,
            weight_increments=0.05, models=["A", "B"])
        return mc.MonteCarloCLI.MCConfig(**{**defaults, **kw})

    def test_total_portfolios(self):
        # 3 equity steps x 3 spending steps x 2 models x 2 regimes
        cfg = self.make(tax_regimes=["none", "current_law_indexed"])
        assert cfg.total_portfolios() == 36
        assert len(list(cfg.portfolio_configs())) == 36

    def test_tax_regimes_default_to_none(self):
        cfg = self.make()
        assert cfg.tax_regimes == ["none"]
        assert cfg.total_portfolios() == 18

    def test_allocations_sum_to_one_and_are_rounded(self):
        for p in self.make().portfolio_configs():
            assert p["equity_allocation"] + p["ladder_allocation"] == 1.0
        equities = {p["equity_allocation"]
                    for p in self.make().portfolio_configs()}
        assert equities == {0.5, 0.55, 0.6}

    def test_tax_regime_is_innermost_loop(self):
        # run() relies on regime variants of one cell being adjacent so they
        # can share a random seed.
        cfg = self.make(models=["A"], tax_regimes=["r1", "r2", "r3"])
        configs = list(cfg.portfolio_configs())
        for i in range(0, len(configs), 3):
            cell = configs[i:i + 3]
            assert [p["tax_regime"] for p in cell] == ["r1", "r2", "r3"]
            assert len({(p["equity_allocation"], p["yearly_spending"])
                        for p in cell}) == 1

    def test_template_is_not_shared_between_configs(self):
        first, second = list(self.make().portfolio_configs())[:2]
        first["yearly_spending"] = -1
        assert second["yearly_spending"] != -1
        assert mc.MonteCarloCLI.MCConfig.PORTFOLIO_CONFIG_TEMPLATE[
            "yearly_spending"] is None


class TestRun:
    def test_result_shape(self, cli):
        raw = cli.raw_results
        assert raw["total_paths"] == 20
        assert set(raw["simulations"]) == set(ALL_MODELS)
        for sims in raw["simulations"].values():
            assert len(sims) == 8  # 2 equity x 2 spending x 2 regimes
            for sim in sims:
                assert len(sim["results"]["Terminal NAV"]) == 20
                assert len(sim["results"]["Terminal SPX"]) == 20
                assert len(sim["ruin_histogram"]) == 120
                assert sum(sim["ruin_histogram"]) <= 20

    def test_ruin_histogram_matches_zero_navs(self, cli):
        for sims in cli.raw_results["simulations"].values():
            for sim in sims:
                zeros = sum(v == 0.0 for v in sim["results"]["Terminal NAV"])
                assert sum(sim["ruin_histogram"]) == zeros

    def test_same_seed_is_reproducible(self, cli, tmp_path, market_cache):
        again = run_cli(tmp_path, market_cache)
        assert comparable(again.raw_results) == comparable(cli.raw_results)

    def test_different_seed_changes_results(self, cli, tmp_path, market_cache):
        other = run_cli(tmp_path, market_cache, master_seed=124)
        assert comparable(other.raw_results) != comparable(cli.raw_results)

    def test_regimes_share_market_paths(self, cli):
        # Common random numbers: same cell => identical simulated markets, so
        # only the tax treatment differs.
        for sims in cli.raw_results["simulations"].values():
            for untaxed, taxed in zip(sims[0::2], sims[1::2]):
                assert untaxed["tax_regime"] == "none"
                assert taxed["tax_regime"] == "current_law_indexed"
                assert (untaxed["spending"], untaxed["equity"]) == (
                    taxed["spending"], taxed["equity"])
                assert (untaxed["results"]["Terminal SPX"]
                        == taxed["results"]["Terminal SPX"])

    def test_taxes_do_not_raise_average_terminal_wealth(self, cli):
        for sims in cli.raw_results["simulations"].values():
            for untaxed, taxed in zip(sims[0::2], sims[1::2]):
                assert (np.mean(taxed["results"]["Terminal NAV"])
                        <= np.mean(untaxed["results"]["Terminal NAV"]))

    def test_run_only_once(self, cli):
        with pytest.raises(RuntimeError, match="only be run once"):
            cli.run()

    def test_unknown_model_rejected(self, tmp_path, market_cache):
        with pytest.raises(ValueError, match="not supported"):
            run_cli(tmp_path, market_cache, models=["NoSuchModel"])


class TestAggregate:
    @staticmethod
    def build(tmp_path, market_cache, histogram, navs):
        cli = mc.MonteCarloCLI("unused.json", str(market_cache))
        cli.simulation_config = mc.MonteCarloCLI.MCConfig(
            models=["M"], retirement_age=60)
        cli.raw_results = {
            "initial_nav": 1_000_000, "years": 10, "total_paths": len(navs),
            "simulations": {"M": [{
                "spending": 50_000, "equity": 0.6, "ladder": 0.4,
                "tax_regime": "none", "ruin_histogram": histogram,
                "results": {"Terminal SPX": [1.0] * len(navs),
                            "Terminal NAV": navs}}]},
        }
        cli.aggregate()
        return cli.agg_results["results"]["M"][0]

    def test_ruin_statistics(self, tmp_path, market_cache):
        histogram = [0.0] * 12
        for month in (1, 3, 5, 7, 9):
            histogram[month] = 1.0
        entry = self.build(tmp_path, market_cache, histogram,
                           [0.0] * 5 + [2_000_000.0] * 5)
        assert entry["ruin_path_count"] == 5
        assert entry["ruin_month_min"] == 1
        assert entry["ruin_month_median"] == 5.0
        # 5th/10th percentiles of [1,3,5,7,9] are 1.4 / 1.8: only month 1
        # falls in either tail.
        assert entry["ruin_month_es5"] == 1.0
        assert entry["ruin_month_es10"] == 1.0
        assert entry["allocation"] == "60-40"

    def test_no_ruin_reports_nones(self, tmp_path, market_cache):
        entry = self.build(tmp_path, market_cache, [0.0] * 12,
                           [1_000_000.0] * 4)
        assert entry["ruin_path_count"] == 0
        for key in ("ruin_month_min", "ruin_month_median",
                    "ruin_month_es5", "ruin_month_es10"):
            assert entry[key] is None

    def test_return_quantiles(self, tmp_path, market_cache):
        navs = [0.0, 500_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
        entry = self.build(tmp_path, market_cache, [0.0] * 12, navs)
        assert entry["p50_return"] == pytest.approx(0.0)
        assert entry["p25_return"] == pytest.approx(-0.5)
        assert entry["p10_return"] == pytest.approx(-0.8)
        assert entry["p5_return"] == pytest.approx(-0.9)

    def test_aggregate_only_once(self, cli):
        with pytest.raises(RuntimeError, match="only be run once"):
            cli.aggregate()


def test_golden_aggregates(cli):
    """
    Regression snapshot of the aggregated results for a fixed seed and
    synthetic market. A failure means simulation behaviour changed: if the
    change is intended, regenerate with `UPDATE_GOLDEN=1 pytest` and review
    the diff of tests/golden/monte_carlo_agg.json.
    """
    actual = json.loads(json.dumps(cli.agg_results))
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(actual, indent=2) + "\n")
    expected = json.loads(GOLDEN.read_text())
    assert_close(actual, expected)


def assert_close(actual, expected, path="agg"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), path
    else:
        assert actual == expected, path
