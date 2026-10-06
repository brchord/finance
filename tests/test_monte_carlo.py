import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import job_files
import monte_carlo as mc

GOLDEN = Path(__file__).parent / "golden" / "monte_carlo_agg.json"
GOLDEN_ASSUMPTIONS = (Path(__file__).parent / "golden" /
                      "monte_carlo_agg_assumptions.json")

SWEEP = dict(
    yearly_spending_floor=80_000, yearly_spending_ceil=90_000,
    spend_increments=10_000, equity_floor=0.5, equity_ceil=0.6,
    weight_increments=0.1, initial_nav=1_000_000, years_to_simulate=10,
    retirement_age=60, total_paths=20, workers=2,
    tax_regimes=["none", "current_law_indexed"], master_seed=123)
ALL_MODELS = [m.name() for m in mc.MonteCarloCLI.SUPPORTED_MODELS]


def run_cli(tmp_path, market_cache, tag="run", backend="process",
            **overrides):
    config = {**SWEEP, "models": ALL_MODELS, **overrides}
    config_file = tmp_path / f"{tag}.json"
    config_file.write_text(json.dumps(config))
    cli = mc.MonteCarloCLI(str(config_file), str(market_cache))
    cli.run(backend=backend)
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

    @pytest.mark.parametrize("years,months", [
        (10, 120), (10.0, 120), (10.5, 126), (63, 756),
        (35.3, 424),  # 423.6 months rounds up, not truncated to 423
    ])
    def test_simulation_months_is_rounded_int(self, years, months):
        cfg = self.make(years_to_simulate=years)
        assert cfg.simulation_months == months
        assert type(cfg.simulation_months) is int
        assert cfg.years == years  # reported value is left as given

    def test_warns_only_when_years_are_not_whole_months(self, caplog):
        with caplog.at_level("WARNING"):
            self.make(years_to_simulate=10.5)
            self.make(years_to_simulate=10.0)
        assert not caplog.records
        with caplog.at_level("WARNING"):
            self.make(years_to_simulate=35.3)
        assert "not a whole number" in caplog.text

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

    def test_dividend_yield_reaches_every_portfolio(self):
        assert all(p["dividend_yield"] == 0.01
                   for p in self.make().portfolio_configs())
        cfg = self.make(dividend_yield=0.02)
        assert all(p["dividend_yield"] == 0.02
                   for p in cfg.portfolio_configs())
        assert cfg.total_portfolios() == self.make().total_portfolios()

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

    def test_nav_bands(self, cli):
        for sims in cli.raw_results["simulations"].values():
            for sim in sims:
                bands = sim["nav_bands"]
                assert bands["years"] == list(range(11))  # 10 years + start
                for key in ("p5", "p10", "p25", "p50"):
                    assert len(bands[key]) == 11
                    assert bands[key][0] == 1_000_000
                for y in bands["years"]:
                    assert (bands["p5"][y] <= bands["p10"][y]
                            <= bands["p25"][y] <= bands["p50"][y])
                # The final year's percentiles are those of terminal NAV.
                navs = sim["results"]["Terminal NAV"]
                assert bands["p50"][-1] == pytest.approx(np.median(navs))

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

    @pytest.mark.parametrize("backend", ["process", "numba"])
    def test_progress_callback_counts_every_portfolio(
            self, backend, tmp_path, market_cache):
        calls = []
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(
            {**SWEEP, "models": ALL_MODELS[:2], "total_paths": 4}))
        cli = mc.MonteCarloCLI(str(config_file), str(market_cache),
                               progress_callback=lambda *a: calls.append(a))
        cli.run(backend=backend)
        # 2 models x 2 equity x 2 spending x 2 regimes; the numba backend
        # reports once per cell, covering both regime variants at once.
        step = 1 if backend == "process" else 2
        assert calls == [(done, 16) for done in range(step, 17, step)]

    def test_fewer_paths_than_workers_completes(
            self, tmp_path, market_cache, call_with_timeout):
        # Used to hang forever: 3 // 4 == 0 paths per chunk.
        cli = call_with_timeout(
            lambda: run_cli(tmp_path, market_cache, total_paths=3, workers=4))
        for sims in cli.raw_results["simulations"].values():
            for sim in sims:
                assert len(sim["results"]["Terminal NAV"]) == 3
                assert len(sim["results"]["Terminal SPX"]) == 3
                assert len(sim["ruin_histogram"]) == 120


class TestChunkSizes:
    def test_fewer_paths_than_workers_terminates(self, call_with_timeout):
        chunks = call_with_timeout(
            lambda: mc.MonteCarloEngine._chunk_sizes(3, 4), seconds=5)
        assert chunks == [1, 1, 1]

    @pytest.mark.parametrize("total_paths,n_workers,expected", [
        (20, 2, [10, 10]),
        (37, 4, [9, 9, 9, 9, 1]),
        (50_000, 20, [2_500] * 20),
        (4, 4, [1, 1, 1, 1]),
    ])
    def test_unchanged_when_paths_cover_workers(
            self, total_paths, n_workers, expected):
        # Chunk sizes decide each chunk's seed, so any change here would
        # silently change results (and the golden snapshot).
        assert mc.MonteCarloEngine._chunk_sizes(
            total_paths, n_workers) == expected

    @pytest.mark.parametrize("total_paths", range(1, 30))
    @pytest.mark.parametrize("n_workers", [1, 2, 3, 7, 20])
    def test_chunks_cover_every_path(
            self, total_paths, n_workers, call_with_timeout):
        chunks = call_with_timeout(
            lambda: mc.MonteCarloEngine._chunk_sizes(total_paths, n_workers),
            seconds=5)
        assert sum(chunks) == total_paths
        assert all(c > 0 for c in chunks)


class TestCommandLine:
    REQUIRED = ["-c", "cfg.json", "-r", "raw.json", "-o", "agg.json"]

    @staticmethod
    def set_command_line(monkeypatch, args):
        monkeypatch.setattr(sys, "argv", ["monte_carlo.py", *args])

    def test_backend_defaults_to_process(self, monkeypatch):
        self.set_command_line(monkeypatch, self.REQUIRED)
        assert mc.parse_args().backend == "process"

    @pytest.mark.parametrize("flag", ["-b", "--backend"])
    @pytest.mark.parametrize("backend", ["process", "numba"])
    def test_backend_flag(self, flag, backend, monkeypatch):
        self.set_command_line(monkeypatch, self.REQUIRED + [flag, backend])
        assert mc.parse_args().backend == backend

    def test_unknown_backend_rejected(self, monkeypatch):
        self.set_command_line(
            monkeypatch, self.REQUIRED + ["--backend", "bogus"])
        with pytest.raises(SystemExit):
            mc.parse_args()

    @pytest.mark.parametrize("backend", ["process", "numba"])
    def test_main_runs_requested_backend(
            self, backend, tmp_path, market_cache, monkeypatch):
        calls = []
        original_run = mc.MonteCarloCLI.run

        def spy(self, *, backend="process"):
            calls.append(backend)
            return original_run(self, backend=backend)

        monkeypatch.setattr(mc.MonteCarloCLI, "run", spy)
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(
            {**SWEEP, "models": [ALL_MODELS[0]], "total_paths": 4}))
        raw_file, agg_file = tmp_path / "raw.json", tmp_path / "agg.json"

        self.set_command_line(monkeypatch, [
            "-c", str(config_file), "-r", str(raw_file),
            "-o", str(agg_file), "-m", str(market_cache),
            "--backend", backend])
        mc.main()

        assert calls == [backend]
        raw = json.loads(raw_file.read_text())
        assert list(raw["simulations"]) == [ALL_MODELS[0]]
        assert all(len(sim["results"]["Terminal NAV"]) == 4
                   for sim in raw["simulations"][ALL_MODELS[0]])
        assert ALL_MODELS[0] in json.loads(agg_file.read_text())["results"]

    def test_requires_outputs_without_job_dir(self, monkeypatch, capsys):
        self.set_command_line(monkeypatch, ["-c", "cfg.json"])
        with pytest.raises(SystemExit):
            mc.parse_args()
        err = capsys.readouterr().err
        assert "-r/--raw-output-file" in err
        assert "-o/--aggregated-output-file" in err
        assert "-c/--config-file" not in err

    @pytest.mark.parametrize("flag", ["-j", "--job-dir"])
    def test_job_dir_supplies_config_and_results_paths(
            self, flag, monkeypatch):
        self.set_command_line(monkeypatch, [flag, "runs/abc"])
        args = mc.parse_args()
        assert args.config_filename == os.path.join("runs/abc", "config.json")
        assert args.agg_output_filename == os.path.join(
            "runs/abc", "results.json")
        assert args.raw_output_filename is None

    def test_explicit_paths_override_job_dir(self, monkeypatch):
        self.set_command_line(monkeypatch, [
            "--job-dir", "runs/abc", "-c", "cfg.json", "-o", "agg.json",
            "-r", "raw.json"])
        args = mc.parse_args()
        assert (args.config_filename, args.agg_output_filename,
                args.raw_output_filename) == (
                    "cfg.json", "agg.json", "raw.json")

    def test_main_with_job_dir(self, tmp_path, market_cache, monkeypatch):
        config = {**SWEEP, "models": [ALL_MODELS[0]], "total_paths": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))
        self.set_command_line(monkeypatch, [
            "--job-dir", str(tmp_path), "-m", str(market_cache),
            "--backend", "numba"])
        mc.main()

        status = json.loads((tmp_path / "status.json").read_text())
        assert status["state"] == job_files.SUCCEEDED
        assert status["pid"] == os.getpid()
        assert (status["cells_done"], status["cells_total"]) == (8, 8)
        assert status["error"] is None

        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["master_seed"] == SWEEP["master_seed"]
        assert meta["backend"] == "numba"
        assert "engine_commit" in meta

        results = json.loads((tmp_path / "results.json").read_text())
        assert len(results["results"][ALL_MODELS[0]]) == 8
        assert results["assumptions"] == {
            "dividend_yield": 0.01, "simulator_params": {}}
        for entry in results["results"][ALL_MODELS[0]]:
            for key in ("p5_real_return", "p10_real_return",
                        "p25_real_return", "p50_real_return",
                        "real_nav_bands", "ruin_prob_by_age"):
                assert entry[key] is not None, key
        # Raw output is opt-in with --job-dir, and nothing is left behind
        # by the atomic writes.
        assert sorted(f.name for f in tmp_path.iterdir()) == [
            "config.json", "meta.json", "results.json", "status.json"]

    def test_main_with_job_dir_records_failure(
            self, tmp_path, market_cache, monkeypatch):
        config = {**SWEEP, "models": ["NoSuchModel"], "total_paths": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))
        self.set_command_line(monkeypatch, [
            "--job-dir", str(tmp_path), "-m", str(market_cache)])
        with pytest.raises(ValueError):
            mc.main()
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["state"] == job_files.FAILED
        assert "NoSuchModel not supported" in status["error"]
        assert not (tmp_path / "results.json").exists()


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
        # Statistics over ruined paths were removed: consumers derive what
        # they need from ruin_histogram.
        for key in ("ruin_month_min", "ruin_month_median",
                    "ruin_month_es5", "ruin_month_es10"):
            assert key not in entry, key
        assert entry["allocation"] == "60-40"
        assert entry["equity"] == 0.6
        assert entry["ruin_histogram"] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

    def test_no_ruin(self, tmp_path, market_cache):
        entry = self.build(tmp_path, market_cache, [0.0] * 12,
                           [1_000_000.0] * 4)
        assert entry["ruin_path_count"] == 0

    def test_return_quantiles(self, tmp_path, market_cache):
        navs = [0.0, 500_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
        entry = self.build(tmp_path, market_cache, [0.0] * 12, navs)
        assert entry["p50_return"] == pytest.approx(0.0)
        assert entry["p25_return"] == pytest.approx(-0.5)
        assert entry["p10_return"] == pytest.approx(-0.8)
        assert entry["p5_return"] == pytest.approx(-0.9)

    def test_nav_bands_default_to_none(self, tmp_path, market_cache):
        entry = self.build(tmp_path, market_cache, [0.0] * 12, [1.0] * 4)
        assert entry["nav_bands"] is None

    def test_raw_results_without_new_fields(self, tmp_path, market_cache):
        # Raw files written before assumptions and real-terms metrics
        # existed still aggregate; the new fields are None, except
        # ruin_prob_by_age, which only needs the ruin histogram.
        histogram = [0.0] * 120
        histogram[30] = 1.0
        cli = mc.MonteCarloCLI("unused.json", str(market_cache))
        cli.simulation_config = mc.MonteCarloCLI.MCConfig(
            models=["M"], retirement_age=60)
        cli.raw_results = {
            "initial_nav": 1_000_000, "years": 10, "total_paths": 4,
            "simulations": {"M": [{
                "spending": 50_000, "equity": 0.6, "ladder": 0.4,
                "tax_regime": "none", "ruin_histogram": histogram,
                "results": {"Terminal SPX": [1.0] * 4,
                            "Terminal NAV": [0.0] + [1.0] * 3}}]},
        }
        cli.aggregate()
        assert cli.agg_results["assumptions"] is None
        entry = cli.agg_results["results"]["M"][0]
        for key in ("p5_real_return", "p10_real_return", "p25_real_return",
                    "p50_real_return", "real_nav_bands"):
            assert entry[key] is None, key
        assert entry["ruin_prob_by_age"] == {"75": 0.25, "85": 0.25,
                                             "95": 0.25}

    def test_aggregate_only_once(self, cli):
        with pytest.raises(RuntimeError, match="only be run once"):
            cli.aggregate()



BACKENDS = ["process", "numba"]


@pytest.fixture(scope="module")
def flat_cpi_market(tmp_path_factory):
    """The synthetic market with a constant CPI: zero inflation."""
    from conftest import make_market_levels
    levels = make_market_levels()
    levels["cpi"] = 97.0
    path = tmp_path_factory.mktemp("flat_cpi") / "market.parquet"
    levels.to_parquet(path)
    return path


class TestAssumptions:
    """simulator_params, dividend_yield and their validation."""
    MODELS = ["RegimeSwitchingValuationVARSimulator",
              "RawBlockBootstrapSimulator"]

    def run(self, tmp_path, market_cache, tag, backend="process",
            **overrides):
        return run_cli(tmp_path, market_cache, tag=tag, backend=backend,
                       models=self.MODELS, **overrides)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_assumptions_are_recorded(self, backend, tmp_path, market_cache):
        params = {"initial_cape": 40.0, "annual_buyback_yield": 0.015}
        cli = self.run(tmp_path, market_cache, "rec", backend,
                       simulator_params=params, dividend_yield=0.02)
        expected = {"dividend_yield": 0.02, "simulator_params": params}
        assert cli.raw_results["assumptions"] == expected
        assert cli.agg_results["assumptions"] == expected

    def test_defaults_match_the_original_model(self, cli):
        assert cli.agg_results["assumptions"] == {
            "dividend_yield": 0.01, "simulator_params": {}}

    @pytest.mark.parametrize("params", [{}, None])
    def test_empty_or_missing_params_change_nothing(
            self, params, tmp_path, market_cache):
        base = self.run(tmp_path, market_cache, "base")
        overrides = {} if params is None else {"simulator_params": params}
        other = self.run(tmp_path, market_cache, "other", **overrides)
        assert comparable(other.raw_results) == comparable(base.raw_results)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_params_reach_only_models_that_accept_them(
            self, backend, tmp_path, market_cache):
        base = self.run(tmp_path, market_cache, "base", backend)
        bought = self.run(tmp_path, market_cache, "bought", backend,
                          simulator_params={"annual_buyback_yield": 0.02})
        for model, changed in (("RegimeSwitchingValuationVARSimulator",
                                True),
                               ("RawBlockBootstrapSimulator", False)):
            a = base.raw_results["simulations"][model][0]["results"]
            b = bought.raw_results["simulations"][model][0]["results"]
            assert (a["Terminal SPX"] != b["Terminal SPX"]) is changed, model

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_dividend_yield_reaches_the_strategy(
            self, backend, tmp_path, market_cache):
        low = self.run(tmp_path, market_cache, "low", backend,
                       dividend_yield=0.01)
        high = self.run(tmp_path, market_cache, "high", backend,
                        dividend_yield=0.03)
        model = "RawBlockBootstrapSimulator"
        a = low.raw_results["simulations"][model][0]["results"]
        b = high.raw_results["simulations"][model][0]["results"]
        assert a["Terminal SPX"] == b["Terminal SPX"]  # same markets
        assert np.mean(b["Terminal NAV"]) > np.mean(a["Terminal NAV"])

    def test_unknown_simulator_param_is_rejected(
            self, tmp_path, market_cache):
        with pytest.raises(ValueError, match="annual_buyback_yeild"):
            self.run(tmp_path, market_cache, "typo",
                     simulator_params={"annual_buyback_yeild": 0.01})

    def test_every_unknown_param_is_listed_sorted(
            self, tmp_path, market_cache):
        with pytest.raises(ValueError, match=r": alpha, zulu$"):
            self.run(tmp_path, market_cache, "typos",
                     simulator_params={"zulu": 1, "initial_cape": 40.0,
                                       "alpha": 2})

    def test_partially_accepted_param_is_logged(
            self, tmp_path, market_cache, caplog):
        with caplog.at_level("INFO"):
            self.run(tmp_path, market_cache, "partial",
                     simulator_params={"annual_buyback_yield": 0.01})
        ignoring = [r.getMessage() for r in caplog.records
                    if "ignores simulator_params" in r.getMessage()]
        assert ignoring == ["RawBlockBootstrapSimulator ignores "
                            "simulator_params annual_buyback_yield"]

    def load(self, tmp_path, market_cache, models, params):
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(
            {**SWEEP, "models": models, "simulator_params": params}))
        cli = mc.MonteCarloCLI(str(config_file), str(market_cache))
        cli._load_config()
        return cli

    def test_valuation_adjusted_takes_cape_but_not_buybacks(
            self, tmp_path, market_cache):
        params = {"initial_cape": 40.0, "annual_buyback_yield": 0.015}
        cli = self.load(tmp_path, market_cache,
                        ["ValuationAdjustedVARSimulator",
                         "RegimeSwitchingValuationVARSimulator"], params)
        cli._validate_simulator_params()
        valuation = cli._build_simulator("ValuationAdjustedVARSimulator")
        assert valuation.initial_cape == 40.0
        assert not hasattr(valuation, "buyback_yield")
        regime = cli._build_simulator("RegimeSwitchingValuationVARSimulator")
        assert (regime.initial_cape, regime.buyback_yield) == (40.0, 0.015)

    def test_buybacks_alone_on_valuation_adjusted_are_rejected(
            self, tmp_path, market_cache):
        cli = self.load(tmp_path, market_cache,
                        ["ValuationAdjustedVARSimulator"],
                        {"annual_buyback_yield": 0.015})
        with pytest.raises(ValueError, match="annual_buyback_yield"):
            cli._validate_simulator_params()


class TestRealMetrics:
    """Terminal Real NAV, real_nav_bands and p*_real_return."""
    MODELS = TestAssumptions.MODELS

    def test_real_metrics(self, cli):
        for model, entries in cli.agg_results["results"].items():
            raw = cli.raw_results["simulations"][model]
            for entry, sim in zip(entries, raw):
                where = f"{model} {entry['allocation']}"
                nominal = np.asarray(sim["results"]["Terminal NAV"])
                real = np.asarray(sim["results"]["Terminal Real NAV"])
                # The synthetic market inflates on every path.
                assert np.all(real <= nominal + 1e-9), where
                assert entry["p50_real_return"] <= entry["p50_return"], where
                bands = entry["real_nav_bands"]
                assert bands["years"] == entry["nav_bands"]["years"], where

    def test_year_zero_is_the_initial_nav(self, cli):
        for entries in cli.agg_results["results"].values():
            for entry in entries:
                bands = entry["real_nav_bands"]
                for key in ("p5", "p10", "p25", "p50"):
                    assert bands[key][0] == SWEEP["initial_nav"], key

    def test_ruined_paths_are_zero_in_real_terms(
            self, tmp_path, market_cache):
        cli = run_cli(tmp_path, market_cache, "ruin",
                      models=["RawBlockBootstrapSimulator"],
                      yearly_spending_floor=140_000,
                      yearly_spending_ceil=140_000, total_paths=40)
        for sim in cli.raw_results["simulations"][
                "RawBlockBootstrapSimulator"]:
            nominal = np.asarray(sim["results"]["Terminal NAV"])
            real = np.asarray(sim["results"]["Terminal Real NAV"])
            assert np.any(nominal == 0.0)  # not vacuous
            np.testing.assert_array_equal(real == 0.0, nominal == 0.0)
            # Ruined paths count as 0 in the bands, not dropped.
            bands = sim["real_nav_bands"]
            for q in mc.NAV_BAND_PERCENTILES:
                assert bands[f"p{q}"][-1] == pytest.approx(
                    np.percentile(real, q))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_zero_inflation_real_equals_nominal(
            self, backend, tmp_path, flat_cpi_market):
        # RawBlockBootstrap resamples the history's CPI returns, which are
        # all zero here, so every price level is exactly 1.
        cli = run_cli(tmp_path, flat_cpi_market, "flat", backend,
                      models=["RawBlockBootstrapSimulator"])
        for sim in cli.raw_results["simulations"][
                "RawBlockBootstrapSimulator"]:
            assert (sim["results"]["Terminal Real NAV"]
                    == sim["results"]["Terminal NAV"])
            assert sim["real_nav_bands"] == sim["nav_bands"]
        for entry in cli.agg_results["results"][
                "RawBlockBootstrapSimulator"]:
            for q in (5, 10, 25, 50):
                assert entry[f"p{q}_real_return"] == pytest.approx(
                    entry[f"p{q}_return"], rel=1e-12, abs=1e-15), q

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_fractional_years_share_the_nominal_snapshots(
            self, backend, tmp_path, market_cache):
        # 10.5 years: the trailing half year has no annual snapshot.
        cli = run_cli(tmp_path, market_cache, "frac", backend,
                      models=self.MODELS, years_to_simulate=10.5)
        for sims in cli.raw_results["simulations"].values():
            for sim in sims:
                assert sim["real_nav_bands"]["years"] == list(range(11))
                assert (sim["real_nav_bands"]["years"]
                        == sim["nav_bands"]["years"])


@pytest.fixture(scope="module")
def ruinous(tmp_path_factory, market_cache):
    """A run where most paths ruin, at many different ages."""
    t = TestRuinProbabilityByAge
    ages = [t.age(m) for m in t.MONTHS] + list(t.EXTRA_AGES)
    return run_cli(tmp_path_factory.mktemp("ruin_age"), market_cache,
                   models=["RawBlockBootstrapSimulator"],
                   yearly_spending_floor=140_000,
                   yearly_spending_ceil=140_000, equity_floor=0.6,
                   equity_ceil=0.6, total_paths=40, tax_regimes=["none"],
                   retirement_age=t.RETIREMENT_AGE, ruin_ages=ages)


class TestRuinProbabilityByAge:
    """ruin_prob_by_age on a run where ruins happen at many ages."""
    RETIREMENT_AGE = 60.5
    # Whole months after retirement, for comparing with Cell.survival().
    MONTHS = (1, 60, 80, 90, 100, 110, 119, 120)
    EXTRA_AGES = (55, 60.5, 200)  # before / at retirement, past horizon

    @staticmethod
    def age(month):
        return TestRuinProbabilityByAge.RETIREMENT_AGE + month / 12.0

    @pytest.fixture
    def entry(self, ruinous):
        return ruinous.agg_results["results"][
            "RawBlockBootstrapSimulator"][0]

    def test_ruins_happen_at_many_ages(self, entry):
        ruin_months = np.nonzero(entry["ruin_histogram"])[0]
        assert len(ruin_months) >= 5
        assert ruin_months.min() < 90 < ruin_months.max()

    def test_is_non_decreasing_and_bounded(self, entry):
        probs = entry["ruin_prob_by_age"]
        by_age = sorted((float(k), v) for k, v in probs.items())
        values = [v for _, v in by_age]
        assert values == sorted(values)
        assert len(set(values)) > 2  # it actually moves with age
        ruined = entry["ruin_path_count"] / 40
        assert max(values) == ruined

    def test_agrees_with_cell_survival(self, ruinous):
        from planner.decision import cells_from_results
        [cell] = cells_from_results(ruinous.agg_results)
        survival = cell.survival()
        for n in self.MONTHS:
            assert cell.ruin_prob_by_age[f"{self.age(n):g}"] == (
                pytest.approx(1.0 - survival[n - 1])), n

    def test_ages_at_or_before_retirement_are_zero(self, entry):
        assert entry["ruin_prob_by_age"]["55"] == 0.0
        assert entry["ruin_prob_by_age"]["60.5"] == 0.0

    def test_ages_past_the_horizon_count_every_ruin(self, entry):
        assert entry["ruin_prob_by_age"]["200"] == (
            entry["ruin_path_count"] / 40)

    def test_default_ages_in_the_shared_run(self, cli):
        paths = cli.agg_results["total_paths"]
        for entries in cli.agg_results["results"].values():
            for entry in entries:
                probs = entry["ruin_prob_by_age"]
                assert list(probs) == ["75", "85", "95"]
                # Retirement at 60 with a 10-year horizon: every age is
                # past the end, so each covers all ruined paths.
                assert all(p == entry["ruin_path_count"] / paths
                           for p in probs.values())

    def test_custom_ruin_ages(self, tmp_path, market_cache):
        cli = run_cli(tmp_path, market_cache, "ages",
                      models=TestAssumptions.MODELS, ruin_ages=[62, 65.5])
        entry = cli.agg_results["results"][TestAssumptions.MODELS[0]][0]
        assert list(entry["ruin_prob_by_age"]) == ["62", "65.5"]


def check_golden(cli, path):
    actual = json.loads(json.dumps(cli.agg_results))
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(json.dumps(actual, indent=2) + "\n")
    expected = json.loads(path.read_text())
    assert_close(actual, expected)


def test_golden_aggregates(cli):
    """
    Regression snapshot of the aggregated results for a fixed seed and
    synthetic market. A failure means simulation behaviour changed: if the
    change is intended, regenerate with `UPDATE_GOLDEN=1 pytest` and review
    the diff of tests/golden/monte_carlo_agg.json.
    """
    check_golden(cli, GOLDEN)


def test_golden_aggregates_with_assumptions(tmp_path, market_cache):
    """
    Same as test_golden_aggregates for non-default simulator_params and
    dividend_yield, pinning the initial_cape / buyback / dividend plumbing.
    Regenerated by the same `UPDATE_GOLDEN=1 pytest`.
    """
    cli = run_cli(
        tmp_path, market_cache, "golden_assumptions",
        models=["HybridValuationVARSimulator",
                "RegimeSwitchingValuationVARSimulator",
                "ValuationAdjustedVARSimulator"],
        tax_regimes=["current_law_indexed"], dividend_yield=0.02,
        simulator_params={"initial_cape": 40.0, "target_cape": 26.0,
                          "annual_buyback_yield": 0.015})
    check_golden(cli, GOLDEN_ASSUMPTIONS)


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
