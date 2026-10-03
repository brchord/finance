import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import job_files
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
        assert entry["ruin_month_min"] == 1
        assert entry["ruin_month_median"] == 5.0
        # 5th/10th percentiles of [1,3,5,7,9] are 1.4 / 1.8: only month 1
        # falls in either tail.
        assert entry["ruin_month_es5"] == 1.0
        assert entry["ruin_month_es10"] == 1.0
        assert entry["allocation"] == "60-40"
        assert entry["equity"] == 0.6
        assert entry["ruin_histogram"] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

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

    def test_nav_bands_default_to_none(self, tmp_path, market_cache):
        entry = self.build(tmp_path, market_cache, [0.0] * 12, [1.0] * 4)
        assert entry["nav_bands"] is None

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
