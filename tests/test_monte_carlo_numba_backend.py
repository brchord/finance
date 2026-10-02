"""
Parity between MonteCarloCLI.run(backend="numba") and the default
backend="process".

backend="numba" draws market paths with the identical chunk-size/seed
sequence backend="process" uses (see MonteCarloEngine._chunk_sizes), and
its operator and regime-switching path generators are bit-identical to
the reference, so results must match exactly. The one exception is
HybridValuationVARSimulator, whose fast path generator matches only to
rounding (see tests/test_fast_hybrid_path_simulation.py); it's compared
with a tolerance.
"""
import json

import numpy as np
import pytest

import monte_carlo as mc

MATCHES_ONLY_TO_ROUNDING = {"HybridValuationVARSimulator"}


def assert_backends_match(model_name, p_run, n_run):
    where = (f"model={model_name} spending={p_run['spending']} "
             f"equity={p_run['equity']} regime={p_run['tax_regime']}")
    assert (p_run["spending"], p_run["equity"], p_run["tax_regime"]) == (
        n_run["spending"], n_run["equity"], n_run["tax_regime"]), where
    p_res, n_res = p_run["results"], n_run["results"]
    if model_name in MATCHES_ONLY_TO_ROUNDING:
        np.testing.assert_allclose(
            n_res["Terminal SPX"], p_res["Terminal SPX"], rtol=1e-9,
            err_msg=f"Terminal SPX mismatch: {where}")
        np.testing.assert_allclose(
            n_res["Terminal NAV"], p_res["Terminal NAV"], rtol=1e-6,
            atol=1e-3, err_msg=f"Terminal NAV mismatch: {where}")
    else:
        assert n_res["Terminal SPX"] == p_res["Terminal SPX"], where
        assert n_res["Terminal NAV"] == p_res["Terminal NAV"], where
    assert n_run["ruin_histogram"] == p_run["ruin_histogram"], where

SWEEP = dict(
    yearly_spending_floor=80_000, yearly_spending_ceil=90_000,
    spend_increments=10_000, equity_floor=0.5, equity_ceil=0.6,
    weight_increments=0.1, initial_nav=1_000_000, years_to_simulate=10,
    retirement_age=60, total_paths=37, workers=4,  # odd/uneven on purpose,
    # to exercise _chunk_sizes' uneven-remainder branch identically on
    # both backends.
    tax_regimes=["none", "current_law_indexed", "historical_average_drift"],
    master_seed=123)
ALL_MODELS = [m.name() for m in mc.MonteCarloCLI.SUPPORTED_MODELS]


def run_cli(tmp_path, market_cache, tag, backend, **overrides):
    config = {**SWEEP, "models": ALL_MODELS, **overrides}
    config_file = tmp_path / f"{tag}.json"
    config_file.write_text(json.dumps(config))
    cli = mc.MonteCarloCLI(str(config_file), str(market_cache))
    cli.run(backend=backend)
    return cli


class TestNumbaBackendMatchesProcessBackend:
    def test_terminal_values_and_ruin_histograms_match(
            self, tmp_path, market_cache):
        process_cli = run_cli(tmp_path, market_cache, "process", "process")
        numba_cli = run_cli(tmp_path, market_cache, "numba", "numba")

        process_sims = process_cli.raw_results["simulations"]
        numba_sims = numba_cli.raw_results["simulations"]

        assert set(process_sims.keys()) == set(numba_sims.keys())

        for model_name in process_sims:
            p_runs = process_sims[model_name]
            n_runs = numba_sims[model_name]
            assert len(p_runs) == len(n_runs)

            for p_run, n_run in zip(p_runs, n_runs):
                assert_backends_match(model_name, p_run, n_run)

    def test_rejects_unknown_backend(self, tmp_path, market_cache):
        config = {**SWEEP, "models": ALL_MODELS}
        config_file = tmp_path / "bogus.json"
        config_file.write_text(json.dumps(config))
        cli = mc.MonteCarloCLI(str(config_file), str(market_cache))
        with pytest.raises(ValueError, match="Unknown backend"):
            cli.run(backend="bogus")

    def test_fractional_years_match_across_backends(
            self, tmp_path, market_cache):
        # years_to_simulate=10.5 arrives from JSON as a float; it used to
        # crash backend="process" (np.zeros(126.0)) while backend="numba"
        # converted on its own. Both now read MCConfig.simulation_months.
        process_cli = run_cli(tmp_path, market_cache, "process_frac",
                              "process", years_to_simulate=10.5)
        numba_cli = run_cli(tmp_path, market_cache, "numba_frac", "numba",
                            years_to_simulate=10.5)

        for model_name, p_runs in process_cli.raw_results[
                "simulations"].items():
            n_runs = numba_cli.raw_results["simulations"][model_name]
            for p_run, n_run in zip(p_runs, n_runs, strict=True):
                assert len(p_run["ruin_histogram"]) == 126
                assert_backends_match(model_name, p_run, n_run)

    def test_fewer_paths_than_workers_matches_process_backend(
            self, tmp_path, market_cache, call_with_timeout):
        # 3 paths over 4 workers used to hang both backends; now it runs as
        # three 1-path chunks, which also exercises the fast kernels at
        # num_paths == 1.
        process_cli, numba_cli = call_with_timeout(lambda: (
            run_cli(tmp_path, market_cache, "process_small", "process",
                    total_paths=3, workers=4),
            run_cli(tmp_path, market_cache, "numba_small", "numba",
                    total_paths=3, workers=4)))

        for model_name, p_runs in process_cli.raw_results[
                "simulations"].items():
            n_runs = numba_cli.raw_results["simulations"][model_name]
            assert len(p_runs) == len(n_runs)
            for p_run, n_run in zip(p_runs, n_runs):
                assert len(n_run["results"]["Terminal NAV"]) == 3
                assert_backends_match(model_name, p_run, n_run)
