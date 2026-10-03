# Monte Carlo retirement engine

- Engine: `monte_carlo.py` (CLI), `market_modelling/`, `portfolio_models/`,
  `tax_models/`. Tests: `python -m pytest -q`; types: `python -m mypy .`.
- UI: Streamlit (`streamlit run app.py`, pages in `views/`, logic in
  `planner/`). Design and decisions: `doc/plans/UI Design.md`.

**Core rule:** the UI launches the CLI as a detached subprocess
(`monte_carlo.py --job-dir`) and only reads its output files
(`job_files.py` is the shared protocol). It never imports or runs
simulation code in the Streamlit process.

## Conventions and decisions

- **CLI parsing:** all argument parsing lives inside each script's
  `parse_args()`, which reads `sys.argv` itself. Don't add `argv`
  parameters to `parse_args()`/`main()` or thread them through functions;
  tests set the command line with `monkeypatch.setattr(sys, "argv", [...])`.
- **No GPU:** the CPU Numba path (`--backend numba`) made GPU work
  unnecessary, and the code may move to non-NVIDIA hardware (e.g. Apple
  Silicon) where CUDA can't run. Don't add CuPy/numba-cuda.
- **Known flaky CI:** GitHub Actions once failed intermittently on `main`
  and passed on re-run; it couldn't be reproduced locally. Suspects: the
  exact-equality Numba parity tests (`tests/test_fast_*.py`) varying with
  the runner's CPU, or unpinned transitive dependencies (llvmlite). If it
  recurs, capture the failing assertions, runner CPU and `pip freeze`
  first.
