# Monte Carlo retirement engine

- Engine: `monte_carlo.py` (CLI), `market_modelling/`, `portfolio_models/`,
  `tax_models/`. Tests: `python -m pytest -q`; types: `python -m mypy .`.
- UI: Streamlit (`streamlit run app.py`, pages in `pages/`, logic in
  `planner/`). Design and decisions: `doc/plans/UI Design.md`.

**Core rule:** the UI launches the CLI as a detached subprocess
(`monte_carlo.py --job-dir`) and only reads its output files
(`job_files.py` is the shared protocol). It never imports or runs
simulation code in the Streamlit process.
