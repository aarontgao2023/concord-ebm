# Changelog

## 0.1.0 — 2026-09-13

First public version.

- `concord.compare(df, group_column=...)`: composition-invariant estimator (pooled measurement model +
  consensus standardised to a common composition) with diagnosis-stratified, pair-restricted and
  max-statistic permutation tests; exact decisions under a fixed permutation budget; composition table,
  effective sample size, resampling stability and paired p-difference diagnostics.
- Estimators: `invariant_min` (default), `invariant_pooled`, `shared`, `standard`, `saebm` (optional).
- Command-line entry point `concord`.
- Worker pool based on `concurrent.futures`: a worker that dies (e.g. an unguarded script re-imported by a
  spawned process) raises a clear error instead of hanging.
- Simulations, protocols and analyses are kept in the companion repository `concord-ebm-paper`.
- Pinned to the unmodified `pyebm==2.0.3`; the package verifies the installed source before fitting.
