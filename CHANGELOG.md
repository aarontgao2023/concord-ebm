# Changelog

## 0.2.0 — 2026-09-26

Names, options and outputs now follow the manuscript "Comparing biomarker orderings between patient
groups with different diagnostic composition". Calls written for 0.1 keep running: the earlier option
names are still accepted, and a given seed gives the same relabelings as in 0.1.1. Two things differ:
without `schemes=`, a call now runs only the within-diagnosis tests (see Changed), and code that reads
result keys must use the new names listed under Changed and Removed.

Added
- Fixed common proportions for CONCORD: `compare(..., common_proportions=...)`, the `CompareSpec`
  field of the same name and the command-line option `--common-proportions`. The value is `None` or
  `"min"` (the minimum rule, the default), a mapping `{label: proportion}` or a sequence in label
  order. Fixed proportions must be nonnegative and sum to one (within 0.001; percentages such as
  48.9/28.6/22.5 are rejected) and are held fixed in the observed fit, every relabeling and every
  bootstrap refit. A diagnosis left out of a mapping gets proportion 0, and participants with a
  diagnosis of proportion 0 get weight 0. `compare()` raises `ValueError` if a diagnosis with a
  positive common proportion has no participants in one of the groups. Common proportions apply to
  CONCORD only; with any other model they raise `ValueError`.
- `concord.common_proportions(*groups, labels=..., group_column=...)` applies the minimum rule jointly
  to any number of groups, for example to the six cohort-by-genotype groups of two cohorts, so that
  both cohorts are compared at the same proportions. For the paper's six groups it gives 0.48926,
  0.28593 and 0.22481 (rounded).
- Adjusted P values: every comparison reports `p_adjusted = min(1, n_pairs × P)` and
  `p_adjusted_bounds`; `summary()` prints them.
- Result fields `common_proportions` (the proportions used in the observed fit), `weights` (per group
  and diagnosis) and `largest_weight`; `decisions["reject"]` (the Bonferroni decision of every
  comparison by scheme) and `decisions["n_pairs"]`.
- `concord.SEARCHES` and `CompareSpec.default_schemes`.

Changed
- Default permutation schemes: `schemes=None` (the new default of `compare()` and of the command
  line) runs within-diagnosis permutation for two groups and pairwise within-diagnosis permutation for
  three or more, as in the paper's *APOE* analyses. Unrestricted permutation runs only when requested;
  0.1 ran unrestricted, within-diagnosis and pairwise within-diagnosis permutation by default. To get
  the 0.1 set of tests, pass `schemes=concord.SCHEMES`. When unrestricted permutation is requested,
  `summary()` notes that it does not keep each group's diagnostic counts.
- Model names: `estimator="concord"` (default), `"pooled_score"` (pooled-score DEBM), `"separate"`
  (separately fitted DEBM) and `"saebm"` (SA-EBM). The earlier names `invariant_min`, `shared` and
  `standard` are accepted. `concord.ESTIMATORS` now holds the new names only,
  `("concord", "pooled_score", "separate", "saebm")`; `invariant_pooled` (not used in the paper) moved
  to `concord.core.EXTRA_ESTIMATORS`.
- Ranking-consensus search: `search="continued"` (default) or `"pyebm"` (the unmodified pyebm 2.0.3
  search). The earlier option `consensus=` and its values `repaired` and `original` are accepted.
  `CompareSpec` stores the search as the field `search`; `consensus` is now only an initialization
  argument, so `spec.consensus` returns `None` and `result.spec` has a `search` key and no `consensus`
  key.
- Permutation schemes: `"unrestricted"`, `"within_diagnosis"` and `"pairwise_within_diagnosis"`; the
  earlier names `diagnosis` and `diagnosis_pair` are accepted. Result keys are renamed accordingly:
  `result.tests["diagnosis"]` is now `result.tests["within_diagnosis"]`, and
  `result.tests["diagnosis_pair_<a>-<b>"]` is now `result.tests["pairwise_within_diagnosis_<a>-<b>"]`.
  With two groups, a request for pairwise within-diagnosis permutation now runs within-diagnosis
  permutation (0.1 ran no test).
- `decisions["reject_any_pair"]` is keyed by scheme (`unrestricted`, `within_diagnosis`,
  `pairwise_within_diagnosis`, the last over all its pairwise tests) instead of by test name.
- The composition table reports the minimum-rule proportions as `composition["min_rule"]` (was
  `min_reference`). The fit diagnostics report `common_proportions`, `common_proportions_rule` and
  `group_weights`.
- The minimum rule raises `ValueError` when no diagnosis is present in every group (0.1 returned
  NaN weights).
- With fixed common proportions, an unrestricted relabeling that leaves a group without participants
  of a diagnosis with a positive common proportion counts as a failed fit, which leaves the decision
  undetermined rather than changing the denominator.
- P values when some relabelings fail (unchanged in behaviour, now documented): `p` and `p_adjusted`
  use the completed relabelings, (1 + E)/(completed + 1), whereas `reject` and `p_bounds` refer to the
  planned B and count failed or unfinished relabelings as possible exceedances.
- The command line checks every option, including `--common-proportions` against the model, before
  reading the input file, and reports invalid combinations as usage errors.
- SA-EBM defaults: 10,000 iterations and 2,500 burn-in iterations (were 2,000 and 500), as in the
  manuscript. The optional dependency is pinned to `pysaebm==7.7.7`.
- Docstrings, command-line help, README, user manual and examples use the manuscript's terms. The
  overview figure in the README shows panels a and b of the manuscript's Fig. 1.

Removed
- The maximum-distance statistic, from the tests (`tests[name]["max"]`), the distances
  (`distances["max"]` and `distances["max_pair"]`), the decisions, early stopping and the summary.
  Decisions are the Bonferroni decisions of the pairwise comparisons.
- The labels attached to the permutation schemes in 0.1 (`decisions["primary"]`,
  `decisions["localisation"]`, `decisions["global"]`).
- `composition["min_reference"]` (now `composition["min_rule"]`).
- The `diagnosis_count` permutation scheme added in 0.1.1 and `concord.core.OPTIONAL_SCHEMES`.

Fixed
- `compare(workers=1)` can be called inside a worker process; the guard against unguarded scripts
  applies only when `workers > 1`.
- The cache of the pooled mixture is keyed by the measurements, the biomarker names and the diagnosis
  codes. In 0.1 it was keyed by the measurements only, so a second call in the same process with the
  same measurements but different diagnoses reused a mixture initialized from other CN and AD
  participants.

## 0.1.1 — 2026-09-22

- Optional permutation within strata of diagnosis and number of observed measurements
  (`schemes=("diagnosis_count",)`; removed in 0.2.0).
- Fixed common proportions for CONCORD through the low-level function
  `concord.invariant.fit_invariant_orderings`, as an alternative to the minimum rule.
- Overview figure in the README.
- Examples: `examples/demo_true_ordering.py` (two groups with the same true ordering and different
  CN/MCI/AD proportions: orderings, bootstrap heatmaps and P values of separately fitted DEBM and
  CONCORD) and a demo on the choice of common proportions (now `examples/demo_common_proportions.py`).

## 0.1.0 — 2026-09-13

First public version.

- `concord.compare(df, group_column=...)`: one abnormality model fitted to the pooled sample,
  participants weighted to common diagnostic proportions, and each group's ordering estimated with the
  DEBM ranking consensus; unrestricted, within-diagnosis and pairwise within-diagnosis permutation
  tests with exact decisions under a fixed number of relabelings; composition table, effective sample
  sizes, bootstrap stability of the orderings and the difference between within-diagnosis and
  unrestricted P values.
- Models (0.1 names): `invariant_min` (CONCORD, default), `invariant_pooled`, `shared` (pooled-score
  DEBM), `standard` (separately fitted DEBM) and `saebm` (optional).
- Command-line entry point `concord`.
- Worker pool based on `concurrent.futures`: a worker that dies (for example, an unguarded script
  re-imported by a spawned process) raises a clear error instead of hanging.
- The simulations and analyses of the paper are kept in the companion repository `concord-ebm-paper`.
- Pinned to the unmodified `pyebm==2.0.3`; the package verifies the installed source before fitting.
