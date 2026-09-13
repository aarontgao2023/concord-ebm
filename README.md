# CONCORD

**CO**mposition-**N**ormalised **C**onsensus for **ORD**ering comparison — a composition-invariant way to
compare event-based model (EBM) orderings between groups, with a permutation test whose reference
distribution matches the null actually being tested.

```python
import pandas as pd, concord

df = pd.read_csv("cohort.csv")        # PTID, Diagnosis (CN/MCI/AD), APOE, biomarker columns (NaN allowed)
result = concord.compare(df, group_column="APOE", B=599, workers=8)
print(result.summary())
result.to_json("result.json")
```

```
$ concord cohort.csv --group APOE --B 599 --workers 8 --output result.json
```

## Why

Orderings fitted separately to groups (genotypes, phenotypes, cohorts) are compared routinely. A per-group
discriminative EBM, however, estimates an ordering whose *target* depends on the group's diagnostic
composition: the measurement scale (a per-group mixture model) and the consensus (an average over the
group's own mix of disease stages) both move with the proportion of healthy, impaired and demented subjects.
Two groups with an identical disease process therefore yield different orderings, and the difference does
not shrink with sample size. A permutation test that exchanges labels among all subjects makes matters
worse: its reference groups all have the pooled composition.

CONCORD removes both dependences and tests against the right reference:

1. **One measurement model** — a single two-component mixture per biomarker, fitted to all subjects and
   used for every group (measurement invariance).
2. **One reference composition** — each group's consensus is a weighted average over its subjects with
   weights `pi_ref(d) / pi_g(d)`, so every group is averaged at the same stage composition (direct
   standardisation; the default reference is the sparsest composition every group can represent).
3. **Stratified permutation** — group labels are exchanged within diagnosis strata (all groups, or one pair
   at a time for localisation), which preserves each group's composition in every permuted data set.

In pre-registered simulations at the ADNI composition (1,000 cohorts per cell) the published procedure rejects
a true null in about 12 % of cohorts at a nominal 5 %; CONCORD is at 4 % with roughly ten times the power of
the standard estimator against a moderate alternative. Details are in the accompanying manuscript.

## What you get back

`concord.compare(...)` returns a `ComparisonResult` with

- the per-group orderings (biomarker names in estimated order) and a position table;
- pairwise normalised Kendall distances and their maximum;
- for each reference distribution — `unrestricted` (the published operator, for comparison), `diagnosis`
  (primary), `diagnosis_pair_*` (one per pair) — the Monte-Carlo *p*, its bounds under the planned budget, and
  the exact decision under the rule *p ≤ α / n_pairs* (pairs) and *p ≤ α* (maximum);
- diagnostics: the group × diagnosis composition table (counts, proportions, pooled and reference
  compositions, χ²), the effective sample size per group after standardisation, the stability of each
  ordering under group × diagnosis stratified resampling, and the stratified-minus-unrestricted *p*
  difference (large positive values flag composition-driven significance under the unrestricted test);
- provenance: package and `pyebm` versions, source hashes, the full specification.

`result.summary()` prints a compact report; `result.to_dict()` / `to_json()` serialise everything.

## Estimators and options

| `estimator=`       | description |
|--------------------|-------------|
| `invariant_min`    | pooled measurement model + consensus standardised to the sparsest common composition (default) |
| `invariant_pooled` | as above, standardised to the pooled composition |
| `shared`           | pooled measurement model only (diagnostic arm) |
| `standard`         | separately fitted co-init DEBM per group (the published practice) |
| `saebm`            | stage-aware EBM per group (`pip install concord-ebm[saebm]`; complete data only) |

Other arguments: `labels` (diagnosis labels in stage order; every `Diagnosis` value must be one of them),
`group_order`, `schemes`, `B` (permutations per reference; 599 by default), `alpha`, `rule` (`"le"` or
`"strict"`), `stop_when_decided` (exact early stopping), `stability_resamples`, `seed`, `workers`,
`fit_timeout_s`, `consensus` (`"repaired"` or `"original"` pyebm consensus arm).

Failed or unfinished fits never change the denominator: a decision is reported as `None` while it could still
go either way.

## Assumptions and when not to use it

Validity conditions on the diagnosis strata: within a stage, biomarker distributions and missingness must be
exchangeable across groups. Group-specific missingness that affects every modality is the documented failure
case (the test over-rejects); group-specific component densities inflate mildly. Check the composition table
and a group × modality availability test before interpreting a rejection. The pooled measurement model is a
modelling choice shared with other multi-group progression models (e.g. common control z-scores).

## Installation

```
pip install concord-ebm            # from PyPI once released
pip install -e ".[dev]"            # from a clone, with test dependencies
```

Requires Python ≥ 3.10 and the unmodified `pyebm==2.0.3`; the package verifies the installed `pyebm` source
against pinned hashes before fitting and refuses to run otherwise. The pooled mixture is label-invariant and
is cached per worker process, so a full comparison at *n* ≈ 1,000, *B* = 599, five references and 20
stability resamples takes about twelve minutes on 32 cores (about 4.7 core-hours); the standard estimator,
which has nothing to cache, takes about four times longer.

## Simulator

`concord.simulate` contains the ADNI-shaped data-generating process used in the paper (`resolve(name)` /
`simulate(cfg, seed)`), including the named cells (`REF_H0`, `IID_H0`, `BALANCED_H0`, `STAGE_H0`,
`PDF_H0`, `MISS_CSF_H0`, `MISS_ALL_H0`, `PWR_E4_K27`, …). `examples/quickstart.py` runs the whole
procedure on one simulated cohort.

## Tests

```
pytest -q
```

## Licence and provenance

GPL-3.0-or-later. CONCORD runs the unmodified `pyebm` 2.0.3 (Erasmus MC, GPL-3) through process-local
patches; `concord/likelihood.py` contains an objective adapted from `pyebm` and is therefore GPL-3 as
well. The `repaired` consensus arm changes one stopping rule of the upstream local search and is documented
in `concord/engine.py`.

## Citation

See `CITATION.cff`. Manuscript in preparation.
