# CONCORD user manual

**CO**mposition-**N**ormalised **C**onsensus for **ORD**ering comparison — a composition-invariant way to
compare event-based model (EBM) orderings between groups, with a permutation test whose reference
distribution matches the null actually being tested. The [README](../README.md) has the short version;
this manual has everything else.

Contents: [Quick start](#quick-start) · [Why](#why) · [Input](#input) · [Options](#options) ·
[Output](#output) · [Reading the diagnostics](#reading-the-diagnostics) · [When not to use it](#assumptions-and-when-not-to-use-it) ·
[Performance](#performance) · [Simulator](#simulator) · [Command line](#command-line) · [Licence](#licence-and-provenance)

## Quick start

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

## Input

A `pandas.DataFrame` with

| column | content |
|---|---|
| `PTID` | subject identifier (any type; created if missing) |
| `Diagnosis` | one of `labels`, in stage order (default `("CN", "MCI", "AD")`: first = control, last = case). Every value must be in `labels` — `pyebm` would otherwise silently treat it as the middle stage |
| group column | the grouping variable (`group_column`, default `"APOE"`); any hashable values, ≥ 2 groups; encoded internally in `group_order` (default: sorted) |
| biomarkers | every other numeric column, or the list passed as `biomarkers`; `NaN` allowed; names must not contain `PTID`, `Diagnosis`, `EXAMDATE` or the group column's name as a substring (`pyebm` matches columns by substring) |

Orientation of the biomarkers (whether abnormal is high or low) is handled by `pyebm`'s mixture model as in a
standard DEBM fit.

## Options

| argument | default | meaning |
|---|---|---|
| `estimator` | `"invariant_min"` | see the table below |
| `schemes` | `("unrestricted", "diagnosis", "diagnosis_pair")` | reference distributions; `diagnosis_pair` expands to one scheme per pair (skipped with two groups) |
| `B` | `599` | permutations per reference |
| `alpha`, `rule` | `0.05`, `"le"` | rejection rule: pair *p* ≤ α / n_pairs, max *p* ≤ α (`"strict"` uses <) |
| `stop_when_decided` | `False` | stop a reference once every required decision is exact; *p* is then a bound |
| `stability_resamples` | `20` | group × diagnosis stratified bootstrap refits for the stability diagnostic (0 to skip) |
| `seed` | `0` | seeds every permutation and resample stream (reproducible) |
| `workers` | `1` | spawn-pool size; each worker caches its own pooled mixture |
| `fit_timeout_s` | `900` | per-fit wall-clock limit; a timed-out fit counts as failed (unknown), never as a non-exceedance |
| `consensus` | `"repaired"` | `pyebm` consensus arm: `"repaired"` continues the local search after an accepted swap; `"original"` is upstream's |
| `fast_likelihood` | `True` | numerically identical, faster GMM objective (see `concord/likelihood.py`) |

## Output

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

## Estimators

| `estimator=`       | description |
|--------------------|-------------|
| `invariant_min`    | pooled measurement model + consensus standardised to the sparsest common composition (default) |
| `invariant_pooled` | as above, standardised to the pooled composition |
| `shared`           | pooled measurement model only (diagnostic arm) |
| `standard`         | separately fitted co-init DEBM per group (the published practice) |
| `saebm`            | stage-aware EBM per group (`pip install concord-ebm[saebm]`; complete data only) |

Failed or unfinished fits never change the denominator: a decision is reported as `None` while it could still
go either way.

## Reading the diagnostics

- **Composition table** (`result.composition`): counts and proportions per group × stage, the pooled and the
  reference composition, and a χ² test of group × stage association. If compositions differ (they usually do),
  the unrestricted *p*-values are not valid and are reported only for comparison.
- **Effective sample size** (`result.effective_sample_size`): `(Σw)² / Σw²` per group after standardisation.
  A group whose ESS falls far below its size is being asked to represent stages it barely has; consider
  `estimator="invariant_min"` (the default; lowest weight variance) or collapsing stages.
- **Stability** (`result.stability`): mean Kendall distance between each group's ordering and its refits on
  stratified resamples, and the per-event position SD. Observed between-group distances inside this noise
  band are not evidence of a difference regardless of *p*.
- **Paired difference** (`result.paired_difference`): `p(diagnosis) − p(unrestricted)` per pair. A large
  positive value means the unrestricted test called a difference that disappears once composition is held
  fixed — the signature of composition-driven significance.
- **Decisions** (`result.tests[scheme]["pairs"][pair]["reject"]`): `True` / `False` / `None`. `None` means the
  budget was not exhausted and the answer could still go either way; the *p* bounds say how far.

## Assumptions and when not to use it

Validity conditions on the diagnosis strata: within a stage, biomarker distributions and missingness must be
exchangeable across groups. Group-specific missingness that affects every modality is the documented failure
case (the test over-rejects); group-specific component densities inflate mildly. Check the composition table
and a group × modality availability test before interpreting a rejection. The pooled measurement model is a
modelling choice shared with other multi-group progression models (e.g. common control z-scores).

## Performance

The pooled mixture is label-invariant and
is cached per worker process, so a full comparison at *n* ≈ 1,000, *B* = 599, five references and 20
stability resamples takes about twelve minutes on 32 cores (about 4.7 core-hours); the standard estimator,
which has nothing to cache, takes about four times longer.

## Simulator

`concord.simulate` contains the ADNI-shaped data-generating process used in the paper (`resolve(name)` /
`simulate(cfg, seed)`), including the named cells (`REF_H0`, `IID_H0`, `BALANCED_H0`, `STAGE_H0`,
`PDF_H0`, `MISS_CSF_H0`, `MISS_ALL_H0`, `PWR_E4_K27`, …). `examples/quickstart.py` runs the whole
procedure on one simulated cohort.

## Command line

```
concord INPUT.csv [--group APOE] [--labels CN,MCI,AD] [--biomarkers a,b,c] [--estimator invariant_min]
        [--schemes unrestricted,diagnosis,diagnosis_pair] [--B 599] [--alpha 0.05] [--stop-when-decided]
        [--stability 20] [--seed 0] [--workers 1] [--fit-timeout 900] [--no-fast-likelihood]
        [--output result.json] [--quiet]
```

Prints the summary; exit status 1 if the observed fit failed. Progress lines go to stderr.

## Development

```
git clone https://github.com/aarontgao2023/concord-ebm && cd concord-ebm
pip install -e ".[dev]"
pytest -q          # 29 tests, ~30 s
```

## Licence and provenance

GPL-3.0-or-later. CONCORD runs the unmodified `pyebm` 2.0.3 (Erasmus MC, GPL-3) through process-local
patches; `concord/likelihood.py` contains an objective adapted from `pyebm` and is therefore GPL-3 as
well. The `repaired` consensus arm changes one stopping rule of the upstream local search and is documented
in `concord/engine.py`.

