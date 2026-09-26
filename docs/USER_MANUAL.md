# CONCORD user manual

CONCORD (composition-normalized consensus for ordering comparison) compares the event orderings of two
or more patient groups whose diagnostic composition differs. The [README](../README.md) has the short
version; this manual describes the method, every option and output, and the settings used in the paper.

Contents: [Quick start](#quick-start) · [Why](#why) · [Method](#method) · [Input](#input) ·
[Options](#options) · [Output](#output) · [Reading the output](#reading-the-output) ·
[Reproducing the paper's settings](#reproducing-the-papers-settings) · [Performance](#performance) ·
[Command line](#command-line) · [Names used in the code](#names-used-in-the-code) ·
[Development](#development) · [License](#license-and-provenance)

## Quick start

```python
import pandas as pd
import concord

if __name__ == "__main__":      # required with workers > 1: the workers are spawned processes
    df = pd.read_csv("cohort.csv")   # PTID, Diagnosis (CN/MCI/AD), APOE, biomarker columns (NaN allowed)
    result = concord.compare(df, group_column="APOE", B=599, workers=8)   # pairwise within-diagnosis permutation
    print(result.summary())
    result.to_json("result.json")
```

Fixed common proportions and pairwise within-diagnosis permutation, as in the paper's *APOE* analyses.
Compute the proportions with `concord.common_proportions(...)` over the six cohort-by-genotype groups:

```python
props = concord.common_proportions(adni, nacc, group_column="APOE")   # minimum rule over all six groups
result = concord.compare(adni, group_column="APOE", common_proportions=props,
                         schemes=("pairwise_within_diagnosis",), B=599, stability_resamples=200, workers=8)
```

On the command line the proportions are given as numbers; the paper's proportions, rounded to five
decimals, are 0.48926, 0.28593 and 0.22481:

```
$ concord cohort.csv --group APOE --common-proportions 0.48926,0.28593,0.22481 \
          --schemes pairwise_within_diagnosis --B 599 --workers 8 --output result.json
```

`examples/quickstart.py` runs a complete comparison on synthetic data in under a minute.

## Why

Event-based models are increasingly used to compare patient groups, such as genotypes, and differences
between the estimated orderings are read as differences in disease course. Groups that differ only in
their proportions of cognitively normal (CN), mildly impaired (MCI) and demented (AD) participants,
however, receive different orderings. With complete measurements, an estimated ordering follows the
average abnormality of the participants in a group. For participants who have not reached an event, a
well-separated event receives an abnormality probability close to zero, whereas a poorly separated
event keeps an intermediate probability. In a CN-heavy group, well-separated events therefore have
lower averages and are placed later; in an AD-heavy group, the same events are placed earlier.

In the paper's simulations, three groups shared one true sequence of 14 events and had the CN/MCI/AD
counts of a published ADNI *APOE* comparison (75, 411 and 485 participants):

- With complete measurements, composition increased the mean-position gap between the CN-heavy and
  AD-heavy groups by 1.14 positions per event for separately fitted DEBM and by 0.039 for CONCORD.
- When the groups shared one ordering and differed only in composition, unrestricted permutation of
  separately fitted DEBM rejected at least one of the three group comparisons in 13.2% of 1,000
  datasets; CONCORD with within-diagnosis permutation rejected in 4.2%.
- In the four conditions with large ordering changes in one group (27 of the 91 event pairs reversed,
  or one event moved 13 positions; Fig. 3c of the paper), CONCORD detected the change 2.1 to 6.6 times
  as often as separately fitted DEBM on the same datasets.

In ADNI and NACC, dividing the same participants into groups of different composition shifted events
in the orderings of likelihood and discriminative EBMs, whereas no event moved by more than 0.085
positions with CONCORD.

## Method

This section follows the Methods and Supplementary Note 1 of the paper.

**Abnormality model.** For each event *i* = 1, …, *K*, DEBM fits a two-component Gaussian mixture to
the biomarker measurements and assigns participant *j* the posterior probability *q<sub>ji</sub>* that
the measurement belongs to the abnormal component. The two components are initialized from CN and AD
participants and then estimated from all participants, including those with MCI. CONCORD fits the
mixtures once, without weights, to all participants of the compared groups, so every group receives
the same abnormality probabilities for the same measurements.

**Ranking loss.** For an ordering *S* of the events, the ranking loss of participant *j* is

  ℓ(*S*; *q<sub>j</sub>*, *M<sub>j</sub>*) = Σ<sub>*a* before *b* in *S*</sub> *M<sub>ja</sub>* *M<sub>jb</sub>* [*q<sub>jb</sub>* − *q<sub>ja</sub>*]<sub>+</sub>,

where *M<sub>ji</sub>* indicates that the measurement is observed and [*x*]<sub>+</sub> = max(*x*, 0).
An ordering is penalized when it places *a* before *b* although the participant's abnormality is higher
for *b*, and only jointly observed event pairs contribute. A group's ordering minimizes the (weighted)
mean loss over its participants. With complete measurements, the minimizer places events in order of
decreasing (weighted) mean abnormality in the group.

**Weights.** Participant *j* with diagnosis *D<sub>j</sub>* in group *g* receives the weight

  *w<sub>j</sub>* = π<sub>\*</sub>(*D<sub>j</sub>*) / π̂<sub>*g*</sub>(*D<sub>j</sub>*),  π̂<sub>*g*</sub>(*d*) = *n<sub>gd</sub>* / *n<sub>g</sub>*,

where *n<sub>gd</sub>* is the number of participants with diagnosis *d* in group *g* and π<sub>\*</sub>(*d*)
are the common proportions. The weights average one within each group, and after weighting every
diagnosis accounts for the same share π<sub>\*</sub>(*d*) of every group. If two groups have the same
joint distribution of measurements and observation patterns within each diagnosis, their expected
weighted losses are equal for every ordering, and so are their best-fitting orderings. This mirrors
direct standardization of rates to a common population. Pooled-score DEBM is the same procedure with
all weights equal to one.

**Common proportions.** Common proportions must be positive only for diagnoses present in every
compared group. By default (`common_proportions=None` or `"min"`) CONCORD uses the smallest proportion
of each diagnosis across the compared groups, rescaled to sum to one,
π<sub>\*</sub>(*d*) = min<sub>*g*</sub> π̂<sub>*g*</sub>(*d*) / Σ<sub>*d′*</sub> min<sub>*g*</sub> π̂<sub>*g*</sub>(*d′*).
This choice gives the smallest possible largest weight, 1/κ with κ = Σ<sub>*d*</sub> min<sub>*g*</sub> π̂<sub>*g*</sub>(*d*).
Fixed proportions can be supplied instead; every diagnosis with a positive proportion must then be
present in every group. A diagnosis left out of a mapping such as `{"CN": 0.6, "AD": 0.4}` gets
proportion 0, and a proportion of 0 gives the participants with that diagnosis weight 0: they still
enter the pooled abnormality model but not the orderings. The minimum rule gives a proportion of 0 to
a diagnosis that is absent from one of the groups, with the same effect. `concord.common_proportions()`
applies the minimum rule jointly to any number of groups, for example to the groups of several cohorts;
use it to compute fixed proportions rather than typing rounded values. In the paper, the rule was
applied to each simulated dataset and gave 48.6% CN, 17.1% MCI and 34.3% AD for the design counts
(largest weight 2.14); for ADNI and NACC it was applied jointly to the six cohort-by-genotype groups,
which gave 48.9% CN, 28.6% MCI and 22.5% AD in both cohorts (0.48926, 0.28593 and 0.22481, rounded;
largest weight 2.73). Weighting reduces each group's effective sample size to
(Σ*w<sub>j</sub>*)² / Σ*w<sub>j</sub>*².

**Search.** Orderings are found with the adjacent-swap search of pyebm 2.0.3, started from events
sorted by decreasing estimated abnormal-component proportion. At each step the best adjacent swap is
accepted if it strictly lowers the loss, and the search continues until no swap lowers the loss
(`search="continued"`, the default); the pyebm 2.0.3 search stops after the first accepted swap
(`search="pyebm"`). Because an adjacent swap changes only the contribution of the swapped pair, the
continued search returns the exact minimizer when measurements are complete. With missing measurements
it stops at an ordering that no adjacent swap improves (or after pyebm's budget of 10,000 loss
evaluations).

**Distance.** The difference between the orderings of two groups is the normalized Kendall distance,
the proportion of event pairs that the two orderings place in opposite order (0 for identical
orderings, 1 for reversed ones).

**Permutation tests.** Each scheme relabels only the group column:

- *Unrestricted permutation* exchanges group labels among all participants of the compared groups.
- *Within-diagnosis permutation* exchanges the labels of all compared groups separately among CN, MCI
  and AD participants, so every relabeled group keeps its numbers of CN, MCI and AD participants and
  therefore its weights.
- *Pairwise within-diagnosis permutation* exchanges only the labels of the two compared groups within
  diagnosis; the other groups keep their labels and remain in the pooled abnormality model. It gives
  one test per pair of groups. With two groups it is within-diagnosis permutation.

By default (`schemes=None`) two groups are compared by within-diagnosis permutation, and three or more
groups by pairwise within-diagnosis permutation, as in the paper's *APOE* analyses. Unrestricted
permutation runs only when it is requested, for example with `schemes=concord.SCHEMES` (all three
schemes); `summary()` then notes that it does not keep each group's diagnostic counts.

After each relabeling, every label-dependent step is repeated: separately fitted mixtures, weights
(and, under unrestricted permutation with the minimum rule, the common proportions) and group
orderings. Fixed common proportions stay fixed. The pooled mixture depends on the measurements and
the diagnoses (its components are initialized from the CN and AD participants), not on the group
labels, so it is fitted once per process and reused after every relabeling of the same data.

Each comparison uses *B* = 599 relabelings by default. The one-sided permutation P value is
P = (1 + *E*)/(*B* + 1), where *E* is the number of relabelings whose distance is at least the
observed distance (ties count). With *m* group comparisons, each is tested at 0.05/*m* (Bonferroni) and
the adjusted P value is min(1, *m*·P). With three comparisons and *B* = 599, a comparison is rejected
when *E* ≤ 9, and the smallest attainable adjusted P value is 0.005; with two groups (one comparison),
it is rejected at P ≤ 0.05, that is *E* ≤ 29. With `stop_when_decided=True` a scheme stops as soon as
no further relabeling can change any of its decisions (the paper's simulations used this; the *APOE*
comparisons used all 599 relabelings).

**Failed relabelings.** When all *B* relabelings were fitted, `p`, `p_adjusted` and `reject` agree.
When some fits failed or timed out, or were not run because of `stop_when_decided`, they are treated
differently:

- `p` and `p_adjusted` use the completed relabelings only: P = (1 + *E*)/(*c* + 1), where *c* is the
  number of completed relabelings.
- `p_bounds` and `p_adjusted_bounds` refer to the planned *B*. The lower bound counts every missing
  relabeling as below the observed distance, the upper bound counts it as an exceedance.
- `reject` also refers to the planned *B*. It is `True` only if the comparison is rejected even when
  every missing relabeling counts as an exceedance, and `False` only if it is not rejected even when
  none does; otherwise it is `None` (undetermined).

A `p` at or below the threshold can therefore come with `reject` equal to `None`; the decision is
`reject`.

**Assumption.** Within-diagnosis permutation assumes that, within each diagnosis, the observed
biomarker profiles of the compared groups, including which measurements are missing, are
exchangeable.

## Input

A `pandas.DataFrame` with one row per participant:

| column | content |
|---|---|
| `PTID` | participant identifier (optional; created if missing) |
| `Diagnosis` | one of `labels`, in disease order (default `("CN", "MCI", "AD")`: the first label is CN, the last AD). Every value must be in `labels`, because pyebm would otherwise treat it as the middle diagnosis. Labels between the first and the last are treated as one middle diagnosis |
| group column | the grouping variable (`group_column`, default `"APOE"`): any values, at least two groups, no missing values; groups are ordered by `group_order` (default: sorted) |
| biomarkers | every other column, or the columns listed in `biomarkers`; numeric, `NaN` marks a missing measurement; at least two. Names must not contain `PTID`, `Diagnosis`, `EXAMDATE` or the group column's name, because pyebm selects columns by substring |

Whether abnormal values are high or low is learned by the mixture model, as in any DEBM fit. CONCORD
needs a stage-related stratum, such as diagnosis, that is recorded for every participant and
represented in every group.

## Options

`concord.compare(df, ...)`:

| argument | default | meaning |
|---|---|---|
| `group_column` | `"APOE"` | column that defines the groups |
| `labels` | `("CN", "MCI", "AD")` | diagnosis labels in disease order |
| `biomarkers` | all other columns | biomarker columns |
| `group_order` | sorted values | order of the groups in the output |
| `estimator` | `"concord"` | `"concord"` (CONCORD), `"pooled_score"` (pooled-score DEBM), `"separate"` (separately fitted DEBM, pyebm 2.0.3) or `"saebm"` (SA-EBM; see below) |
| `common_proportions` | `None` | CONCORD only. `None` or `"min"`: the minimum rule over the compared groups. Fixed: a mapping such as `{"CN": 0.5, "MCI": 0.25, "AD": 0.25}` or a sequence in label order; nonnegative, summing to one (within 0.001; percentages are rejected); held fixed in every relabeling and bootstrap refit. A diagnosis left out of a mapping gets proportion 0, and participants with a diagnosis of proportion 0 get weight 0. Compute fixed proportions with `concord.common_proportions()` |
| `search` | `"continued"` | ranking-consensus search: `"continued"` or `"pyebm"` |
| `schemes` | `None` | any of `"unrestricted"`, `"within_diagnosis"`, `"pairwise_within_diagnosis"`. `None`: within-diagnosis permutation for two groups, pairwise within-diagnosis permutation for three or more |
| `B` | `599` | relabelings per test |
| `alpha`, `rule` | `0.05`, `"le"` | a comparison is rejected when P ≤ `alpha`/*m* (`"strict"`: P < `alpha`/*m*) |
| `stop_when_decided` | `False` | stop a test once no further relabeling can change its decisions; P is then reported with its bounds over the planned *B* |
| `stability_resamples` | `20` | bootstrap refits within group-by-diagnosis cells (0 skips them) |
| `seed` | `0` | seed of every relabeling and bootstrap resample; results are reproducible from it |
| `workers` | `1` | worker processes. With `workers > 1` the fits run in a pool of spawned processes, so in a script the call must sit under `if __name__ == "__main__":` (CONCORD raises a clear error otherwise) |
| `fit_timeout_s` | `900` | time limit per fit in seconds; a timed-out fit counts as failed, never as a relabeling below the observed distance |
| `fast_likelihood` | `True` | evaluate pyebm's mixture objective with `concord/likelihood.py` (same objective, faster) |
| `saebm_iterations`, `saebm_burn_in` | `10000`, `2500` | SA-EBM sampler settings |
| `verbose`, `progress_every` | `True`, `10.0` | progress messages on stderr, at most every `progress_every` seconds |

SA-EBM (`estimator="saebm"`) needs `pip install "concord-ebm[saebm]"` (`pysaebm==7.7.7`). It is fitted
in each group with conjugate priors and CN participants as non-diseased, reports the ordering with the
highest likelihood and requires complete measurements.

Earlier names are still accepted: `estimator="invariant_min"` (CONCORD), `"shared"` (pooled-score
DEBM), `"standard"` (separately fitted DEBM); `consensus=` for `search=`, with `"repaired"`
(continued) and `"original"` (pyebm); schemes `"diagnosis"` (within-diagnosis) and `"diagnosis_pair"`
(pairwise within-diagnosis). `estimator="invariant_pooled"` weights to the diagnostic proportions of
the pooled sample; it was not used in the paper.

`concord.common_proportions(*groups, labels=("CN", "MCI", "AD"), group_column=None)` returns
`{label: proportion}` by the minimum rule over all groups it is given. Each argument is a data frame
with a `Diagnosis` column (one group per value of `group_column`, or the whole frame as one group), a
count table with one column per label, the counts of one group as a mapping or sequence, or a
two-dimensional array of counts.

## Output

`concord.compare()` returns a `ComparisonResult`; `summary()` prints a report, `to_dict()` and
`to_json(path)` serialize everything.

| field | content |
|---|---|
| `spec` | every setting, with the model, search and schemes under their current names, plus `B`, the test names and `stability_resamples` |
| `groups`, `biomarkers`, `status` | group values, biomarker names, `"ok"` or the reason the observed fit failed |
| `orderings` | group → biomarker names, earliest event first |
| `positions` | biomarker → {group: position}, 0 = earliest |
| `distances` | normalized Kendall distance for every pair of groups |
| `tests` | test name (`unrestricted`, `within_diagnosis`, `pairwise_within_diagnosis_<a>-<b>`) → scheme, relabelings planned, completed and failed, and per comparison: `exceedances` (*E*), `p` and `p_adjusted` (over the completed relabelings), `p_bounds` and `p_adjusted_bounds` (over the planned *B*), `reject` (`True`/`False`/`None`, over the planned *B*) and `threshold` |
| `decisions` | the Bonferroni decisions by scheme and comparison (`reject`), whether any comparison was rejected under each scheme (`reject_any_pair`), `alpha`, *m*, the per-comparison threshold and the rule |
| `composition` | group-by-diagnosis counts and proportions, pooled proportions, the minimum-rule proportions (`min_rule`), a χ² test of group-by-diagnosis association, and notes (e.g. a group without participants of some diagnosis) |
| `common_proportions` | the common proportions used in the observed fit (`None` for unweighted models) |
| `weights`, `largest_weight` | weight of each diagnosis in each group (`None` where a group has no participants of that diagnosis; 1.0 for unweighted models) and the largest weight |
| `effective_sample_size` | (Σ*w*)²/Σ*w*² per group (the group size for unweighted models) |
| `stability` | bootstrap refits within group-by-diagnosis cells: per group the mean and SD of the Kendall distance to the observed ordering and the SD of each event's position, and the mean distance between the refitted orderings of every pair of groups |
| `paired_difference` | within-diagnosis minus unrestricted P for each comparison (when both tests ran) |
| `fits`, `timing`, `provenance`, `observed_diagnostics` | fit counts, run times, package and dependency versions and source hashes, and the diagnostics of the observed fit |

## Reading the output

- **Decisions.** `result.tests[name]["pairs"][pair]["reject"]` is `True`, `False` or `None`. `None`
  means that failed or unfinished relabelings could still change the decision; `p_bounds` gives the
  range of P over their possible outcomes. `p` is computed over the completed relabelings only (see
  [Failed relabelings](#method)).
- **Composition.** `result.composition` shows how much the groups differ in composition. When they
  differ, unrestricted permutation compares each group with relabeled groups of the pooled composition;
  within-diagnosis permutation keeps every group's composition.
- **Weights and effective sample size.** A group whose effective sample size falls far below its size
  has few participants of a diagnosis that the common proportions weight heavily. The minimum rule
  gives the smallest possible largest weight.
- **Stability.** The mean distance between each group's ordering and its bootstrap refits, and the SD
  of each event's position, show how precisely each ordering is estimated.
- **Paired difference.** A large positive within-diagnosis minus unrestricted P shows a comparison that
  unrestricted permutation calls significant but within-diagnosis permutation does not.

## Reproducing the paper's settings

The code, seeds and analysis scripts of the paper are in
[concord-ebm-paper](https://github.com/aarontgao2023/concord-ebm-paper). The settings map onto this
package as follows.

- ***APOE* analyses (ADNI, 1,203 participants; NACC, 1,584).** CONCORD at fixed common proportions of
  48.9% CN, 28.6% MCI and 22.5% AD, computed with `concord.common_proportions(adni, nacc,
  group_column="APOE")` over the six cohort-by-genotype groups of the analysis data (rounded to five
  decimals: 0.48926, 0.28593 and 0.22481); `schemes=("pairwise_within_diagnosis",)` (the default for
  three groups), `B=599` with all relabelings (`stop_when_decided=False`),
  Bonferroni over the three comparisons; `stability_resamples=200` for the 200 bootstrap refits within
  the nine genotype-by-diagnosis cells, with the pooled mixture and all three orderings refitted and the
  common proportions held fixed. CSF measurements were adjusted for age and sex, and cognitive scores
  for age, sex and education, by linear regression in all CN participants before the comparisons; this
  adjustment is part of the analysis code, not of the package. The sensitivity analysis without p-tau
  is the same call with p-tau left out of `biomarkers`. The package reports position SDs and distances
  across refits; the probabilities that one event precedes another across refits are computed by the
  analysis code.
- **Simulations.** The minimum rule applied to each dataset (the default), the permutation schemes
  shown in each figure, `B=599` and `stop_when_decided=True`.
- **Models.** The continued search for all DEBM variants, except that separately fitted DEBM used the
  pyebm search (`search="pyebm"`) in four settings of Fig. 4e. Separately fitted DEBM stops the mixture
  fitting when the mean change in mixing proportions falls below 0.01 in the first group (pyebm's
  default, used in Fig. 2, Fig. 3a and Fig. 4e); the other analyses of the paper required every group's
  mixtures to meet this criterion, which is implemented in the analysis code. SA-EBM used pysaebm 7.7.7
  with conjugate priors, 10,000 iterations and 2,500 burn-in iterations (the package defaults). The
  likelihood EBM used kde_ebm 0.0.3 and is run from the paper repository.
- **Software.** Python 3.12, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.3, scikit-learn 1.5.2 and pyebm
  2.0.3.

## Performance

A comparison runs one observed fit, *B* fits per test and one fit per bootstrap refit. With three
groups the default schemes give three tests (pairwise within-diagnosis permutation for each pair), that
is 3 × 599 = 1,797 relabeling fits; the *APOE* setting above adds 200 refits. Requesting all three
schemes (`schemes=concord.SCHEMES`) gives five tests (unrestricted, within-diagnosis and three pairwise
within-diagnosis), that is 5 × 599 = 2,995 relabeling fits. With two groups the default is one test,
599 relabeling fits. The fits are independent and run in parallel with `workers`. For CONCORD and
pooled-score DEBM, the pooled mixture does not depend on the group labels and is cached in every
worker process, so a relabeling refits only the weights and orderings; bootstrap refits change the
data and refit the mixture. Separately fitted DEBM refits its mixtures after every relabeling and is
slower. `examples/quickstart.py` (500 participants, 8 biomarkers, *B* = 59, 188 fits) runs in under a
minute on four cores.

## Command line

```
concord INPUT.csv [--group APOE] [--labels CN,MCI,AD] [--biomarkers a,b,c] [--estimator concord]
        [--common-proportions min | 0.5,0.25,0.25 | CN=0.5,MCI=0.25,AD=0.25]
        [--search continued] [--schemes within_diagnosis | pairwise_within_diagnosis | ...]
        [--B 599] [--alpha 0.05] [--stop-when-decided] [--stability 20] [--seed 0] [--workers 1]
        [--fit-timeout 900] [--no-fast-likelihood] [--saebm-iterations 10000] [--saebm-burn-in 2500]
        [--output result.json] [--quiet]
```

Prints the summary and exits with status 1 if the observed fit failed. Progress messages go to stderr.
Without `--schemes`, two groups are compared by within-diagnosis permutation and three or more by
pairwise within-diagnosis permutation. Fixed proportions are best computed with
`concord.common_proportions(...)`; for the paper's six cohort-by-genotype groups they are
0.48926,0.28593,0.22481 (rounded). Invalid options, including `--common-proportions` with a model other
than CONCORD, are reported before the input file is read.

## Names used in the code

Some internal identifiers keep the names of earlier versions and are recorded in the fit diagnostics:
engine mode `repaired` is the continued search and `original` the pyebm search; the weighting variant
`invariant_min` is CONCORD, `shared` is pooled-score DEBM and `invariant_pooled` weights to the pooled
proportions. The random streams of the relabelings keep their earlier names, so a given seed gives the
same relabelings as version 0.1.

## Development

```
git clone https://github.com/aarontgao2023/concord-ebm && cd concord-ebm
pip install -e ".[dev]"
pytest -q          # 38 tests
```

## License and provenance

GNU General Public License v3.0 or later (GPL-3.0-or-later). CONCORD runs the unmodified `pyebm` 2.0.3
(Erasmus MC, licensed under the GNU GPL v3) through temporary, process-local patches and verifies its
source before every fit. `concord/likelihood.py` contains an objective adapted from `pyebm` and is
licensed under the GNU GPL v3 (`GPL-3.0-only`). The continued
search changes one stopping rule of pyebm's adjacent-swap search and is documented in
`concord/engine.py`.
