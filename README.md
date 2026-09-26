# CONCORD

**Composition-normalized consensus for ordering comparison**

CONCORD compares the event orderings of two or more patient groups, such as *APOE* genotypes, whose
diagnostic composition differs. First, it fits one abnormality model for each biomarker (a
two-component Gaussian mixture) to the pooled sample of the compared groups, so that the same
measurement receives the same probability of being abnormal in every group. Second, it weights
participants by diagnosis so that CN, MCI and AD participants contribute in the same proportions to
every group; a participant's weight is the common proportion of their diagnosis divided by its
proportion in their own group. Third, it estimates each group's ordering from the weighted abnormality
probabilities with the DEBM ranking consensus. The difference between two groups is the normalized
Kendall distance, the proportion of event pairs that their orderings place in opposite order, and it
is tested by permuting group labels within each diagnosis. Two groups with the same biomarker
distributions and the same pattern of missing measurements within each diagnosis then have the same
best-fitting ordering, whatever their composition.

![Overview of CONCORD](docs/figures/concord_mechanism.png)

**a** Separately fitted DEBM (gray boxes) and CONCORD (blue boxes). CONCORD fits one abnormality model
to the pooled sample, weights participants so that every group has the same CN/MCI/AD proportions
(marker area shows the weight) and estimates each group's ordering from the weighted abnormality
probabilities. **b** Group differences are tested by permuting group labels within diagnosis.
**a**,**b** are schematic; the proportions and weights shown are illustrative (Fig. 1a,b of the
manuscript).

## Install

```bash
pip install git+https://github.com/aarontgao2023/concord-ebm
```

Python 3.10 or later. The unmodified `pyebm==2.0.3` is installed as a dependency and its source is
verified before every fit. The stage-aware EBM needs the optional extra
`pip install "concord-ebm[saebm] @ git+https://github.com/aarontgao2023/concord-ebm"` (`pysaebm==7.7.7`).

## Use

```python
import pandas as pd
import concord

if __name__ == "__main__":      # required with workers > 1: the workers are spawned processes
    df = pd.read_csv("cohort.csv")   # one row per participant: PTID, Diagnosis (CN/MCI/AD), APOE, biomarkers
    result = concord.compare(df, group_column="APOE", B=599, workers=8)
    print(result.summary())
    result.to_json("result.json")
```

By default, group labels are permuted within diagnosis: two groups are compared by within-diagnosis
permutation, and with three or more groups each pair is compared by pairwise within-diagnosis
permutation. Unrestricted permutation runs only when requested (`schemes=`).

By default the common proportions are the smallest proportion of each diagnosis across the compared
groups, rescaled to sum to one. They can also be fixed. To compare several datasets at the same
proportions, compute them with `concord.common_proportions(...)` over all their groups. In the paper's
*APOE* analyses, this rule was applied jointly to the six cohort-by-genotype groups of ADNI and NACC,
which gave 48.9% CN, 28.6% MCI and 22.5% AD in both cohorts, and each pair of genotypes was compared by
pairwise within-diagnosis permutation:

```python
props = concord.common_proportions(adni, nacc, group_column="APOE")   # minimum rule over the six groups
result = concord.compare(adni, group_column="APOE", common_proportions=props,
                         schemes=("pairwise_within_diagnosis",), B=599,
                         stability_resamples=200, workers=8)
```

The result holds each group's ordering, the normalized Kendall distance between every pair of groups,
and for every comparison the permutation P value, P = (1 + E)/(B + 1) with E the number of relabelings
whose distance is at least the observed one (P = (1 + E)/600 with B = 599), the Bonferroni decision at
0.05/m for m comparisons and the adjusted P value min(1, m·P). It also reports the group-by-diagnosis
counts, the common proportions, the weights, the effective sample sizes and the stability of each
ordering under bootstrap refits. From the shell, fixed proportions are given as numbers; the paper's
proportions, rounded to five decimals, are 0.48926, 0.28593 and 0.22481:

```bash
concord cohort.csv --group APOE --common-proportions 0.48926,0.28593,0.22481 \
        --schemes pairwise_within_diagnosis --B 599 --workers 8 --output result.json
```

## Models

| `estimator=` | model |
|---|---|
| `concord` (default) | CONCORD |
| `pooled_score` | pooled-score DEBM: CONCORD without the weighting step |
| `separate` | separately fitted DEBM (pyebm 2.0.3) |
| `saebm` | stage-aware EBM (SA-EBM, optional `pysaebm` 7.7.7; complete measurements only) |

The likelihood EBM comparison of the paper is in
[concord-ebm-paper](https://github.com/aarontgao2023/concord-ebm-paper).

## Read more

- [User manual](docs/USER_MANUAL.md): the method, every option and output, how to reproduce the
  paper's settings, performance.
- [concord-ebm-paper](https://github.com/aarontgao2023/concord-ebm-paper): the code used for the
  simulations, the ADNI and NACC analyses, and the figures and tables of the paper, including the
  likelihood EBM comparison.
- [Changelog](CHANGELOG.md)

## Cite

Gao T, Liu H, Yan J. Comparing biomarker orderings between patient groups with different diagnostic
composition. Manuscript in preparation. See [`CITATION.cff`](CITATION.cff).

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later). CONCORD runs the unmodified `pyebm`
2.0.3, licensed under the GNU GPL v3, and adapts one function from it (`concord/likelihood.py`, GNU
GPL v3).
