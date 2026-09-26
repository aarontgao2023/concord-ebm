"""CONCORD quick start on a small synthetic dataset.

    python examples/quickstart.py            # about a minute on 4 cores at B=59

Three groups share one true event ordering but differ in their proportions of CN, MCI and AD
participants, the situation in which orderings fitted separately to each group drift apart.
Replace `df` by your own data frame (one row per participant: PTID, Diagnosis, a group column and
the biomarker columns). The `if __name__ == "__main__":` guard is required whenever workers > 1:
the worker processes are spawned and re-import this file.

With three groups, compare() tests each pair of groups by pairwise within-diagnosis permutation
by default (with two groups, by within-diagnosis permutation); schemes=concord.SCHEMES adds
unrestricted permutation and within-diagnosis permutation of all groups. B=59 keeps the example
fast; use the default B=599 for real analyses. The code for the simulations, the ADNI and NACC
analyses, and the figures and tables of the paper is in
https://github.com/aarontgao2023/concord-ebm-paper.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import concord

OUTPUT = Path(__file__).resolve().parent / "output_quickstart.json"


def synthetic_cohort(seed=0, events=8, composition=((0.75, 0.10, 0.15), (0.45, 0.30, 0.25), (0.20, 0.35, 0.45)),
                     sizes=(80, 200, 220), sigma=0.7):
    """Same true ordering (event 0 first) in every group; diagnosis drawn from the group's proportions."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, (mix, n) in enumerate(zip(composition, sizes)):
        stage_of = {"CN": (0, 2), "MCI": (2, 6), "AD": (6, events + 1)}   # events already occurred
        for i in range(n):
            dx = rng.choice(["CN", "MCI", "AD"], p=mix)
            k = rng.integers(*stage_of[dx])
            values = rng.normal(0.0, sigma, events) + 2.0 * (np.arange(events) < k)
            rows.append({"PTID": f"g{g}_{i:04d}", "Diagnosis": dx, "group": g,
                         **{f"marker{j}": float(v) for j, v in enumerate(values)}})
    return pd.DataFrame(rows)


def _drop_local_paths(value):
    """Remove the installation path of pyebm, so that the saved example output is portable."""
    if isinstance(value, dict):
        return {k: _drop_local_paths(v) for k, v in value.items() if k != "pyebm_path"}
    if isinstance(value, list):
        return [_drop_local_paths(v) for v in value]
    return value


def main():
    df = synthetic_cohort()
    print(pd.crosstab(df["group"], df["Diagnosis"])[["CN", "MCI", "AD"]])

    # Common proportions by the minimum rule (the default of compare()); pass a result like this
    # as common_proportions=... to hold the proportions fixed, e.g. across several datasets.
    print("common proportions (minimum rule):", concord.common_proportions(df, group_column="group"))

    # default schemes: one pairwise within-diagnosis permutation test per pair of groups
    result = concord.compare(df, group_column="group", labels=("CN", "MCI", "AD"), estimator="concord",
                             B=59, stability_resamples=10, seed=1, workers=4)
    print(result.summary())
    OUTPUT.write_text(json.dumps(_drop_local_paths(result.to_dict()), indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
