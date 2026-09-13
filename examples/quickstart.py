"""CONCORD quick start on a small synthetic cohort.

    python examples/quickstart.py            # about a minute on 4 cores at B=59

Three groups share one true event ordering but differ in their mix of CN / MCI / AD subjects, which is
exactly the situation in which per-group EBM orderings drift apart. Replace `df` by your own data frame
(columns PTID, Diagnosis, a group column, biomarker columns). The `if __name__ == "__main__"` guard is
required whenever workers > 1: the worker processes are spawned and re-import this file.

The ADNI-shaped simulator and the pre-registered simulation protocols of the paper live in the companion
repository https://github.com/aarontgao2023/concord-ebm-paper.
"""
import numpy as np
import pandas as pd

import concord


def synthetic_cohort(seed=0, events=8, composition=((0.75, 0.10, 0.15), (0.45, 0.30, 0.25), (0.20, 0.35, 0.45)),
                     sizes=(80, 200, 220), sigma=0.7):
    """Same true ordering (event 0 first) in every group; stage drawn from the group's composition."""
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


def main():
    df = synthetic_cohort()
    print(pd.crosstab(df["group"], df["Diagnosis"]))
    result = concord.compare(df, group_column="group", labels=("CN", "MCI", "AD"), estimator="invariant_min",
                             B=59, stability_resamples=10, seed=1, workers=4)
    print(result.summary())
    result.to_json("examples/output_quickstart.json")


if __name__ == "__main__":
    main()
