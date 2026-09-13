"""CONCORD quick start on one simulated cohort (ADNI-shaped composition, same true ordering in all groups).

    python examples/quickstart.py            # ~2-3 minutes on 4 cores at B=59

Replace `df` by your own data frame: columns PTID, Diagnosis (CN/MCI/AD), a group column and biomarkers.
"""
import concord
from concord.simulate import resolve, simulate

df, truth = simulate(resolve("REF_H0"), seed=31900000)
print(f"{len(df)} subjects; groups {sorted(df.APOE.unique())}; true ordering identical in every group")

result = concord.compare(df, group_column="APOE", labels=("CN", "MCI", "AD"), estimator="invariant_min",
                         B=59, stability_resamples=10, seed=1, workers=4)
print(result.summary())
result.to_json("examples/output_quickstart.json")
