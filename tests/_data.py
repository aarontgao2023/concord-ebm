"""Toy data set shared by the tests: three groups with identical diagnostic composition."""
import numpy as np
import pandas as pd


def small_dataset():
    rng = np.random.default_rng(7184)
    rows = []
    for group in range(3):
        for diagnosis, stages in (("CN", [0] * 8),
                                  ("MCI", [1, 2, 3, 1, 2, 3, 1, 3]),
                                  ("AD", [4] * 8)):
            for stage in stages:
                values = rng.normal(0.0, 0.65, 4) + 2.5 * (np.arange(4) < stage)
                rows.append({"PTID": str(len(rows)), "Diagnosis": diagnosis,
                             "APOE": group,
                             **{f"b{i}": float(value) for i, value in enumerate(values)}})
    return pd.DataFrame(rows)


def compare_in_worker(workers):
    """Run concord.compare inside a worker process (used by the worker-guard test)."""
    import concord
    try:
        result = concord.compare(small_dataset(), B=0, stability_resamples=0, schemes=("within_diagnosis",),
                                 workers=workers, verbose=False)
    except RuntimeError as exc:
        return f"RuntimeError: {exc}"
    return result.status
