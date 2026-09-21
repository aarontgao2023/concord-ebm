"""Minimal, fully visible demonstration of the composition problem and of CONCORD.

    python examples/demo_true_ordering.py            # ~3-5 minutes on 4 cores

1. Simulate eight biomarkers A..H that turn abnormal in that order along a latent disease
   time (sigmoid trajectories, figure 1).  Diagnosis (CN/MCI/AD) follows the latent time.
2. Split the cohort into two groups with the SAME true ordering but a different diagnostic
   mix: "healthy-heavy" and "impaired-heavy".
3. Fit the standard per-group DEBM and CONCORD on 30 stratified bootstrap resamples of the
   cohort and draw the classic DEBM positional-variance heatmaps (biomarker x position),
   one per group and estimator (figure 2).  A correct, comparable estimate is a diagonal.
4. Run the packaged comparison (concord.compare) with both estimators and print the
   pairwise distance and the permutation p-values under the published shuffle and the
   within-diagnosis shuffle.

Outputs are written to examples/output_demo/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import concord
from concord.core import CompareSpec, fit_once, kendall_distance, prepare_data

OUT = Path(__file__).resolve().parent / "output_demo"
EVENTS = list("ABCDEFGH")                       # true ordering: A first, H last
ONSET = np.linspace(3.0, 17.0, len(EVENTS))     # latent time at which each biomarker crosses 0.5
SLOPE = 0.9
T_MAX = 20.0


# ------------------------------------------------------------------ 1. simulation
def simulate(n_per_group=(300, 300), mixes=((0.72, 0.18, 0.10), (0.15, 0.30, 0.55)), sigma=0.22, seed=7):
    """Two groups, one true ordering; a group's diagnostic mix is set by sampling latent times
    from stage windows in the given CN/MCI/AD proportions."""
    rng = np.random.default_rng(seed)
    windows = {"CN": (0.0, 6.5), "MCI": (6.5, 13.0), "AD": (13.0, T_MAX)}
    rows = []
    for g, (n, mix) in enumerate(zip(n_per_group, mixes)):
        for i in range(n):
            dx = rng.choice(["CN", "MCI", "AD"], p=mix)
            t = rng.uniform(*windows[dx])
            values = 1 / (1 + np.exp(-(t - ONSET) / SLOPE)) + rng.normal(0, sigma, len(EVENTS))
            rows.append({"PTID": f"g{g}_{i:04d}", "Diagnosis": dx, "group": g, "t": t,
                         **{e: float(v) for e, v in zip(EVENTS, values)}})
    return pd.DataFrame(rows)


def figure_trajectories(df, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(EVENTS)))
    t = np.linspace(0, T_MAX, 300)
    for e, tau, c in zip(EVENTS, ONSET, colors):
        ax.scatter(df["t"], df[e], s=5, color=c, alpha=0.35, linewidths=0)
        ax.plot(t, 1 / (1 + np.exp(-(t - tau) / SLOPE)), color=c, lw=2, label=f"{e} (onset {tau:.1f})")
    for x, lab in ((6.5, "CN | MCI"), (13.0, "MCI | AD")):
        ax.axvline(x, color="grey", ls="--", lw=0.8); ax.text(x, 1.32, lab, ha="center", fontsize=8, color="grey")
    ax.set_xlabel("latent disease time"); ax.set_ylabel("biomarker value (sigmoid + noise)")
    ax.set_title("True ordering A < B < C < D < E < F < G < H, identical in both groups")
    ax.legend(ncol=4, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ 2./3. bootstrap heatmaps
def stratified_resample(frame, rng):
    idx = []
    for _, block in frame.groupby(["group", "Diagnosis"], sort=False):
        idx.extend(rng.choice(block.index.to_numpy(), size=len(block), replace=True))
    out = frame.loc[idx].reset_index(drop=True)
    out["PTID"] = [f"b{i:05d}" for i in range(len(out))]
    return out


def positional_counts(frame, spec, n_boot, seed):
    """counts[group][biomarker, position] over bootstrap resamples; also the point estimate."""
    rng = np.random.default_rng(seed)
    k = len(EVENTS)
    counts = np.zeros((spec.n_groups, k, k), dtype=int)
    point = fit_once(frame, spec)
    assert point.ok, point.diagnostics.get("error")
    b = 0
    while b < n_boot:
        res = fit_once(stratified_resample(frame, rng), spec)
        if not res.ok:
            continue
        for g, order in enumerate(res.orderings):
            for pos, event in enumerate(order):
                counts[g, event, pos] += 1
        b += 1
    return counts, [list(o) for o in point.orderings]


def figure_heatmaps(results, group_names, comp, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
    for col, (title, counts, point) in enumerate(results):
        for row in range(2):
            ax = axes[row, col]
            im = ax.imshow(counts[row], cmap="Blues", vmin=0, vmax=counts[row].sum(axis=1).max())
            ax.set_xticks(range(len(EVENTS))); ax.set_xticklabels(range(1, len(EVENTS) + 1))
            ax.set_yticks(range(len(EVENTS))); ax.set_yticklabels(EVENTS)
            ax.set_xlabel("event position"); ax.set_ylabel("biomarker (true order top to bottom)")
            est = "".join(EVENTS[i] for i in point[row])
            d = kendall_distance(point[row], list(range(len(EVENTS))))
            ax.set_title(f"{title}\n{group_names[row]}  ({comp[row]})\nestimate {est}   distance to truth {d:.2f}", fontsize=9)
            for (i, j), v in np.ndenumerate(counts[row]):
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if v > counts[row].max() / 2 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, label="bootstrap resamples")
    fig.suptitle("DEBM positional-variance heatmaps: same true ordering, two diagnostic mixes")
    fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ main
def main(n_boot=30, B=99, workers=4):
    OUT.mkdir(exist_ok=True)
    df = simulate()
    group_names = {0: "healthy-heavy", 1: "impaired-heavy"}
    table = pd.crosstab(df["group"].map(group_names), df["Diagnosis"])[["CN", "MCI", "AD"]]
    comp = [f"CN/MCI/AD = {'/'.join(map(str, table.loc[group_names[g]]))}" for g in (0, 1)]
    print("composition\n", table, "\n")
    figure_trajectories(df, OUT / "fig1_trajectories.png")

    frame, group_values, names = prepare_data(df.drop(columns="t"), group_column="group")
    results = []
    for estimator, title in (("standard", "standard per-group DEBM"), ("invariant_min", "CONCORD")):
        spec = CompareSpec(group_column="group", group_values=tuple(group_values), labels=("CN", "MCI", "AD"),
                           biomarkers=tuple(names), estimator=estimator, seed=1)
        counts, point = positional_counts(frame, spec, n_boot, seed=11)
        results.append((title, counts, point))
        for g in (0, 1):
            est = "".join(EVENTS[i] for i in point[g])
            print(f"{title:26s} {group_names[g]:15s} estimate {est}  distance to truth "
                  f"{kendall_distance(point[g], range(len(EVENTS))):.3f}")
        print(f"{title:26s} between-group distance {kendall_distance(point[0], point[1]):.3f}\n")
    figure_heatmaps(results, [group_names[0], group_names[1]], comp, OUT / "fig2_heatmaps.png")

    print(f"permutation tests, B={B}")
    for estimator in ("standard", "invariant_min"):
        r = concord.compare(df.drop(columns="t"), group_column="group", estimator=estimator, B=B,
                            schemes=("unrestricted", "diagnosis"), stability_resamples=0, seed=3,
                            workers=workers, verbose=False)
        pair = "0-1"
        print(f"  {estimator:14s} distance {r.distances['pairs'][pair]:.3f}   "
              f"p(shuffle all) = {r.tests['unrestricted']['pairs'][pair]['p']:.3f}   "
              f"p(within diagnosis) = {r.tests['diagnosis']['pairs'][pair]['p']:.3f}   "
              f"ESS {r.effective_sample_size}")
        r.to_json(OUT / f"compare_{estimator}.json")
    print(f"\nfigures and JSON in {OUT}")


if __name__ == "__main__":
    main()
