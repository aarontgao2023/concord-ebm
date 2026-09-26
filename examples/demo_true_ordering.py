"""Visual demonstration of the effect of diagnostic composition and of CONCORD.

    python examples/demo_true_ordering.py            # ~10 min with its 10 worker processes (main(workers=10))

Toy example: eight biomarkers A..H become abnormal in that order along a latent disease time
(sigmoid trajectories); diagnosis follows the time. Two groups share the SAME true ordering but
differ in their CN/MCI/AD proportions (a CN-heavy and an AD-heavy group). Orderings are shown as
DEBM positional-variance heatmaps (biomarker x estimated position over bootstrap resamples; a
correct estimate that agrees between the groups is a diagonal in both).

  fig1  the true ordering (trajectories)
  fig2  separately fitted DEBM and CONCORD in both groups
  fig3  separately fitted DEBM at n and 4n participants per group, and with equal proportions
  fig4  P values over datasets without a true difference: separately fitted DEBM with unrestricted
        permutation and CONCORD with within-diagnosis permutation. These P values are computed here
        over the relabelings whose fits succeeded, (1 + E) / (successful relabelings + 1);
        concord.compare() reports the same P value and, in addition, the decision and the bounds of
        P over all planned relabelings, with failed fits counted as possible exceedances.

Outputs: examples/output_demo/fig*.png and README.md (not tracked by git).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from concord.core import CompareSpec, fit_once, kendall_distance, permute_labels, prepare_data, scheme_definitions

OUT = Path(__file__).resolve().parent / "output_demo"
EVENTS = list("ABCDEFGH")                       # true ordering: A first, H last
ONSET = np.linspace(3.0, 15.0, len(EVENTS))     # latent time at which each biomarker crosses 0.5
SLOPE = 0.9
T_MAX = 20.0
LABELS = ("CN", "MCI", "AD")
WINDOWS = {"CN": (0.0, 6.5), "MCI": (6.5, 13.0), "AD": (13.0, T_MAX)}
# noise sd per biomarker: moderate separations (with very little noise every model is exact in every group)
SIGMA = 0.6 * np.array([1.2, 0.9, 1.1, 1.3, 0.9, 1.2, 1.0, 1.1])
MIX_DIFF = ((0.72, 0.18, 0.10), (0.15, 0.30, 0.55))
MIX_SAME = ((0.45, 0.25, 0.30), (0.45, 0.25, 0.30))
GROUP = {0: "CN-heavy", 1: "AD-heavy"}
MODEL = {"separate": "separately fitted DEBM", "concord": "CONCORD"}
BLUE, RED, TEAL = "#2E6F95", "#C2542D", "#1F6F8B"


# ------------------------------------------------------------------ simulation
def simulate(n_per_group=(300, 300), mixes=MIX_DIFF, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for g, (n, mix) in enumerate(zip(n_per_group, mixes)):
        for i in range(n):
            dx = rng.choice(LABELS, p=mix)
            t = rng.uniform(*WINDOWS[dx])
            values = 1 / (1 + np.exp(-(t - ONSET) / SLOPE)) + rng.normal(0, 1, len(EVENTS)) * SIGMA
            rows.append({"PTID": f"g{g}_{i:04d}", "Diagnosis": dx, "group": g, "t": t,
                         **{e: float(v) for e, v in zip(EVENTS, values)}})
    return pd.DataFrame(rows)


def prepared(df):
    frame, gv, names = prepare_data(df.drop(columns="t"), group_column="group")
    return frame, tuple(gv), tuple(names)


def spec_for(gv, names, estimator):
    return CompareSpec(group_column="group", group_values=gv, labels=LABELS, biomarkers=names, estimator=estimator, seed=1)


def comp_label(df, g):
    c = pd.crosstab(df["group"], df["Diagnosis"])[list(LABELS)].loc[g]
    return "CN/MCI/AD = " + "/".join(map(str, c))


# ------------------------------------------------------------------ figure 1
def figure_trajectories(df, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(EVENTS)))
    t = np.linspace(0, T_MAX, 300)
    for e, tau, c in zip(EVENTS, ONSET, colors):
        ax.scatter(df["t"], df[e], s=4, color=c, alpha=0.25, linewidths=0)
        ax.plot(t, 1 / (1 + np.exp(-(t - tau) / SLOPE)), color=c, lw=2.2, label=f"{e} (onset {tau:.0f})")
    for x, lab in ((6.5, "CN | MCI"), (13.0, "MCI | AD")):
        ax.axvline(x, color="grey", ls="--", lw=0.8); ax.text(x, 2.05, lab, ha="center", fontsize=8, color="grey")
    ax.set_ylim(-1.6, 2.3); ax.set_xlabel("latent disease time"); ax.set_ylabel("biomarker value")
    ax.set_title("One true ordering, A < B < C < D < E < F < G < H, the same in both groups")
    ax.legend(ncol=4, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ bootstrap heatmaps (parallel)
def _boot_job(args):
    frame_dict, gv, names, estimator, seed = args
    frame = pd.DataFrame(frame_dict)
    rng = np.random.default_rng(seed)
    idx = []
    for _, block in frame.groupby(["group", "Diagnosis"], sort=False):
        idx.extend(rng.choice(block.index.to_numpy(), size=len(block), replace=True))
    boot = frame.loc[idx].reset_index(drop=True); boot["PTID"] = [f"b{i:05d}" for i in range(len(boot))]
    r = fit_once(boot, spec_for(gv, names, estimator))
    return [list(map(int, o)) for o in r.orderings] if r.ok else None


def heatmap_counts(frame, gv, names, estimator, n_boot, pool, seed=11):
    point = fit_once(frame, spec_for(gv, names, estimator)); assert point.ok, point.diagnostics.get("error")
    jobs = [(frame.to_dict("list"), gv, names, estimator, seed + b) for b in range(n_boot)]
    k = len(EVENTS); counts = np.zeros((len(gv), k, k), dtype=int)
    for orders in pool.map(_boot_job, jobs):
        if orders is None:
            continue
        for g, order in enumerate(orders):
            for pos, event in enumerate(order):
                counts[g, event, pos] += 1
    return counts, [list(map(int, o)) for o in point.orderings]


def draw_heatmap(ax, counts, point, title, vmax):
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(EVENTS))); ax.set_xticklabels(range(1, len(EVENTS) + 1), fontsize=8)
    ax.set_yticks(range(len(EVENTS))); ax.set_yticklabels(EVENTS, fontsize=8)
    est = "".join(EVENTS[i] for i in point); d = kendall_distance(point, list(range(len(EVENTS))))
    ax.set_title(f"{title}\nestimate {est}   error vs truth {d:.2f}", fontsize=9)
    for (i, j), v in np.ndenumerate(counts):
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color="white" if v > vmax / 2 else "black")


def figure_core(df, frame, gv, names, n_boot, pool, path):
    """fig2: rows = groups, columns = separately fitted DEBM vs CONCORD."""
    res = {est: heatmap_counts(frame, gv, names, est, n_boot, pool) for est in ("separate", "concord")}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), constrained_layout=True)
    distance = {}
    for col, (est, color) in enumerate((("separate", RED), ("concord", TEAL))):
        counts, point = res[est]
        for row in range(2):
            draw_heatmap(axes[row, col], counts[row], point[row], f"{MODEL[est]}\n{GROUP[row]} group  ({comp_label(df, row)})", n_boot)
            axes[row, col].title.set_color(color)
        distance[est] = kendall_distance(point[0], point[1])
        axes[1, col].set_xlabel(f"estimated position\n\ndistance between the two groups' orderings: {distance[est]:.3f}", fontsize=10, color=color, weight="bold")
    axes[0, 0].set_ylabel("biomarker (true order)", fontsize=8); axes[1, 0].set_ylabel("biomarker (true order)", fontsize=8)
    fig.suptitle("Same true ordering in both groups; the groups differ only in their CN/MCI/AD proportions", fontsize=12)
    fig.savefig(path, dpi=160); plt.close(fig)
    return distance


def figure_more_data(n_boot, pool, path):
    """fig3: separately fitted DEBM; columns = different proportions at n, at 4n, equal proportions at 4n."""
    cells = ((MIX_DIFF, 300, "different proportions,  n = 300 per group"),
             (MIX_DIFF, 1200, "different proportions,  4n = 1200 per group"),
             (MIX_SAME, 1200, "equal proportions,  4n = 1200 per group"))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), constrained_layout=True)
    distance = []
    for col, (mixes, n, title) in enumerate(cells):
        df = simulate((n, n), mixes, seed=21 + col); frame, gv, names = prepared(df)
        counts, point = heatmap_counts(frame, gv, names, "separate", n_boot, pool)
        for row in range(2):
            draw_heatmap(axes[row, col], counts[row], point[row], f"{title}\n{GROUP[row] + ' group' if mixes is MIX_DIFF else 'group ' + str(row + 1)}  ({comp_label(df, row)})", n_boot)
        d = kendall_distance(point[0], point[1]); distance.append((title, d))
        axes[1, col].set_xlabel(f"estimated position\n\nbetween-group distance {d:.3f}", fontsize=10, weight="bold",
                                color=RED if mixes is MIX_DIFF else "grey")
    axes[0, 0].set_ylabel("biomarker (true order)", fontsize=8); axes[1, 0].set_ylabel("biomarker (true order)", fontsize=8)
    fig.suptitle("Separately fitted DEBM with four times as many participants, and with equal CN/MCI/AD proportions", fontsize=12)
    fig.savefig(path, dpi=160); plt.close(fig)
    return distance


# ------------------------------------------------------------------ figure 4: P values over datasets without a true difference
def _pvalue_job(args):
    """One dataset without a true difference: P for separately fitted DEBM with unrestricted
    permutation and for CONCORD with within-diagnosis permutation. P divides by the successful
    relabelings, (1 + E) / (successful + 1); failed fits are left out."""
    seed, B = args
    df = simulate((300, 300), MIX_DIFF, seed); frame, gv, names = prepared(df)
    out = {}
    for est, scheme in (("separate", "unrestricted"), ("concord", "within_diagnosis")):
        spec = CompareSpec(group_column="group", group_values=gv, labels=LABELS, biomarkers=names, estimator=est, seed=seed)
        defs = scheme_definitions(spec, (scheme,))
        obs_fit = fit_once(frame, spec); obs = kendall_distance(obs_fit.orderings[0], obs_fit.orderings[1])
        ref = []
        for b in range(B):
            r = fit_once(permute_labels(frame, spec, scheme, defs[scheme], b), spec)
            if r.ok:
                ref.append(kendall_distance(r.orderings[0], r.orderings[1]))
        out[est] = (1 + sum(d >= obs - 1e-9 for d in ref)) / (len(ref) + 1)
    return out


def figure_pvalues(n_datasets, B, pool, path):
    res = list(pool.map(_pvalue_job, [(100 + s, B) for s in range(n_datasets)]))
    panels = (("separate", "separately fitted DEBM, unrestricted permutation", RED),
              ("concord", "CONCORD, within-diagnosis permutation", TEAL))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True, sharey=True)
    for ax, (est, title, color) in zip(axes, panels):
        p = np.array([r[est] for r in res]); fp = np.mean(p <= 0.05)
        ax.hist(p, bins=np.linspace(0, 1, 11), color=color, alpha=0.85, edgecolor="white")
        ax.axhline(n_datasets / 10, color="grey", ls="--", lw=0.8)
        ax.text(0.98, 0.95, f"P ≤ 0.05 in {100*fp:.0f}% of datasets\n(nominal level 5%)", ha="right", va="top", transform=ax.transAxes, fontsize=9, color=color, weight="bold")
        ax.set_xlabel("P value"); ax.set_title(title, fontsize=9, color=color)
    axes[0].set_ylabel(f"datasets (of {n_datasets})")
    fig.suptitle(f"{n_datasets} datasets without a true difference: P values of each procedure (dashed line = uniform)", fontsize=10)
    fig.savefig(path, dpi=160); plt.close(fig)
    return res


# ------------------------------------------------------------------ main
def main(n_boot=30, n_datasets=40, B=39, workers=10):
    OUT.mkdir(exist_ok=True); t0 = time.monotonic()
    df = simulate(); frame, gv, names = prepared(df)
    print("composition\n", pd.crosstab(df["group"].map(GROUP), df["Diagnosis"])[list(LABELS)], "\n", flush=True)
    figure_trajectories(df, OUT / "fig1_true_ordering.png")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print("fig2 separately fitted DEBM and CONCORD ...", flush=True); core = figure_core(df, frame, gv, names, n_boot, pool, OUT / "fig2_core_heatmaps.png")
        print("fig3 more participants ...", flush=True); more = figure_more_data(n_boot, pool, OUT / "fig3_more_data.png")
        print("fig4 P values ...", flush=True); pv = figure_pvalues(n_datasets, B, pool, OUT / "fig4_pvalues.png")
    more_text = "; ".join(f"{' '.join(title.split())}: {d:.3f}" for title, d in more)
    (OUT / "README.md").write_text(f"""# CONCORD in four figures

Toy example: eight biomarkers A–H become abnormal in that order along a latent disease time; diagnosis follows the time.
Two groups of 300 participants **share the true ordering** and differ only in their CN/MCI/AD proportions
({comp_label(df, 0)} in the CN-heavy group, {comp_label(df, 1)} in the AD-heavy group). Heatmaps: biomarker × estimated
position over {n_boot} bootstrap resamples within group-by-diagnosis cells; a correct estimate is a diagonal.

## 1. The true ordering
![](fig1_true_ordering.png)

## 2. Separately fitted DEBM and CONCORD
Separately fitted DEBM fits the abnormality model and the ordering in each group. CONCORD fits one abnormality model to
the pooled sample and weights participants so that CN, MCI and AD have the same proportions in both groups.
Normalized Kendall distance between the two groups' orderings: {core['separate']:.3f} (separately fitted DEBM) and
{core['concord']:.3f} (CONCORD).
![](fig2_core_heatmaps.png)

## 3. More participants, and equal proportions (separately fitted DEBM)
Left to right: the two groups at n and at 4n participants per group, and two groups with equal proportions at 4n.
Distance between the two groups' orderings: {more_text}.
![](fig3_more_data.png)

## 4. P values over {n_datasets} datasets without a true difference
Unrestricted permutation exchanges group labels among all participants; within-diagnosis permutation exchanges them
only among participants with the same diagnosis, so every relabeled group keeps its CN/MCI/AD counts.
P ≤ 0.05 in {100*np.mean([r["separate"] <= 0.05 for r in pv]):.0f}% of datasets for separately fitted DEBM with
unrestricted permutation and in {100*np.mean([r["concord"] <= 0.05 for r in pv]):.0f}% for CONCORD with within-diagnosis
permutation ({B} relabelings per dataset).
![](fig4_pvalues.png)

_{(time.monotonic() - t0) / 60:.1f} min on {workers} workers._
""")
    print(f"done in {(time.monotonic()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
