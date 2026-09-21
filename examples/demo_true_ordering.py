"""Minimal, fully visual demonstration of the composition problem and of CONCORD.

    python examples/demo_true_ordering.py            # ~10 min on 8 cores

Toy world: eight biomarkers A..H turn abnormal in that order along a latent disease time (sigmoid
trajectories); diagnosis follows the time.  Two groups share the SAME true ordering but differ in
their CN/MCI/AD mix ("healthy-heavy" vs "impaired-heavy").  Everything is shown as DEBM
positional-variance heatmaps (biomarker x estimated position over bootstrap resamples; a correct,
comparable estimate is a diagonal).

  fig1  the true ordering (trajectories)
  fig2  THE CORE: per-group DEBM vs CONCORD, both groups            -> same truth, two different heatmaps; one after the fix
  fig3  more data does not help: per-group DEBM at n and 4n         -> the scramble stays; with an identical mix it goes away
  fig4  the test: p-values over many no-difference cohorts, published procedure vs CONCORD

Outputs: examples/output_demo/fig*.png and README.md.
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
# noise sd per marker: ADNI-like separations (too little noise and every estimator is perfect in every group)
SIGMA = 0.6 * np.array([1.2, 0.9, 1.1, 1.3, 0.9, 1.2, 1.0, 1.1])
MIX_DIFF = ((0.72, 0.18, 0.10), (0.15, 0.30, 0.55))
MIX_SAME = ((0.45, 0.25, 0.30), (0.45, 0.25, 0.30))
GROUP = {0: "healthy-heavy", 1: "impaired-heavy"}
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
    ax.set_title("One true ordering, A < B < C < D < E < F < G < H, identical in both groups")
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
    """fig2: rows = groups, columns = standard vs CONCORD."""
    res = {est: heatmap_counts(frame, gv, names, est, n_boot, pool) for est in ("standard", "invariant_min")}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), constrained_layout=True)
    for col, (est, name, color) in enumerate((("standard", "per-group DEBM (current practice)", RED), ("invariant_min", "CONCORD", TEAL))):
        counts, point = res[est]
        for row in range(2):
            draw_heatmap(axes[row, col], counts[row], point[row], f"{name}\n{GROUP[row]}  ({comp_label(df, row)})", n_boot)
            axes[row, col].title.set_color(color)
        d = kendall_distance(point[0], point[1])
        axes[1, col].set_xlabel(f"estimated position\n\ndistance between the two groups' estimates: {d:.3f}", fontsize=10, color=color, weight="bold")
    axes[0, 0].set_ylabel("biomarker (true order)", fontsize=8); axes[1, 0].set_ylabel("biomarker (true order)", fontsize=8)
    fig.suptitle("Same true ordering in both groups — the only difference is who is in each group", fontsize=12)
    fig.savefig(path, dpi=160); plt.close(fig)
    return res


def figure_more_data(n_boot, pool, path):
    """fig3: per-group DEBM; columns = different mix at n, different mix at 4n, identical mix at 4n."""
    cells = ((MIX_DIFF, 300, "different mix,  n = 300 per group"),
             (MIX_DIFF, 1200, "different mix,  4n = 1200 per group"),
             (MIX_SAME, 1200, "identical mix,  4n = 1200 per group"))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), constrained_layout=True)
    for col, (mixes, n, title) in enumerate(cells):
        df = simulate((n, n), mixes, seed=21 + col); frame, gv, names = prepared(df)
        counts, point = heatmap_counts(frame, gv, names, "standard", n_boot, pool)
        for row in range(2):
            draw_heatmap(axes[row, col], counts[row], point[row], f"{title}\n{GROUP[row] if mixes is MIX_DIFF else 'group ' + str(row + 1)}  ({comp_label(df, row)})", n_boot)
        d = kendall_distance(point[0], point[1])
        axes[1, col].set_xlabel(f"estimated position\n\nbetween-group distance {d:.3f}", fontsize=10, weight="bold",
                                color=RED if mixes is MIX_DIFF else "grey")
    axes[0, 0].set_ylabel("biomarker (true order)", fontsize=8); axes[1, 0].set_ylabel("biomarker (true order)", fontsize=8)
    fig.suptitle("Per-group DEBM: four times the data barely reduces the disagreement — an identical mix removes it", fontsize=12)
    fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ figure 4: the test over many no-difference cohorts
def _pvalue_job(args):
    """One no-difference cohort: p under (per-group DEBM, shuffle all) and (CONCORD, shuffle within diagnosis)."""
    seed, B = args
    df = simulate((300, 300), MIX_DIFF, seed); frame, gv, names = prepared(df)
    out = {}
    for est, scheme in (("standard", "unrestricted"), ("invariant_min", "diagnosis")):
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


def figure_pvalues(n_cohorts, B, pool, path):
    res = list(pool.map(_pvalue_job, [(100 + s, B) for s in range(n_cohorts)]))
    panels = (("standard", "per-group DEBM + shuffle labels among everyone\n(the published test)", RED),
              ("invariant_min", "CONCORD + shuffle labels within diagnosis", TEAL))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True, sharey=True)
    for ax, (est, title, color) in zip(axes, panels):
        p = np.array([r[est] for r in res]); fp = np.mean(p <= 0.05)
        ax.hist(p, bins=np.linspace(0, 1, 11), color=color, alpha=0.85, edgecolor="white")
        ax.axhline(n_cohorts / 10, color="grey", ls="--", lw=0.8)
        ax.text(0.98, 0.95, f"p ≤ 0.05 in {100*fp:.0f} % of cohorts\n(should be ≤ 5 %)", ha="right", va="top", transform=ax.transAxes, fontsize=9, color=color, weight="bold")
        ax.set_xlabel("p-value"); ax.set_title(title, fontsize=9, color=color)
    axes[0].set_ylabel(f"cohorts (of {n_cohorts})")
    fig.suptitle(f"{n_cohorts} cohorts with NO true difference: the p-values each procedure produces (dashed = uniform)", fontsize=10)
    fig.savefig(path, dpi=160); plt.close(fig)
    return res


# ------------------------------------------------------------------ main
def main(n_boot=30, n_cohorts=40, B=39, workers=10):
    OUT.mkdir(exist_ok=True); t0 = time.monotonic()
    df = simulate(); frame, gv, names = prepared(df)
    print("composition\n", pd.crosstab(df["group"].map(GROUP), df["Diagnosis"])[list(LABELS)], "\n", flush=True)
    figure_trajectories(df, OUT / "fig1_true_ordering.png")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print("fig2 core ...", flush=True); res = figure_core(df, frame, gv, names, n_boot, pool, OUT / "fig2_core_heatmaps.png")
        print("fig3 more data ...", flush=True); figure_more_data(n_boot, pool, OUT / "fig3_more_data.png")
        print("fig4 p-values ...", flush=True); pv = figure_pvalues(n_cohorts, B, pool, OUT / "fig4_pvalues.png")
    (OUT / "README.md").write_text(f"""# CONCORD in four pictures

Toy world: eight biomarkers A–H turn abnormal in that order along a latent disease time; diagnosis follows the time.
Two groups, n = 300 each, **share the true ordering**; they differ only in their CN/MCI/AD mix
({comp_label(df, 0)} vs {comp_label(df, 1)}). Heatmaps: biomarker × estimated position over {n_boot} bootstrap
resamples; a correct, comparable estimate is a diagonal.

## 1. The truth
![](fig1_true_ordering.png)

## 2. The core: same truth, two different answers — one answer after the fix
Per-group DEBM: the healthy-heavy group gets the early events right and scrambles the late ones; the impaired-heavy
group does the opposite. Each is "right about the half of the disease its subjects can see." CONCORD (one measurement
model, one reference composition) gives the two groups the same heatmap.
![](fig2_core_heatmaps.png)

## 3. More data barely helps; an identical mix does
Left to right: the two groups at n, at 4n, and two groups with an identical mix at 4n. Sampling noise shrinks with n; the disagreement produced by the mix does not go away — it is a property of what each group's estimator is aiming at.
![](fig3_more_data.png)

## 4. The test, over {n_cohorts} cohorts with no true difference
Shuffling labels among everyone compares against a world where the groups have the same mix: the p-values pile up near
zero and the published test calls a difference in {100*np.mean([r["standard"] <= 0.05 for r in pv]):.0f} % of cohorts.
CONCORD with a within-diagnosis shuffle: {100*np.mean([r["invariant_min"] <= 0.05 for r in pv]):.0f} %, and the p-values are flat, as they must be when nothing differs.
![](fig4_pvalues.png)

_{(time.monotonic() - t0) / 60:.1f} min on {workers} workers._
""")
    print(f"done in {(time.monotonic()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
