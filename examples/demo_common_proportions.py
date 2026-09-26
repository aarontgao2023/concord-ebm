"""How much does the choice of common proportions matter?  A small demo.

    python examples/demo_common_proportions.py            # ~3 min on 8 cores

Same toy example as demo_true_ordering.py (eight biomarkers A..H, one true ordering, a CN-heavy and
an AD-heavy group). CONCORD weights each group's participants to common CN/MCI/AD proportions.
Here the common proportions are varied:

  minimum rule   the smallest proportion of each diagnosis across the groups, rescaled to sum to
                 one (CONCORD default)
  pooled         the proportions of the pooled sample
  equal thirds   1/3, 1/3, 1/3
  CN-heavy       0.8, 0.1, 0.1     (an extreme choice, close to the CN-heavy group's own proportions)
  AD-heavy       0.1, 0.1, 0.8     (an extreme choice, close to the AD-heavy group's own proportions)

and, for contrast, separately fitted DEBM (no weighting). For each choice, over R datasets without a
true difference: the normalized Kendall distance between the two groups' orderings, each group's
distance to the true ordering, and each group's effective sample size after weighting.

Outputs: examples/output_demo/fig5_common_proportions.png and COMMON_PROPORTIONS.md (not tracked by git).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concord.core import kendall_distance
from concord.engine import EngineConfig, fit_orderings
from concord.invariant import fit_invariant_orderings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_true_ordering import EVENTS, GROUP, LABELS, MIX_DIFF, OUT, prepared, simulate  # noqa: E402

# (label, common proportions): None = separately fitted DEBM; "min"/"pooled" = rules; tuple = fixed (CN, MCI, AD)
CHOICES = [("separately fitted DEBM", None), ("minimum rule (default)", "min"), ("pooled sample", "pooled"),
           ("equal thirds", (1 / 3, 1 / 3, 1 / 3)), ("CN-heavy 0.8/0.1/0.1", (0.8, 0.1, 0.1)),
           ("AD-heavy 0.1/0.1/0.8", (0.1, 0.1, 0.8))]
RED, TEAL, GREY = "#C2542D", "#1F6F8B", "#8A949C"


def _job(args):
    seed, n = args
    df = simulate((n, n), MIX_DIFF, seed); frame, gv, names = prepared(df)
    # engine identifiers: mode "repaired" is the continued search; variant "invariant_min" is CONCORD
    cfg = EngineConfig(mode="repaired", expected_events=len(names), group_column="group", group_values=tuple(range(len(gv))),
                       labels=LABELS, biomarker_names=tuple(names))
    out = {}
    for label, proportions in CHOICES:
        if proportions is None:
            r = fit_orderings(frame, cfg)
        elif isinstance(proportions, str):
            r = fit_invariant_orderings(frame, cfg, variant=f"invariant_{proportions}")
        else:
            r = fit_invariant_orderings(frame, cfg, variant="invariant_min", reference=proportions)
        if not r.ok:
            out[label] = None; continue
        o = [list(map(int, x)) for x in r.orderings]
        ess = r.diagnostics.get("effective_sample_size") or {"0": float(n), "1": float(n)}
        out[label] = {"between": kendall_distance(o[0], o[1]),
                      "error": [kendall_distance(o[g], list(range(len(EVENTS)))) for g in (0, 1)],
                      "ess": [float(ess["0"]), float(ess["1"])]}
    return out


def main(R=30, n=300, workers=8):
    OUT.mkdir(exist_ok=True); t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_job, [(500 + s, n) for s in range(R)]))
    labels = [l for l, _ in CHOICES]
    between = np.array([[r[l]["between"] for r in res if r[l]] for l in labels])           # choice x R
    err = np.array([[r[l]["error"] for r in res if r[l]] for l in labels])                 # choice x R x 2
    ess = np.array([[r[l]["ess"] for r in res if r[l]] for l in labels])                   # choice x R x 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    x = np.arange(len(labels)); colors = [RED] + [TEAL] * (len(labels) - 1)
    m, se = between.mean(1), between.std(1, ddof=1) / np.sqrt(between.shape[1])
    axes[0].bar(x, m, yerr=1.96 * se, color=colors, capsize=3)
    axes[0].set_ylabel("Kendall distance between the two groups' orderings"); axes[0].set_title("Agreement between the groups without a true difference\n(lower is better; 0 = identical orderings)", fontsize=10)
    w = 0.38
    for g, (hatch, name) in enumerate(((None, GROUP[0]), ("//", GROUP[1]))):
        axes[1].bar(x + (g - 0.5) * w, err[:, :, g].mean(1), w, yerr=1.96 * err[:, :, g].std(1, ddof=1) / np.sqrt(err.shape[1]),
                    color=[c if g == 0 else "white" for c in colors], edgecolor=colors, hatch=hatch, capsize=2, label=name)
        axes[2].bar(x + (g - 0.5) * w, ess[:, :, g].mean(1), w, color=[c if g == 0 else "white" for c in colors], edgecolor=colors, hatch=hatch, label=name)
    axes[1].set_ylabel("Kendall distance to the true ordering"); axes[1].set_title("Error of each group's ordering\n(lower is better)", fontsize=10); axes[1].legend(fontsize=8, frameon=False)
    axes[2].axhline(n, color="grey", ls="--", lw=0.8); axes[2].set_ylabel(f"effective sample size (n = {n} per group)"); axes[2].set_title("Effective sample size after weighting\n(proportions far from a group's own lower it)", fontsize=10); axes[2].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.suptitle(f"Choice of common proportions: {R} datasets without a true difference, {GROUP[0]} group {MIX_DIFF[0]} vs {GROUP[1]} group {MIX_DIFF[1]}", fontsize=11)
    fig.savefig(OUT / "fig5_common_proportions.png", dpi=160); plt.close(fig)

    lines = ["# How much does the choice of common proportions matter?\n",
             f"{R} datasets, n = {n} per group, same true ordering, CN/MCI/AD proportions {MIX_DIFF[0]} (CN-heavy group) vs {MIX_DIFF[1]} (AD-heavy group). Mean ± 1.96 SE over datasets.\n",
             "| common proportions | between-group distance | error vs truth, CN-heavy | error vs truth, AD-heavy | ESS CN-heavy | ESS AD-heavy |",
             "|---|---|---|---|---|---|"]
    for i, l in enumerate(labels):
        lines.append(f"| {l} | {between[i].mean():.3f} ± {1.96*between[i].std(ddof=1)/np.sqrt(R):.3f} | {err[i,:,0].mean():.3f} | {err[i,:,1].mean():.3f} | {ess[i,:,0].mean():.0f} | {ess[i,:,1].mean():.0f} |")
    lines += ["", "![](fig5_common_proportions.png)", "",
              "**How to read it.** Every choice of common proportions gives both groups the same CN/MCI/AD proportions after weighting; "
              "the choice determines at which proportions both orderings are estimated and how strongly each group is reweighted. "
              "A participant's weight is the common proportion of their diagnosis divided by its proportion in their own group, so "
              "proportions far from a group's own raise the weights of its rarest diagnosis (equal thirds give the CN-heavy group's "
              "AD participants a weight of about 0.33/0.10 = 3.3) and lower its effective sample size. "
              "The minimum rule gives the smallest possible largest weight. "
              "Every diagnosis with a positive common proportion must be present in every group.",
              f"\n_{(time.monotonic()-t0)/60:.1f} min on {workers} workers._"]
    (OUT / "COMMON_PROPORTIONS.md").write_text("\n".join(lines))
    print("\n".join(lines[:12])); print(f"done in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
