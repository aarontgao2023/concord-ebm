"""Does the choice of reference composition matter?  A small demo.

    python examples/demo_reference_choice.py            # ~3 min on 8 cores

Same toy world as demo_true_ordering.py (eight biomarkers A..H, one true ordering, two groups with a
healthy-heavy vs impaired-heavy mix).  CONCORD standardises each group's consensus to a common reference
composition pi_ref.  Here the reference is varied:

  min        the sparsest mix every group can represent (CONCORD default)
  pooled     the mix of the combined sample
  equal      1/3, 1/3, 1/3
  CN-heavy   0.8, 0.1, 0.1     (an extreme choice: close to group 1's own mix)
  AD-heavy   0.1, 0.1, 0.8     (an extreme choice: close to group 2's own mix)

and, for contrast, the standard per-group DEBM (no common reference at all).  For each choice, over R
no-difference cohorts: the distance between the two groups' orderings (comparability), each group's
error against the truth (accuracy), and the effective sample size after weighting (the price).

Outputs: examples/output_demo/fig5_reference_choice.png and REFERENCE.md.
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

REFERENCES = [("standard per-group DEBM", None), ("min (default)", "min"), ("pooled", "pooled"),
              ("equal thirds", (1 / 3, 1 / 3, 1 / 3)), ("CN-heavy 0.8/0.1/0.1", (0.8, 0.1, 0.1)),
              ("AD-heavy 0.1/0.1/0.8", (0.1, 0.1, 0.8))]
RED, TEAL, GREY = "#C2542D", "#1F6F8B", "#8A949C"


def _job(args):
    seed, n = args
    df = simulate((n, n), MIX_DIFF, seed); frame, gv, names = prepared(df)
    cfg = EngineConfig(mode="repaired", expected_events=len(names), group_column="group", group_values=tuple(range(len(gv))),
                       labels=LABELS, biomarker_names=tuple(names))
    out = {}
    for label, ref in REFERENCES:
        if ref is None:
            r = fit_orderings(frame, cfg)
        elif isinstance(ref, str):
            r = fit_invariant_orderings(frame, cfg, variant=f"invariant_{ref}")
        else:
            r = fit_invariant_orderings(frame, cfg, variant="invariant_min", reference=ref)
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
    labels = [l for l, _ in REFERENCES]
    between = np.array([[r[l]["between"] for r in res if r[l]] for l in labels])           # ref x R
    err = np.array([[r[l]["error"] for r in res if r[l]] for l in labels])                 # ref x R x 2
    ess = np.array([[r[l]["ess"] for r in res if r[l]] for l in labels])                   # ref x R x 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    x = np.arange(len(labels)); colors = [RED] + [TEAL] * (len(labels) - 1)
    m, se = between.mean(1), between.std(1, ddof=1) / np.sqrt(between.shape[1])
    axes[0].bar(x, m, yerr=1.96 * se, color=colors, capsize=3)
    axes[0].set_ylabel("Kendall distance between the two groups' orderings"); axes[0].set_title("Comparability under NO true difference\n(lower is better; 0 = identical orderings)", fontsize=10)
    w = 0.38
    for g, (hatch, name) in enumerate(((None, GROUP[0]), ("//", GROUP[1]))):
        axes[1].bar(x + (g - 0.5) * w, err[:, :, g].mean(1), w, yerr=1.96 * err[:, :, g].std(1, ddof=1) / np.sqrt(err.shape[1]),
                    color=[c if g == 0 else "white" for c in colors], edgecolor=colors, hatch=hatch, capsize=2, label=name)
        axes[2].bar(x + (g - 0.5) * w, ess[:, :, g].mean(1), w, color=[c if g == 0 else "white" for c in colors], edgecolor=colors, hatch=hatch, label=name)
    axes[1].set_ylabel("Kendall distance to the TRUE ordering"); axes[1].set_title("Accuracy of each group's estimate\n(lower is better)", fontsize=10); axes[1].legend(fontsize=8, frameon=False)
    axes[2].axhline(n, color="grey", ls="--", lw=0.8); axes[2].set_ylabel(f"effective sample size (n = {n} per group)"); axes[2].set_title("The price of the reference\n(a reference far from a group's own mix costs information)", fontsize=10); axes[2].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.suptitle(f"Does the reference composition matter?  {R} no-difference cohorts, groups {GROUP[0]} {MIX_DIFF[0]} vs {GROUP[1]} {MIX_DIFF[1]}", fontsize=11)
    fig.savefig(OUT / "fig5_reference_choice.png", dpi=160); plt.close(fig)

    lines = ["# Does the reference composition matter?\n",
             f"{R} cohorts, n = {n} per group, same true ordering, mixes {MIX_DIFF[0]} vs {MIX_DIFF[1]}. Mean ± 1.96 SE over cohorts.\n",
             "| reference | between-group distance | error vs truth, healthy-heavy | error vs truth, impaired-heavy | ESS healthy-heavy | ESS impaired-heavy |",
             "|---|---|---|---|---|---|"]
    for i, l in enumerate(labels):
        lines.append(f"| {l} | {between[i].mean():.3f} ± {1.96*between[i].std(ddof=1)/np.sqrt(R):.3f} | {err[i,:,0].mean():.3f} | {err[i,:,1].mean():.3f} | {ess[i,:,0].mean():.0f} | {ess[i,:,1].mean():.0f} |")
    lines += ["", "![](fig5_reference_choice.png)", "",
              "**Reading.** Any *common* reference makes the two groups estimate the same target: the between-group distance drops from the per-group value to the noise level for every choice, including equal thirds. "
              "The reference therefore does not decide *whether* the orderings agree — it decides *which composition's ordering* both groups estimate, and *how much information is spent* to get there. "
              "A reference far from a group's own mix asks that group to up-weight its rarest subjects (equal thirds gives the healthy-heavy group's few AD subjects weight 0.33/0.10 ≈ 3.3; an AD-heavy reference asks even more), which lowers its effective sample size and raises its error. "
              "The sparsest common mix (`min`) keeps every weight ≤ 1 inside the group that defines it, so it is the cheapest common reference; the pooled mix is a close second. "
              "Under a genuine group-by-stage interaction the choice of reference would also change the estimand itself — that is the setting to worry about, and the composition table plus the ESS tell you when you are in it.",
              f"\n_{(time.monotonic()-t0)/60:.1f} min on {workers} workers._"]
    (OUT / "REFERENCE.md").write_text("\n".join(lines))
    print("\n".join(lines[:12])); print(f"done in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
