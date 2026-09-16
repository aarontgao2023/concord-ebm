"""CONCORD: composition-invariant comparison of event-based model orderings between groups.

    import concord
    result = concord.compare(df, group_column="APOE")

Data frame in (PTID, Diagnosis, one group column, biomarker columns; NaN allowed in
biomarkers); out come the per-group orderings, the pairwise ordering distances, the
permutation p-values under four reference distributions, the exact decisions under
the frozen rule, and the diagnostics the method section asks for: the group-by-
diagnosis composition table, the effective sample size after standardisation, the
stability of every group ordering under stratified subject resampling, and the paired
stratified-minus-unrestricted p difference.

Estimators (``estimator=``):
  invariant_min     pooled mixture + consensus standardised to the sparsest common
                    composition (default; the paper's recommended procedure)
  invariant_pooled  pooled mixture + consensus standardised to the pooled composition
  shared            pooled mixture only (diagnostic arm)
  standard          separately fitted co-init DEBM per group (the published practice)
  saebm             stage-aware EBM per group (optional dependency ``pysaebm``;
                    complete data only)

Reference distributions (``schemes=``), all relabelling only the group column:
  unrestricted      labels exchanged among all subjects (the published U-all operator)
  diagnosis         labels exchanged within diagnosis strata (D-all; primary)
  diagnosis_pair    for each pair, only that pair's labels exchanged within diagnosis
  diagnosis_count   labels exchanged within diagnosis x number-of-observed-biomarkers strata
                    (the repair operator for group-specific missingness)
                    strata; the third group is left untouched (D-pair; localisation)
The maximum pairwise distance is evaluated against the unrestricted and diagnosis
references (U-max, D-max).

Decision rule (frozen protocol): Monte-Carlo p with the +1 convention, ties counted
as exceedances, ``le`` comparison, Bonferroni alpha/n_pairs per pair, alpha for the
maximum. Decisions are exact over every completion of the planned budget, so failed
or unfinished fits leave a decision undetermined rather than change the denominator.
Optional exact early stopping (``stop_when_decided``) stops a reference once every
required endpoint is determined; the reported p is then a bound, not a point value.

The pooled mixture depends on biomarker values only, so it is fitted once per worker
process and reused across every relabelling (``concord.invariant``). Everything
label-dependent (weights, consensus) is recomputed at every fit. Uses the audited
engines unchanged; requires the pinned pyebm 2.0.3 (``concord._pyebm``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import sys
import time
import warnings

import numpy as np
import pandas as pd

from ._pyebm import verify_pyebm
from ._version import __version__
from .engine import EngineConfig, FitResult, fit_orderings
from .invariant import fit_invariant_orderings
from .likelihood import fast_likelihood_context

PACKAGE_VERSION = f"concord-{__version__}"
ESTIMATORS = ("invariant_min", "invariant_pooled", "shared", "standard", "saebm")
SCHEMES = ("unrestricted", "diagnosis", "diagnosis_pair")          # default set (frozen protocol)
OPTIONAL_SCHEMES = ("diagnosis_count",)                             # repair operator, on request
PERMUTATION_STREAM = 61000000      # same stream constant as the simulation runner (reproducible permutations)
RESAMPLE_STREAM = 62000000
_RESERVED_TOKENS = ("PTID", "Diagnosis", "EXAMDATE")   # pyebm drops columns by substring


# ----------------------------------------------------------------------------- specification

@dataclass(frozen=True)
class CompareSpec:
    """Everything a worker needs; picklable and recorded verbatim in the result."""
    group_column: str
    group_values: tuple            # original values in code order 0..G-1
    labels: tuple                  # diagnosis labels in stage order (first=control, last=case)
    biomarkers: tuple
    estimator: str = "invariant_min"
    consensus: str = "repaired"    # pyebm consensus arm: 'repaired' or 'original'
    fast_likelihood: bool = True
    fit_timeout_s: int = 900
    seed: int = 0
    saebm_iterations: int = 2000
    saebm_burn_in: int = 500

    def __post_init__(self):
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.consensus not in ("original", "repaired"):
            raise ValueError("consensus must be 'original' or 'repaired'")
        if len(self.group_values) < 2:
            raise ValueError("at least two groups are required")
        if len(self.labels) < 2:
            raise ValueError("at least two diagnosis labels are required")

    @property
    def n_groups(self):
        return len(self.group_values)

    @property
    def pairs(self):
        return list(combinations(range(self.n_groups), 2))

    def pair_name(self, pair):
        a, b = pair
        return f"{self.group_values[a]}-{self.group_values[b]}"


# ----------------------------------------------------------------------------- data preparation

def prepare_data(df, group_column="APOE", labels=("CN", "MCI", "AD"), biomarkers=None,
                 group_order=None):
    """Validate and encode the input frame for pyebm; returns (frame, group_values, biomarkers).

    Diagnosis values must all be in ``labels``: pyebm silently codes any other value as the
    middle stage, so this is checked here. Group values are encoded 0..G-1 in ``group_order``
    (default: sorted unique values) because pyebm sorts groups numerically.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for column in ("Diagnosis", group_column):
        if column not in df.columns:
            raise ValueError(f"column {column!r} not found")
    labels = tuple(labels)
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be distinct")
    unknown = sorted(set(map(str, df["Diagnosis"].unique())) - set(map(str, labels)))
    if unknown:
        raise ValueError(f"Diagnosis values outside labels {labels}: {unknown}")
    if df[group_column].isna().any():
        raise ValueError(f"missing values in group column {group_column!r}")
    observed = sorted((v.item() if hasattr(v, "item") else v for v in df[group_column].unique()),
                      key=lambda v: (str(type(v)), v))
    if group_order is None:
        group_values = tuple(observed)
    else:
        group_values = tuple(group_order)
        if set(group_values) != set(observed):
            raise ValueError(f"group_order {group_values} does not match observed groups {observed}")
    if len(group_values) < 2:
        raise ValueError("at least two groups are required")
    metadata = {"PTID", "Diagnosis", "EXAMDATE", group_column}
    if biomarkers is None:
        biomarkers = [c for c in df.columns if c not in metadata]
    biomarkers = list(biomarkers)
    if len(biomarkers) < 2:
        raise ValueError("at least two biomarker columns are required")
    if len(set(biomarkers)) != len(biomarkers):
        raise ValueError("duplicate biomarker names")
    for name in biomarkers:
        if name not in df.columns:
            raise ValueError(f"biomarker column {name!r} not found")
        if any(token in name for token in _RESERVED_TOKENS) or group_column in name:
            raise ValueError(f"biomarker name {name!r} contains a reserved token "
                             f"{_RESERVED_TOKENS + (group_column,)}; pyebm matches by substring")
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f"biomarker column {name!r} is not numeric")
    values = df[biomarkers].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("infinite biomarker values")
    if np.isnan(values).all(axis=0).any():
        raise ValueError("a biomarker column is entirely missing")
    code = {value: index for index, value in enumerate(group_values)}
    label_of = {str(label): label for label in labels}
    out = pd.DataFrame({
        "PTID": (df["PTID"].astype(str).to_numpy() if "PTID" in df.columns
                 else [f"s{i:07d}" for i in range(len(df))]),
        "Diagnosis": [label_of[str(v)] for v in df["Diagnosis"]],
        group_column: [code[v] for v in df[group_column]],
    })
    for name in biomarkers:
        out[name] = values[:, biomarkers.index(name)]
    return out.reset_index(drop=True), group_values, tuple(biomarkers)


# ----------------------------------------------------------------------------- statistics

def kendall_distance(a, b):
    """Normalised Kendall tau distance between two orderings (lists of event indices)."""
    ra, rb = np.argsort(np.asarray(a)), np.argsort(np.asarray(b))
    n = len(ra)
    if n < 2:
        return 0.0
    discordant = sum((ra[i] - ra[j]) * (rb[i] - rb[j]) < 0
                     for i in range(n) for j in range(i + 1, n))
    return float(discordant / (n * (n - 1) / 2))


def pair_distances(orderings, pairs):
    return [kendall_distance(orderings[a], orderings[b]) for a, b in pairs]


def exact_decision(count, done, budget, threshold, rule="le"):
    """Reject / not / undetermined over every completion of ``budget - done`` unknown fits.

    ``count`` is the number of completed fits whose statistic is >= the observed one
    (ties are exceedances); rejection under ``le`` means (1 + count_final) <= threshold*(budget+1).
    The same rule the pre-registered simulation runner used for the pair and maximum endpoints.
    """
    if not 0 <= done <= budget or count > done:
        raise ValueError("invalid count/done/budget")
    threshold = Fraction(str(threshold))
    if rule == "le":
        def ok(n):
            return n * threshold.denominator <= (budget + 1) * threshold.numerator
    elif rule == "strict":
        def ok(n):
            return n * threshold.denominator < (budget + 1) * threshold.numerator
    else:
        raise ValueError("rule must be 'le' or 'strict'")
    lower = 1 + count
    if ok(lower + budget - done):
        return True
    if not ok(lower):
        return False
    return None


# ----------------------------------------------------------------------------- relabelling operators

def _stream(seed, name, index, stream):
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), tag, int(index), stream]))


def scheme_definitions(spec: CompareSpec, schemes):
    """Expand scheme names into {name: {'stratify', 'groups', 'pair_index'}}."""
    out = {}
    all_groups = list(range(spec.n_groups))
    for scheme in schemes:
        if scheme == "unrestricted":
            out[scheme] = {"stratify": "none", "groups": all_groups, "pair_index": None}
        elif scheme == "diagnosis":
            out[scheme] = {"stratify": "diagnosis", "groups": all_groups, "pair_index": None}
        elif scheme == "diagnosis_pair":
            if spec.n_groups < 3:
                continue        # identical to 'diagnosis' with two groups
            for index, pair in enumerate(spec.pairs):
                out[f"diagnosis_pair_{spec.pair_name(pair)}"] = {
                    "stratify": "diagnosis", "groups": list(pair), "pair_index": index}
        elif scheme == "diagnosis_count":
            # repair operator: strata of diagnosis x number of observed biomarkers
            out[scheme] = {"stratify": "diagnosis_count", "groups": all_groups, "pair_index": None}
        else:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES + OPTIONAL_SCHEMES}")
    return out


def permute_labels(df, spec: CompareSpec, name, definition, index):
    """Exchange group labels among the allowed groups within strata; nothing else moves."""
    rng = _stream(spec.seed, name, index, PERMUTATION_STREAM)
    labels = df[spec.group_column].to_numpy().copy()
    if definition["stratify"] == "none":
        strata = np.zeros(len(df), dtype=int)
    elif definition["stratify"] == "diagnosis":
        strata = df["Diagnosis"].to_numpy()
    elif definition["stratify"] == "diagnosis_count":
        observed = df[list(spec.biomarkers)].notna().sum(axis=1).to_numpy()
        strata = np.asarray([f"{d}:{k}" for d, k in zip(df["Diagnosis"].to_numpy(), observed)])
    else:
        raise ValueError(f"unknown stratification {definition['stratify']!r}")
    allowed = np.isin(labels, definition["groups"])
    for stratum in np.unique(strata):
        rows = np.flatnonzero((strata == stratum) & allowed)
        labels[rows] = rng.permutation(labels[rows])
    out = df.copy()
    out[spec.group_column] = labels
    return out


def resample_within_strata(df, spec: CompareSpec, index):
    """Bootstrap subjects within every group-by-diagnosis stratum (composition held fixed)."""
    rng = _stream(spec.seed, "stability", index, RESAMPLE_STREAM)
    keys = list(zip(df[spec.group_column].to_numpy(), df["Diagnosis"].to_numpy()))
    chosen = []
    for key in sorted(set(keys), key=str):
        rows = np.flatnonzero([k == key for k in keys])
        chosen.extend(rng.choice(rows, size=len(rows), replace=True).tolist())
    out = df.iloc[sorted(chosen)].reset_index(drop=True)
    out["PTID"] = [f"r{index:05d}_{i:07d}" for i in range(len(out))]
    return out


# ----------------------------------------------------------------------------- one fit

def _saebm_fit(df, spec: CompareSpec):
    """Per-group stage-aware EBM through pysaebm (conjugate priors); complete data only."""
    import shutil
    import tempfile
    try:
        from pysaebm import run_ebm
    except ImportError as exc:
        raise ImportError("estimator='saebm' requires the optional package pysaebm") from exc
    names = list(spec.biomarkers)
    if df[names].isna().any().any():
        return FitResult(None, "error", {"error": "saebm requires complete biomarker data"})
    orderings, seconds, healthy = [], [], []
    started = time.monotonic()
    import logging
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(max(previous_level, logging.WARNING))   # pysaebm logs every 20th MCMC iteration
    for code in range(spec.n_groups):
        sub = df[df[spec.group_column] == code].reset_index(drop=True)
        rows = [{"participant": i, "biomarker": b, "measurement": float(r[b]),
                 "diseased": bool(r["Diagnosis"] != spec.labels[0])}
                for i, r in sub.iterrows() for b in names]
        tmp = tempfile.mkdtemp(prefix="saebm_")
        try:
            path = os.path.join(tmp, "g.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            t0 = time.monotonic()
            res = run_ebm(algorithm="conjugate_priors", data_file=path,
                          output_dir=os.path.join(tmp, "out"), n_iter=spec.saebm_iterations,
                          burn_in=spec.saebm_burn_in, skip_heatmap=True, skip_traceplot=True,
                          save_results=False, save_details=True, seed=(spec.seed + code) % 100000)
            seconds.append(time.monotonic() - t0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        position = res["order_with_highest_ll"]
        orderings.append(np.asarray([names.index(b) for b, _ in
                                     sorted(position.items(), key=lambda kv: kv[1])]))
        healthy.append(float(res.get("healthy_ratio", float("nan"))))
    root.setLevel(previous_level)
    diagnostics = {"estimator": "saebm", "group_seconds": seconds, "healthy_ratio": healthy,
                   "seconds": time.monotonic() - started, "quality_flags": []}
    return FitResult(orderings, "ok", diagnostics)


def fit_once(df, spec: CompareSpec) -> FitResult:
    """Fit every group ordering with the chosen estimator; never raises on a fit failure."""
    if spec.estimator == "saebm":
        return _saebm_fit(df, spec)
    config = EngineConfig(mode=spec.consensus, expected_events=len(spec.biomarkers),
                          group_column=spec.group_column,
                          group_values=tuple(range(spec.n_groups)), labels=tuple(spec.labels),
                          biomarker_names=tuple(spec.biomarkers))
    with fast_likelihood_context(spec.fast_likelihood):
        if spec.estimator == "standard":
            return fit_orderings(df, config)
        return fit_invariant_orderings(df, config, variant=spec.estimator)


# ----------------------------------------------------------------------------- workers

_WORK: dict = {}


class _FitTimeout(TimeoutError):
    pass


def _alarm(signum, frame):
    raise _FitTimeout("fit exceeded fit_timeout_s")


def _init_worker(df, spec, definitions):
    _WORK.update(df=df, spec=spec, definitions=definitions)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)


def _run_task(task):
    kind, name, index = task
    df, spec, definitions = _WORK["df"], _WORK["spec"], _WORK["definitions"]
    started = time.monotonic()
    record = {"kind": kind, "scheme": name, "index": index}
    if hasattr(signal, "SIGALRM"):
        signal.alarm(int(spec.fit_timeout_s))
    try:
        if kind == "permutation":
            data = permute_labels(df, spec, name, definitions[name], index)
        elif kind == "stability":
            data = resample_within_strata(df, spec, index)
        else:
            data = df
        result = fit_once(data, spec)
        record["status"] = result.status
        if result.ok:
            record["distances"] = pair_distances(result.orderings, spec.pairs)
            if kind != "permutation":
                record["orderings"] = [list(map(int, o)) for o in result.orderings]
        d = result.diagnostics
        record["diagnostics"] = {k: d.get(k) for k in
                                 ("quality_flags", "error", "effective_sample_size",
                                  "pooled_mixture_cached", "seconds")}
        if kind == "observed":
            record["full_diagnostics"] = _jsonable(d)
    except _FitTimeout:
        record.update(status="timeout", diagnostics={"error": "timeout"})
    except Exception as exc:
        record.update(status="error", diagnostics={"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    record["seconds"] = time.monotonic() - started
    return record


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Runner:
    """Runs tasks inline (workers=1) or in a spawn-based process pool; one API for both.

    A ``concurrent.futures`` pool is used rather than ``multiprocessing.Pool`` because a worker that
    dies (for example when an unguarded script is re-imported by the spawned process) surfaces as
    ``BrokenProcessPool`` instead of an indefinite hang.
    """

    def __init__(self, df, spec, definitions, workers):
        self.workers = max(1, int(workers))
        self.pool = None
        if self.workers == 1:
            _init_worker(df, spec, definitions)
        else:
            from concurrent.futures import ProcessPoolExecutor
            previous = {}
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                previous[var] = os.environ.get(var)
                os.environ[var] = "1"          # inherited by the spawned workers only
            try:
                self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"),
                                                initializer=_init_worker, initargs=(df, spec, definitions))
            finally:
                for var, value in previous.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value

    def map(self, tasks):
        if self.pool is None:
            for task in tasks:
                yield _run_task(task)
            return
        from concurrent.futures import as_completed
        from concurrent.futures.process import BrokenProcessPool
        futures = [self.pool.submit(_run_task, task) for task in tasks]
        try:
            for future in as_completed(futures):
                yield future.result()
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "a CONCORD worker process died before returning a result. The usual cause is a script that "
                "calls concord.compare(..., workers>1) outside an `if __name__ == '__main__':` guard, so the "
                "spawned workers re-run it; other causes are out-of-memory or a crash in a native library. "
                "Run with workers=1 to see the underlying error.") from exc

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


# ----------------------------------------------------------------------------- result

@dataclass
class ComparisonResult:
    spec: dict
    groups: list
    biomarkers: list
    status: str
    orderings: dict                 # group value -> biomarker names in estimated order
    positions: dict                 # biomarker -> {group value: position}
    distances: dict                 # 'pairs': {pair: d}, 'max': d, 'max_pair': pair
    tests: dict                     # scheme -> {'pairs': {...}, 'max': {...}, 'nperm', ...}
    decisions: dict                 # summary of the frozen rule
    composition: dict               # counts, proportions, chi-square
    effective_sample_size: dict     # group value -> ESS (n for the standard estimator)
    stability: dict                 # per-group resampling summary
    paired_difference: dict         # stratified minus unrestricted p (pairs and max)
    fits: dict                      # counts of ok/failed/timeout fits by kind
    timing: dict
    provenance: dict
    observed_diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(asdict(self))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=True) + "\n")

    def summary(self):
        lines = [f"{PACKAGE_VERSION}: estimator={self.spec['estimator']}, "
                 f"groups={self.groups}, {len(self.biomarkers)} events, status={self.status}"]
        if self.status != "ok":
            lines.append(f"observed fit failed: {self.observed_diagnostics.get('error')}")
            return "\n".join(lines)
        lines.append("orderings:")
        for group, order in self.orderings.items():
            lines.append(f"  {group}: {' < '.join(order)}")
        lines.append("pairwise Kendall distance: " + ", ".join(
            f"{k}={v:.3f}" for k, v in self.distances["pairs"].items())
                     + f"; max={self.distances['max']:.3f} ({self.distances['max_pair']})")
        lines.append("permutation tests (p over completed fits; decision under planned budget):")
        for scheme, test in self.tests.items():
            parts = []
            for pair, cell in test["pairs"].items():
                if cell["p"] is None:
                    continue
                flag = {True: "*", False: "", None: "?"}[cell["reject"]]
                parts.append(f"{pair} p={cell['p']:.3f}{flag}")
            if test.get("max") and test["max"]["p"] is not None:
                flag = {True: "*", False: "", None: "?"}[test["max"]["reject"]]
                parts.append(f"max p={test['max']['p']:.3f}{flag}")
            lines.append(f"  {scheme} (B={test['completed']}/{test['budget']}): " + "; ".join(parts))
        comp = self.composition
        lines.append("composition (counts): " + "; ".join(
            f"{g}: " + "/".join(str(comp['counts'][g][l]) for l in comp['labels'])
            for g in self.groups) + f"  [chi-square p={comp['chi2_p']:.3g}]")
        lines.append("effective sample size: " + ", ".join(
            f"{g}={v:.1f}" for g, v in self.effective_sample_size.items()))
        if self.stability.get("resamples"):
            lines.append(f"stability ({self.stability['resamples']} resamples), mean Kendall "
                         "distance to observed: " + ", ".join(
                             f"{g}={v['mean_distance']:.3f}" for g, v in self.stability["groups"].items()))
        if self.paired_difference.get("pairs"):
            lines.append("stratified minus unrestricted p: " + ", ".join(
                f"{k}={v:+.3f}" for k, v in self.paired_difference["pairs"].items() if v is not None))
        lines.append(f"fits: {self.fits}; wall {self.timing['wall_seconds']/60:.1f} min")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- main entry point

def compare(df, group_column="APOE", labels=("CN", "MCI", "AD"), biomarkers=None,
                      group_order=None, estimator="invariant_min", consensus="repaired",
                      schemes=SCHEMES, B=599, alpha=0.05, rule="le", stop_when_decided=False,
                      stability_resamples=20, seed=0, workers=1, fit_timeout_s=900,
                      fast_likelihood=True, saebm_iterations=2000, saebm_burn_in=500,
                      verbose=True, progress_every=10.0) -> ComparisonResult:
    """Compare separately estimated event orderings between groups.

    See the module docstring for estimators, reference distributions and the rule.
    ``workers`` > 1 runs fits in a spawn pool (each worker caches its own pooled mixture).
    ``stability_resamples`` = 0 skips the resampling diagnostic.
    """
    import multiprocessing
    if multiprocessing.parent_process() is not None:
        raise RuntimeError(
            "concord.compare() was called inside a worker process. With workers > 1 the workers are spawned "
            "and re-import the main script, so put the call under `if __name__ == '__main__':` "
            "(the command-line entry point and importable modules are already safe).")
    wall_started = time.monotonic()
    provenance = {"package_version": PACKAGE_VERSION, "started_utc": _utc()}
    if estimator != "saebm":
        provenance["pyebm"] = verify_pyebm()
    frame, group_values, names = prepare_data(df, group_column, labels, biomarkers, group_order)
    spec = CompareSpec(group_column=group_column, group_values=tuple(group_values),
                       labels=tuple(labels), biomarkers=tuple(names), estimator=estimator,
                       consensus=consensus, fast_likelihood=bool(fast_likelihood),
                       fit_timeout_s=int(fit_timeout_s), seed=int(seed),
                       saebm_iterations=int(saebm_iterations), saebm_burn_in=int(saebm_burn_in))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    B = int(B)
    if B < 0 or stability_resamples < 0:
        raise ValueError("B and stability_resamples must be nonnegative")
    definitions = scheme_definitions(spec, schemes)
    pairs = spec.pairs
    pair_names = [spec.pair_name(p) for p in pairs]
    pair_alpha = Fraction(str(alpha)) / len(pairs)

    def log(message):
        if verbose:
            print(f"[concord {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)

    composition = _composition(frame, spec)
    log(f"{len(frame)} subjects, {spec.n_groups} groups {list(group_values)}, {len(names)} events; "
        f"estimator={estimator}; schemes={list(definitions)}; B={B}; workers={workers}")
    for flag in composition["flags"]:
        log("composition warning: " + flag)

    runner = _Runner(frame, spec, definitions, workers)
    fits = {"ok": 0, "error": 0, "timeout": 0}
    try:
        # 1. observed fit
        observed = next(runner.map([("observed", "observed", 0)]))
        fits[observed["status"] if observed["status"] in fits else "error"] += 1
        if observed["status"] != "ok":
            log(f"observed fit failed: {observed['diagnostics'].get('error')}")
            return _failed_result(spec, group_values, names, observed, composition,
                                  provenance, fits, wall_started)
        distances = np.asarray(observed["distances"])
        log("observed distances " + ", ".join(f"{n}={d:.3f}" for n, d in zip(pair_names, distances))
            + f" (fit {observed['seconds']:.0f}s)")

        # 2. permutation references (+ stability resamples in the same pool)
        counts = {name: {"pairs": np.zeros(len(pairs), dtype=int), "max": 0, "done": 0,
                         "failed": 0, "seconds": 0.0} for name in definitions}
        remaining = {name: list(range(B)) for name in definitions}
        stability_records = []
        stability_todo = list(range(int(stability_resamples)))
        total = B * len(definitions) + len(stability_todo)
        finished = 0
        last_report = time.monotonic()
        batch_size = max(2 * runner.workers, 8)

        def required_determined(name):
            c = counts[name]
            d = definitions[name]
            indexes = [d["pair_index"]] if d["pair_index"] is not None else range(len(pairs))
            # failed fits stay unknown in the planned budget (never a smaller denominator)
            decided = all(exact_decision(int(c["pairs"][i]), c["done"], B, pair_alpha, rule)
                          is not None for i in indexes)
            if d["pair_index"] is None:
                decided = decided and exact_decision(c["max"], c["done"], B, alpha, rule) is not None
            return decided

        while True:
            batch = []
            for name in definitions:
                if not remaining[name]:
                    continue
                if stop_when_decided and required_determined(name):
                    remaining[name] = []
                    continue
                take, remaining[name] = remaining[name][:batch_size], remaining[name][batch_size:]
                batch.extend(("permutation", name, i) for i in take)
            take, stability_todo = stability_todo[:batch_size], stability_todo[batch_size:]
            batch.extend(("stability", "stability", i) for i in take)
            if not batch:
                break
            for record in runner.map(batch):
                finished += 1
                status = record["status"] if record["status"] in fits else "error"
                fits[status] += 1
                if record["kind"] == "stability":
                    stability_records.append(record)
                else:
                    c = counts[record["scheme"]]
                    c["seconds"] += record["seconds"]
                    if record["status"] == "ok":
                        t = np.asarray(record["distances"])
                        c["pairs"] += (t >= distances - 1e-9)
                        c["max"] += int(t.max() >= distances.max() - 1e-9)
                        c["done"] += 1
                    else:
                        c["failed"] += 1
                if verbose and time.monotonic() - last_report >= progress_every:
                    elapsed = time.monotonic() - wall_started
                    rate = finished / max(elapsed, 1e-9)
                    eta = (total - finished) / rate if rate > 0 else float("nan")
                    log(f"{finished}/{total} fits ({100 * finished / total:.0f}%), "
                        f"{60 * rate:.1f} fits/min, eta {eta / 60:.0f} min")
                    last_report = time.monotonic()
    finally:
        runner.close()

    # 3. assemble
    tests = {}
    for name, d in definitions.items():
        c = counts[name]
        completed = c["done"]
        unknown = B - completed
        cells = {}
        indexes = [d["pair_index"]] if d["pair_index"] is not None else range(len(pairs))
        for i in indexes:
            count = int(c["pairs"][i])
            cells[pair_names[i]] = {
                "exceedances": count,
                "p": (1 + count) / (completed + 1) if completed else None,
                "p_bounds": [(1 + count) / (B + 1), (1 + count + unknown) / (B + 1)],
                "reject": exact_decision(count, completed, B, pair_alpha, rule),
                "threshold": float(pair_alpha)}
        entry = {"definition": d, "budget": B, "completed": completed, "failed": c["failed"],
                 "pairs": cells, "max": None, "seconds": c["seconds"]}
        if d["pair_index"] is None:
            entry["max"] = {
                "exceedances": c["max"],
                "p": (1 + c["max"]) / (completed + 1) if completed else None,
                "p_bounds": [(1 + c["max"]) / (B + 1), (1 + c["max"] + unknown) / (B + 1)],
                "reject": exact_decision(c["max"], completed, B, alpha, rule),
                "threshold": float(alpha)}
        tests[name] = entry
    decisions = {
        "rule": rule, "alpha": alpha, "per_pair_threshold": float(pair_alpha),
        "convention": "p = (1 + #{perm stat >= observed}) / (B + 1); ties are exceedances; "
                      "reject iff p <= threshold; decisions are exact over unfinished fits",
        "primary": "diagnosis", "localisation": "diagnosis_pair", "global": "diagnosis:max",
        "reject_any_pair": {name: any(cell["reject"] is True for cell in test["pairs"].values())
                            for name, test in tests.items()},
        "stop_when_decided": bool(stop_when_decided)}

    orderings = {str(g): [names[i] for i in observed["orderings"][k]] for k, g in enumerate(group_values)}
    positions = {name: {str(g): int(np.flatnonzero(np.asarray(observed["orderings"][k]) == j)[0])
                        for k, g in enumerate(group_values)} for j, name in enumerate(names)}
    max_index = int(np.argmax(distances))
    distance_block = {"pairs": dict(zip(pair_names, map(float, distances))),
                      "max": float(distances[max_index]), "max_pair": pair_names[max_index],
                      "statistic": "normalised Kendall tau distance"}
    ess = observed["diagnostics"].get("effective_sample_size")
    if ess is None:
        sizes = frame[group_column].value_counts()
        ess = {str(g): float(sizes.get(k, 0)) for k, g in enumerate(group_values)}
    else:
        ess = {str(g): float(ess[str(k)]) for k, g in enumerate(group_values)}
    paired = {"pairs": {}, "max": None,
              "note": "p(diagnosis) - p(unrestricted); a large positive value marks composition-driven "
                      "significance under the unrestricted reference"}
    if "diagnosis" in tests and "unrestricted" in tests:
        for pn in pair_names:
            a, b = tests["diagnosis"]["pairs"][pn]["p"], tests["unrestricted"]["pairs"][pn]["p"]
            paired["pairs"][pn] = None if a is None or b is None else a - b
        a, b = tests["diagnosis"]["max"]["p"], tests["unrestricted"]["max"]["p"]
        paired["max"] = None if a is None or b is None else a - b
    stability = _stability_summary(stability_records, observed["orderings"], group_values, names,
                                   int(stability_resamples))
    timing = {"wall_seconds": time.monotonic() - wall_started, "observed_fit_seconds": observed["seconds"],
              "mean_fit_seconds": float(np.mean([t["seconds"] / max(t["completed"] + t["failed"], 1)
                                                 for t in tests.values()])) if tests else None,
              "workers": runner.workers}
    provenance.update(finished_utc=_utc(), engine_diagnostics_keys=sorted(observed.get("full_diagnostics", {})))
    result = ComparisonResult(
        spec=asdict(spec) | {"B": B, "schemes": list(definitions), "stability_resamples": int(stability_resamples)},
        groups=[str(g) for g in group_values], biomarkers=list(names), status="ok",
        orderings=orderings, positions=positions, distances=distance_block, tests=tests,
        decisions=decisions, composition=composition, effective_sample_size=ess,
        stability=stability, paired_difference=paired, fits=fits, timing=timing,
        provenance=provenance, observed_diagnostics=observed.get("full_diagnostics", {}))
    log("done")
    return result


compare_orderings = compare   # name used by the research scripts


def _failed_result(spec, group_values, names, observed, composition, provenance, fits, wall_started):
    return ComparisonResult(spec=asdict(spec), groups=[str(g) for g in group_values],
                            biomarkers=list(names), status=observed["status"], orderings={},
                            positions={}, distances={}, tests={}, decisions={}, composition=composition,
                            effective_sample_size={}, stability={}, paired_difference={}, fits=fits,
                            timing={"wall_seconds": time.monotonic() - wall_started}, provenance=provenance,
                            observed_diagnostics=observed.get("diagnostics", {}))


def _composition(frame, spec):
    from scipy.stats import chi2_contingency
    table = pd.crosstab(frame[spec.group_column], frame["Diagnosis"]).reindex(
        index=range(spec.n_groups), columns=list(spec.labels), fill_value=0)
    counts = {str(g): {str(l): int(table.loc[k, l]) for l in spec.labels}
              for k, g in enumerate(spec.group_values)}
    proportions = {g: {l: (v / sum(row.values()) if sum(row.values()) else float("nan"))
                       for l, v in row.items()} for g, row in counts.items()}
    flags = []
    for g, row in counts.items():
        empty = [l for l, v in row.items() if v == 0]
        if empty:
            flags.append(f"group {g} has no subjects with diagnosis {empty}; standardisation to a "
                         "common composition is only partial for this stratum")
        if sum(row.values()) < 30:
            flags.append(f"group {g} has only {sum(row.values())} subjects")
    try:
        chi2, p, dof, _ = chi2_contingency(table.to_numpy())
        chi2, p, dof = float(chi2), float(p), int(dof)
    except ValueError:
        chi2 = p = float("nan"); dof = 0
    pooled = {str(l): float((frame["Diagnosis"] == l).mean()) for l in spec.labels}
    minimum = np.min(np.stack([[proportions[str(g)][str(l)] for l in spec.labels]
                               for g in spec.group_values]), axis=0)
    minimum = minimum / minimum.sum() if minimum.sum() > 0 else minimum
    return {"labels": [str(l) for l in spec.labels], "counts": counts, "proportions": proportions,
            "pooled": pooled, "min_reference": dict(zip(map(str, spec.labels), map(float, minimum))),
            "chi2": chi2, "chi2_dof": dof, "chi2_p": p, "flags": flags}


def _stability_summary(records, observed_orderings, group_values, names, requested):
    ok = [r for r in records if r["status"] == "ok"]
    out = {"resamples": requested, "completed": len(ok), "failed": len(records) - len(ok), "groups": {}}
    if not ok:
        return out
    for k, g in enumerate(group_values):
        ref = np.asarray(observed_orderings[k])
        dists = [kendall_distance(r["orderings"][k], ref) for r in ok]
        pos = np.asarray([np.argsort(np.asarray(r["orderings"][k])) for r in ok])   # event -> position
        out["groups"][str(g)] = {
            "mean_distance": float(np.mean(dists)), "sd_distance": float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
            "position_sd": {name: float(pos[:, j].std(ddof=1)) if len(ok) > 1 else 0.0
                            for j, name in enumerate(names)}}
    # distance between resampled orderings of different groups, for reference against the observed
    out["between_group_distance_mean"] = {}
    for a, b in combinations(range(len(group_values)), 2):
        out["between_group_distance_mean"][f"{group_values[a]}-{group_values[b]}"] = float(np.mean(
            [kendall_distance(r["orderings"][a], r["orderings"][b]) for r in ok]))
    return out


def _utc():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(prog="concord", description="CONCORD: compare event-based model orderings between groups with a composition-invariant estimator and a stratified permutation test.")
    ap.add_argument("input", help="CSV with PTID, Diagnosis, the group column and biomarker columns")
    ap.add_argument("--group", default="APOE")
    ap.add_argument("--labels", default="CN,MCI,AD", help="diagnosis labels in stage order")
    ap.add_argument("--biomarkers", default=None, help="comma list; default: every other column")
    ap.add_argument("--estimator", default="invariant_min", choices=ESTIMATORS)
    ap.add_argument("--consensus", default="repaired", choices=("repaired", "original"))
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--B", type=int, default=599)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--stop-when-decided", action="store_true")
    ap.add_argument("--stability", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fit-timeout", type=int, default=900)
    ap.add_argument("--no-fast-likelihood", action="store_true")
    ap.add_argument("--output", type=Path, default=None, help="JSON result path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.input)
    result = compare(
        df, group_column=args.group, labels=tuple(args.labels.split(",")),
        biomarkers=args.biomarkers.split(",") if args.biomarkers else None,
        estimator=args.estimator, consensus=args.consensus, schemes=tuple(args.schemes.split(",")),
        B=args.B, alpha=args.alpha, stop_when_decided=args.stop_when_decided,
        stability_resamples=args.stability, seed=args.seed, workers=args.workers,
        fit_timeout_s=args.fit_timeout, fast_likelihood=not args.no_fast_likelihood,
        verbose=not args.quiet)
    if args.output:
        result.to_json(args.output)
    print(result.summary())
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
