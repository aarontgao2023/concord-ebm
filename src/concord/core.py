"""CONCORD: compare event-based model orderings between groups at common diagnostic proportions.

    import concord
    result = concord.compare(df, group_column="APOE")
    print(result.summary())

The input is a data frame with one row per participant: PTID (optional), Diagnosis, one group
column and the biomarker columns (NaN marks a missing measurement). The result holds each group's
ordering, the normalized Kendall distance between every pair of groups, the permutation P value,
adjusted P value and Bonferroni decision of every comparison, the group-by-diagnosis counts, the
common proportions, the weights and effective sample sizes, and the stability of each ordering
under bootstrap refits.

Models (``estimator=``):
  concord       CONCORD (default): one abnormality model, a two-component Gaussian mixture per
                biomarker, fitted to the pooled sample of the compared groups; participants
                weighted by diagnosis to common proportions; each group's ordering estimated from
                the weighted abnormality probabilities with the DEBM ranking consensus
  pooled_score  pooled-score DEBM: the same pooled abnormality model with all weights equal to one
  separate      separately fitted DEBM (pyebm 2.0.3): mixtures initialized from all groups, then
                group-specific mixtures and orderings
  saebm         stage-aware EBM (SA-EBM, optional package pysaebm) fitted in each group with CN
                participants as non-diseased; the ordering with the highest likelihood is
                reported; complete data only
Earlier names are accepted: invariant_min (concord), shared (pooled_score) and standard
(separate). invariant_pooled weights to the diagnostic proportions of the pooled sample; it was
not used in the paper.

Common proportions (``common_proportions=``, CONCORD only):
  None or 'min'  the smallest proportion of each diagnosis across the compared groups, rescaled to
                 sum to one; this gives the smallest possible largest weight. It is recomputed
                 after every unrestricted relabeling; within-diagnosis relabelings and bootstrap
                 refits within group-by-diagnosis cells leave it unchanged.
  fixed          a mapping {label: proportion} or a sequence in label order, summing to one, held
                 fixed in the observed fit, every relabeling and every bootstrap refit. Every
                 diagnosis with a positive proportion must be present in every compared group.
                 A diagnosis left out of a mapping gets proportion 0, and a proportion of 0 gives
                 the participants with that diagnosis weight 0 (they still enter the pooled
                 abnormality model). concord.common_proportions() applies the minimum rule
                 jointly to any number of groups, for example to the six cohort-by-genotype
                 groups of two cohorts; this is how the paper's proportions were computed.

Search (``search=``): 'continued' (default) continues the adjacent-swap search of the ranking
consensus until no swap lowers the loss; 'pyebm' is the unmodified pyebm 2.0.3 search, which stops
after the first accepted swap. Earlier names: repaired (continued) and original (pyebm).

Permutation schemes (``schemes=``). Each relabels only the group column; after each relabeling
every label-dependent step is repeated (separately fitted mixtures, weights and, under the
minimum rule, common proportions, and the orderings):
  unrestricted               labels exchanged among all participants of the compared groups
  within_diagnosis           labels of all groups exchanged separately among the participants of
                             each diagnosis, so every group keeps its diagnostic counts
  pairwise_within_diagnosis  one scheme per pair of groups: only that pair's labels are exchanged
                             within diagnosis; the other groups keep their labels and stay in the
                             pooled abnormality model. With two groups this is within_diagnosis.
The default (``schemes=None``) is within-diagnosis permutation for two groups and pairwise
within-diagnosis permutation for three or more, as in the paper's APOE analyses. Unrestricted
permutation runs only when requested; it does not keep each group's diagnostic counts.
Earlier names: diagnosis (within_diagnosis) and diagnosis_pair (pairwise_within_diagnosis).

Tests: P = (1 + E) / (B + 1), where E is the number of relabelings whose distance is at least the
observed distance. Each of the n_pairs comparisons is tested at alpha / n_pairs (Bonferroni) and
the adjusted P value is min(1, n_pairs * P); with B = 599 and three groups a comparison is rejected
when E <= 9. When every planned relabeling was fitted, P, the adjusted P value and the decision
agree. When some relabelings failed or were not run (``stop_when_decided``), the reported ``p``
and ``p_adjusted`` use the completed relabelings only, P = (1 + E) / (completed + 1), whereas the
decision ``reject`` and the bounds ``p_bounds`` refer to the planned B: the upper bound counts
every missing relabeling as an exceedance, and a comparison is rejected (True) or not (False)
only if the decision is the same whatever the missing relabelings would have given; otherwise
it is undetermined (None). With ``stop_when_decided`` a scheme stops once no further relabeling
can change any of its decisions.

The pooled mixture depends on the measurements and the diagnoses (its components are initialized
from the CN and AD participants), not on the group labels, so it is fitted once per process and
reused after every relabeling (concord.invariant). Requires the pinned pyebm 2.0.3
(concord._pyebm).
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field
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

import numpy as np
import pandas as pd

from ._pyebm import verify_pyebm
from ._version import __version__
from .engine import EngineConfig, FitResult, fit_orderings
from .invariant import fit_invariant_orderings
from .likelihood import fast_likelihood_context

PACKAGE_VERSION = f"concord-{__version__}"
ESTIMATORS = ("concord", "pooled_score", "separate", "saebm")
ESTIMATOR_ALIASES = {"invariant_min": "concord", "shared": "pooled_score", "standard": "separate"}
EXTRA_ESTIMATORS = ("invariant_pooled",)      # weights to the pooled diagnostic proportions; not in the paper
SEARCHES = ("continued", "pyebm")
SEARCH_ALIASES = {"repaired": "continued", "original": "pyebm"}
SCHEMES = ("unrestricted", "within_diagnosis", "pairwise_within_diagnosis")    # default: CompareSpec.default_schemes
SCHEME_ALIASES = {"diagnosis": "within_diagnosis", "diagnosis_pair": "pairwise_within_diagnosis"}
_VARIANT = {"concord": "invariant_min", "pooled_score": "shared", "invariant_pooled": "invariant_pooled"}
_ENGINE_MODE = {"continued": "repaired", "pyebm": "original"}
PERMUTATION_STREAM = 61000000      # random-stream constants; relabelings are reproducible from the seed
RESAMPLE_STREAM = 62000000
_RESERVED_TOKENS = ("PTID", "Diagnosis", "EXAMDATE")   # pyebm drops columns by substring
_SUM_TOLERANCE = 1e-3


# ----------------------------------------------------------------------------- names

def _canonical(value, names, aliases, what):
    key = str(value)
    if key in names:
        return key
    if key in aliases:
        return aliases[key]
    raise ValueError(f"unknown {what} {value!r}; choose from {names}")


def _canonical_estimator(name):
    return _canonical(name, ESTIMATORS + EXTRA_ESTIMATORS, ESTIMATOR_ALIASES, "estimator")


def _canonical_search(name):
    return _canonical(name, SEARCHES, SEARCH_ALIASES, "search")


def _canonical_scheme(name):
    return _canonical(name, SCHEMES, SCHEME_ALIASES, "permutation scheme")


def _code_names(labels):
    """[(index of the pyebm diagnosis code, name)]. pyebm codes the first label as CN, the last as AD
    and every other label as MCI."""
    labels = [str(label) for label in labels]
    out = [(0, labels[0])]
    if len(labels) > 2:
        out.append((1, "+".join(labels[1:-1])))
    out.append((2, labels[-1]))
    return out


# ----------------------------------------------------------------------------- common proportions

def _fixed_proportions(value, labels):
    """Validate fixed common proportions; returns a tuple in label order that sums to one."""
    labels = tuple(labels)
    if len(labels) not in (2, 3):
        raise ValueError("fixed common proportions require two or three diagnosis labels")
    if isinstance(value, (Mapping, pd.Series)):
        given = {str(k): v for k, v in value.items()}
        unknown = sorted(set(given) - set(map(str, labels)))
        if unknown:
            raise ValueError(f"common proportions given for unknown diagnoses {unknown}; labels are {labels}")
        vector = [given.get(str(label), 0.0) for label in labels]
    else:
        vector = list(value)
        if len(vector) != len(labels):
            raise ValueError(f"common proportions need one value per diagnosis label {labels}")
    try:
        vector = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("common proportions must be numbers") from exc
    if not np.isfinite(vector).all() or (vector < 0).any():
        raise ValueError("common proportions must be finite and nonnegative")
    total = float(vector.sum())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"common proportions must sum to one (they sum to {total:.6g})")
    return tuple(float(x) for x in vector / total)


def _resolve_common_proportions(estimator, value, labels):
    """Canonical form: 'min', a tuple of fixed proportions in label order, 'pooled', or None."""
    is_str = isinstance(value, str)
    if estimator == "concord":
        if value is None or (is_str and value == "min"):
            return "min"
        if is_str:
            raise ValueError("common_proportions must be None, 'min' or fixed proportions")
        return _fixed_proportions(value, labels)
    if estimator == "invariant_pooled":
        if value is None or (is_str and value == "pooled"):
            return "pooled"
        raise ValueError("estimator 'invariant_pooled' takes no common_proportions; use estimator='concord'")
    if value is not None:
        raise ValueError(f"common_proportions applies to estimator 'concord' only, not {estimator!r}")
    return None


def _code_vector(proportions, labels):
    """Fixed proportions in label order -> (CN, MCI, AD) in pyebm code order."""
    if len(labels) == 3:
        return tuple(proportions)
    return (proportions[0], 0.0, proportions[1])


def common_proportions(*groups, labels=("CN", "MCI", "AD"), group_column=None, diagnosis_column="Diagnosis"):
    """Common proportions by the minimum rule, applied jointly to any number of groups.

    Each positional argument adds groups:
      - a data frame with a ``diagnosis_column``: one group per value of ``group_column`` when
        that is given, otherwise the whole frame is one group;
      - a count table, i.e. a data frame with one column per label: one group per row;
      - the counts of one group as a mapping {label: count} or a sequence in label order;
      - a two-dimensional array of counts: one group per row, columns in label order.
    Returns {label: proportion}: the smallest proportion of each diagnosis across all groups,
    rescaled to sum to one. Among all common proportions this gives the smallest possible largest
    weight, 1 / sum_d min_g pihat_g(d). Pass the result to compare(common_proportions=...) to hold
    it fixed in every fit, for example after applying the rule to the six cohort-by-genotype
    groups of two cohorts:

        props = concord.common_proportions(cohort_a, cohort_b, group_column="APOE")
        result = concord.compare(cohort_a, group_column="APOE", common_proportions=props)
    """
    labels = tuple(labels)
    if len(set(map(str, labels))) != len(labels) or len(labels) < 2:
        raise ValueError("labels must be at least two distinct diagnosis labels")
    names = [str(label) for label in labels]
    rows = []
    for item in groups:
        if isinstance(item, pd.DataFrame):
            if diagnosis_column in item.columns:
                diagnosis = item[diagnosis_column]
                if diagnosis.isna().any():
                    raise ValueError(f"missing values in column {diagnosis_column!r}")
                diagnosis = diagnosis.astype(str)
                unknown = sorted(set(diagnosis.unique()) - set(names))
                if unknown:
                    raise ValueError(f"Diagnosis values outside labels {labels}: {unknown}")
                if group_column is None:
                    parts = [diagnosis]
                else:
                    if group_column not in item.columns:
                        raise ValueError(f"column {group_column!r} not found")
                    if item[group_column].isna().any():
                        raise ValueError(f"missing values in group column {group_column!r}")
                    parts = [part for _, part in diagnosis.groupby(item[group_column].to_numpy(), sort=True)]
                rows.extend([float((part == name).sum()) for name in names] for part in parts)
            else:
                columns = {str(c): c for c in item.columns}
                missing = [name for name in names if name not in columns]
                if missing:
                    raise ValueError(f"a data frame needs a {diagnosis_column!r} column or one count column "
                                     f"per label; missing {missing}")
                rows.extend(item[[columns[name] for name in names]].to_numpy(dtype=float).tolist())
        elif isinstance(item, (Mapping, pd.Series)):
            given = {str(k): v for k, v in item.items()}
            unknown = sorted(set(given) - set(names))
            if unknown:
                raise ValueError(f"counts given for unknown diagnoses {unknown}; labels are {labels}")
            rows.append([float(given.get(name, 0)) for name in names])
        else:
            array = np.asarray(item, dtype=float)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or array.shape[1] != len(labels):
                raise ValueError(f"counts need one value per diagnosis label {labels}")
            rows.extend(array.tolist())
    if not rows:
        raise ValueError("no groups given")
    counts = np.asarray(rows, dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    totals = counts.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("every group needs at least one participant")
    minimum = (counts / totals[:, None]).min(axis=0)
    if not minimum.sum() > 0:
        raise ValueError("no diagnosis is present in every group, so the minimum rule is undefined")
    return {label: float(value) for label, value in zip(labels, minimum / minimum.sum())}


# ----------------------------------------------------------------------------- specification

@dataclass(frozen=True)
class CompareSpec:
    """Everything a worker needs; picklable and recorded in the result.

    ``estimator`` and ``search`` are stored under their manuscript names (earlier names are
    accepted). ``common_proportions`` is stored as 'min', a tuple of fixed proportions in label
    order, 'pooled' (estimator invariant_pooled) or None (unweighted estimators).
    ``consensus`` is the earlier name of ``search``. ``default_schemes`` gives the permutation
    schemes run when compare() is called with ``schemes=None``.
    """
    group_column: str
    group_values: tuple            # original values in code order 0..G-1
    labels: tuple                  # diagnosis labels in disease order (first CN, last AD)
    biomarkers: tuple
    estimator: str = "concord"
    search: str = "continued"
    fast_likelihood: bool = True
    fit_timeout_s: int = 900
    seed: int = 0
    saebm_iterations: int = 10000
    saebm_burn_in: int = 2500
    common_proportions: object = None
    consensus: InitVar[str | None] = None

    def __post_init__(self, consensus):
        estimator = _canonical_estimator(self.estimator)
        search = _canonical_search(self.search if consensus is None else consensus)
        if len(self.group_values) < 2:
            raise ValueError("at least two groups are required")
        if len(self.labels) < 2:
            raise ValueError("at least two diagnosis labels are required")
        proportions = _resolve_common_proportions(estimator, self.common_proportions, self.labels)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "search", search)
        object.__setattr__(self, "common_proportions", proportions)

    @property
    def n_groups(self):
        return len(self.group_values)

    @property
    def pairs(self):
        return list(combinations(range(self.n_groups), 2))

    @property
    def default_schemes(self):
        """Within-diagnosis permutation for two groups, pairwise within-diagnosis permutation for
        three or more; unrestricted permutation only when requested."""
        return ("within_diagnosis",) if self.n_groups == 2 else ("pairwise_within_diagnosis",)

    def pair_name(self, pair):
        a, b = pair
        return f"{self.group_values[a]}-{self.group_values[b]}"


# ----------------------------------------------------------------------------- data preparation

def prepare_data(df, group_column="APOE", labels=("CN", "MCI", "AD"), biomarkers=None,
                 group_order=None):
    """Validate and encode the input frame for pyebm; returns (frame, group_values, biomarkers).

    Diagnosis values must all be in ``labels``: pyebm silently codes any other value as the
    middle diagnosis, so this is checked here. Group values are encoded 0..G-1 in ``group_order``
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
    """Normalized Kendall distance: the proportion of event pairs that two orderings (lists of
    event indices, earliest first) place in opposite order."""
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
    """Reject (True), not (False) or undetermined (None) over every completion of the
    ``budget - done`` relabelings still unknown.

    ``count`` is the number of completed relabelings whose distance is at least the observed one.
    Under ``le`` a comparison is rejected when (1 + final count) <= threshold * (budget + 1), that is
    when P <= threshold; under ``strict`` when P < threshold.
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


# ----------------------------------------------------------------------------- relabeling

def _stream(seed, name, index, stream):
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), tag, int(index), stream]))


def scheme_definitions(spec: CompareSpec, schemes):
    """Expand scheme names into {test name: definition}.

    A definition holds 'scheme' (the scheme name), 'stratify' ('none' or 'diagnosis'), 'groups'
    (codes of the groups whose labels are exchanged), 'pair_index' (the one comparison tested, or
    None for all comparisons) and 'stream' (the name that seeds the relabelings; the version 0.1
    names are kept so that a given seed gives the same relabelings as before). Pairwise
    within-diagnosis permutation gives one test per pair, named
    'pairwise_within_diagnosis_<group a>-<group b>'; with two groups it is within-diagnosis
    permutation. ``schemes=None`` gives ``spec.default_schemes``.
    """
    if schemes is None:
        schemes = spec.default_schemes
    if isinstance(schemes, str):
        schemes = (schemes,)
    out = {}
    all_groups = list(range(spec.n_groups))
    for scheme in (_canonical_scheme(s) for s in schemes):
        if scheme == "unrestricted":
            out[scheme] = {"scheme": scheme, "stratify": "none", "groups": all_groups,
                           "pair_index": None, "stream": "unrestricted"}
        elif scheme == "within_diagnosis" or spec.n_groups < 3:
            out["within_diagnosis"] = {"scheme": "within_diagnosis", "stratify": "diagnosis",
                                       "groups": all_groups, "pair_index": None, "stream": "diagnosis"}
        else:
            for index, pair in enumerate(spec.pairs):
                name = spec.pair_name(pair)
                out[f"pairwise_within_diagnosis_{name}"] = {
                    "scheme": scheme, "stratify": "diagnosis", "groups": list(pair),
                    "pair_index": index, "stream": f"diagnosis_pair_{name}"}
    return out


def permute_labels(df, spec: CompareSpec, name, definition, index):
    """Exchange group labels among the allowed groups within strata; nothing else moves."""
    rng = _stream(spec.seed, definition.get("stream", name), index, PERMUTATION_STREAM)
    labels = df[spec.group_column].to_numpy().copy()
    if definition["stratify"] == "none":
        strata = np.zeros(len(df), dtype=int)
    elif definition["stratify"] == "diagnosis":
        strata = df["Diagnosis"].to_numpy()
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
    """Bootstrap participants within every group-by-diagnosis cell (cell counts held fixed)."""
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
    """SA-EBM fitted in each group through pysaebm (conjugate priors) with CN participants as
    non-diseased; reports the ordering with the highest likelihood; complete data only."""
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
    """Fit every group's ordering with the chosen estimator; never raises on a fit failure."""
    if spec.estimator == "saebm":
        return _saebm_fit(df, spec)
    config = EngineConfig(mode=_ENGINE_MODE[spec.search], expected_events=len(spec.biomarkers),
                          group_column=spec.group_column,
                          group_values=tuple(range(spec.n_groups)), labels=tuple(spec.labels),
                          biomarker_names=tuple(spec.biomarkers))
    with fast_likelihood_context(spec.fast_likelihood):
        if spec.estimator == "separate":
            return fit_orderings(df, config)
        reference = (_code_vector(spec.common_proportions, spec.labels)
                     if isinstance(spec.common_proportions, tuple) else None)
        return fit_invariant_orderings(df, config, variant=_VARIANT[spec.estimator], reference=reference)


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

_FLAG = {True: "*", False: "", None: "?"}


@dataclass
class ComparisonResult:
    spec: dict
    groups: list
    biomarkers: list
    status: str
    orderings: dict                 # group -> biomarker names, earliest event first
    positions: dict                 # biomarker -> {group: position, 0 = earliest}
    distances: dict                 # 'pairs': {pair: normalized Kendall distance}, 'statistic'
    tests: dict                     # test name -> {'scheme', 'definition', 'budget', 'completed',
                                    #   'failed', 'pairs': {pair: cell}, 'seconds'}
    decisions: dict                 # Bonferroni decisions by scheme and pair, and the rule
    composition: dict               # counts, proportions, minimum-rule proportions, chi-square
    common_proportions: dict | None  # {diagnosis: proportion} used in the observed fit
    weights: dict                   # group -> {diagnosis: weight}; None where the group has none
    largest_weight: float | None
    effective_sample_size: dict     # group -> Kish effective sample size (group size if unweighted)
    stability: dict                 # bootstrap refits within group-by-diagnosis cells
    paired_difference: dict         # within-diagnosis minus unrestricted P, per comparison
    fits: dict                      # counts of ok/failed/timed-out fits
    timing: dict
    provenance: dict
    observed_diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(asdict(self))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=True) + "\n")

    def summary(self):
        spec = self.spec
        lines = [f"{PACKAGE_VERSION}: estimator={spec.get('estimator')}, search={spec.get('search')}, "
                 f"groups={self.groups}, {len(self.biomarkers)} biomarkers, status={self.status}"]
        if self.status != "ok":
            lines.append(f"observed fit failed: {self.observed_diagnostics.get('error')}")
            return "\n".join(lines)
        lines.append("orderings (earliest event first):")
        for group, order in self.orderings.items():
            lines.append(f"  {group}: {' < '.join(order)}")
        lines.append("normalized Kendall distance: " + ", ".join(
            f"{k}={v:.3f}" for k, v in self.distances["pairs"].items()))
        if self.tests:
            n_pairs = self.decisions["n_pairs"]
            lines.append(f"permutation tests (P over completed relabelings; adjusted P = min(1, {n_pairs} x P); "
                         f"* rejected at {self.decisions['alpha']:g}/{n_pairs} over all planned relabelings, "
                         "? undetermined):")
            for name, test in self.tests.items():
                parts = []
                for pair, cell in test["pairs"].items():
                    flag = _FLAG[cell["reject"]]
                    if cell["p"] is None:
                        low, high = cell["p_bounds"]
                        parts.append(f"{pair} P in [{low:.3f}, {high:.3f}]{flag}")
                    else:
                        parts.append(f"{pair} P={cell['p']:.3f}, adjusted P={cell['p_adjusted']:.3f}{flag}")
                lines.append(f"  {name} ({test['completed']}/{test['budget']} relabelings): " + "; ".join(parts))
            if any(test["scheme"] == "unrestricted" for test in self.tests.values()):
                lines.append("  note: unrestricted permutation does not keep each group's diagnostic counts; "
                             "the within-diagnosis tests are the recommended ones")
        comp = self.composition
        lines.append("composition (" + "/".join(comp["labels"]) + " counts): " + "; ".join(
            f"{g} " + "/".join(str(comp["counts"][g][l]) for l in comp["labels"]) for g in self.groups)
            + f"  [chi-square P={comp['chi2_p']:.3g}]")
        if self.common_proportions:
            rule = spec.get("common_proportions")
            how = "fixed" if isinstance(rule, (list, tuple)) else {"min": "minimum rule",
                                                                    "pooled": "pooled sample"}.get(rule, str(rule))
            lines.append(f"common proportions ({how}): " + ", ".join(
                f"{k} {v:.3f}" for k, v in self.common_proportions.items()))
        if self.weights:
            names = list(next(iter(self.weights.values())))
            line = "weights (" + "/".join(names) + "): " + "; ".join(
                f"{g} " + "/".join("-" if w is None else f"{w:.2f}" for w in row.values())
                for g, row in self.weights.items())
            if self.largest_weight is not None:
                line += f"; largest weight {self.largest_weight:.2f}"
            lines.append(line)
        lines.append("effective sample size: " + ", ".join(
            f"{g}={v:.1f}" for g, v in self.effective_sample_size.items()))
        if self.stability.get("resamples"):
            lines.append(f"stability ({self.stability['resamples']} bootstrap refits), mean Kendall "
                         "distance to the observed ordering: " + ", ".join(
                             f"{g}={v['mean_distance']:.3f}" for g, v in self.stability["groups"].items()))
        if self.paired_difference.get("pairs"):
            lines.append("within-diagnosis minus unrestricted P: " + ", ".join(
                f"{k}={v:+.3f}" for k, v in self.paired_difference["pairs"].items() if v is not None))
        lines.append(f"fits: {self.fits}; wall {self.timing['wall_seconds']/60:.1f} min")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- main entry point

def compare(df, group_column="APOE", labels=("CN", "MCI", "AD"), biomarkers=None,
            group_order=None, estimator="concord", search="continued",
            schemes=None, B=599, alpha=0.05, rule="le", stop_when_decided=False,
            stability_resamples=20, seed=0, workers=1, fit_timeout_s=900,
            fast_likelihood=True, saebm_iterations=10000, saebm_burn_in=2500,
            verbose=True, progress_every=10.0, *, common_proportions=None,
            consensus=None) -> ComparisonResult:
    """Compare the event orderings of two or more groups.

    See the module docstring for the models, common proportions, permutation schemes and tests.

    df                   one row per participant: Diagnosis, the group column, the biomarker
                         columns (NaN allowed) and optionally PTID
    group_column         column that defines the groups
    labels               diagnosis labels in disease order (first CN, last AD)
    biomarkers           biomarker columns (default: every other column)
    group_order          order of the groups in the output (default: sorted values)
    estimator            'concord' (default), 'pooled_score', 'separate' or 'saebm'
    search               ranking-consensus search: 'continued' (default) or 'pyebm'
    schemes              any of 'unrestricted', 'within_diagnosis', 'pairwise_within_diagnosis';
                         None (default): within_diagnosis for two groups and
                         pairwise_within_diagnosis for three or more. Unrestricted permutation
                         runs only when requested
    B                    relabelings per scheme (599)
    alpha                family-wise level; each comparison is tested at alpha / number of pairs
    rule                 'le' rejects when P <= alpha / n_pairs, 'strict' when P < alpha / n_pairs
    stop_when_decided    stop a scheme once no further relabeling can change its decisions
    stability_resamples  bootstrap refits within group-by-diagnosis cells (0 skips them)
    seed                 seed of the relabelings and bootstrap resamples
    workers              processes; with workers > 1 fits run in a spawn pool, so call compare()
                         under ``if __name__ == '__main__':``
    fit_timeout_s        time limit per fit in seconds (where SIGALRM is available)
    fast_likelihood      evaluate the mixture objective with concord.likelihood
    saebm_iterations     SA-EBM iterations (10,000)
    saebm_burn_in        SA-EBM burn-in iterations (2,500)
    common_proportions   CONCORD only: None or 'min' (minimum rule over the compared groups), or
                         fixed proportions as {label: proportion} or a sequence in label order.
                         A diagnosis left out of a mapping gets proportion 0; participants with a
                         diagnosis of proportion 0 get weight 0
    consensus            earlier name of ``search``
    """
    import multiprocessing
    if int(workers) > 1 and multiprocessing.parent_process() is not None:
        raise RuntimeError(
            "concord.compare() was called inside a worker process. With workers > 1 the workers are spawned "
            "and re-import the main script, so put the call under `if __name__ == '__main__':` "
            "(the command-line entry point and importable modules are already safe).")
    wall_started = time.monotonic()
    estimator = _canonical_estimator(estimator)
    search = _canonical_search(search if consensus is None else consensus)
    provenance = {"package_version": PACKAGE_VERSION, "started_utc": _utc()}
    if estimator != "saebm":
        provenance["pyebm"] = verify_pyebm()
    frame, group_values, names = prepare_data(df, group_column, labels, biomarkers, group_order)
    spec = CompareSpec(group_column=group_column, group_values=tuple(group_values),
                       labels=tuple(labels), biomarkers=tuple(names), estimator=estimator,
                       search=search, fast_likelihood=bool(fast_likelihood),
                       fit_timeout_s=int(fit_timeout_s), seed=int(seed),
                       saebm_iterations=int(saebm_iterations), saebm_burn_in=int(saebm_burn_in),
                       common_proportions=common_proportions)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    B = int(B)
    if B < 0 or stability_resamples < 0:
        raise ValueError("B and stability_resamples must be nonnegative")
    definitions = scheme_definitions(spec, schemes)
    pairs = spec.pairs
    n_pairs = len(pairs)
    pair_names = [spec.pair_name(p) for p in pairs]
    pair_alpha = Fraction(str(alpha)) / n_pairs

    def log(message):
        if verbose:
            print(f"[concord {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)

    composition = _composition(frame, spec)
    _check_common_proportions(spec, composition)
    log(f"{len(frame)} participants, {spec.n_groups} groups {list(group_values)}, {len(names)} biomarkers; "
        f"estimator={estimator}; common proportions={_describe(spec.common_proportions, spec.labels)}; "
        f"schemes={list(definitions)}; B={B}; workers={workers}")
    for flag in composition["flags"]:
        log("composition note: " + flag)

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

        # 2. relabelings (+ bootstrap refits in the same pool)
        counts = {name: {"pairs": np.zeros(n_pairs, dtype=int), "done": 0, "failed": 0, "seconds": 0.0}
                  for name in definitions}
        remaining = {name: list(range(B)) for name in definitions}
        stability_records = []
        stability_todo = list(range(int(stability_resamples)))
        total = B * len(definitions) + len(stability_todo)
        finished = 0
        last_report = time.monotonic()
        batch_size = max(2 * runner.workers, 8)

        def decided(name):
            # failed fits stay unknown in the planned budget (decisions never use a smaller denominator)
            c, d = counts[name], definitions[name]
            indexes = [d["pair_index"]] if d["pair_index"] is not None else range(n_pairs)
            return all(exact_decision(int(c["pairs"][i]), c["done"], B, pair_alpha, rule) is not None
                       for i in indexes)

        while True:
            batch = []
            for name in definitions:
                if not remaining[name]:
                    continue
                if stop_when_decided and decided(name):
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
                        c["pairs"] += (np.asarray(record["distances"]) >= distances - 1e-9)
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
    def adjusted(p):
        return None if p is None else min(1.0, n_pairs * p)

    tests = {}
    for name, d in definitions.items():
        c = counts[name]
        completed = c["done"]
        unknown = B - completed
        cells = {}
        indexes = [d["pair_index"]] if d["pair_index"] is not None else range(n_pairs)
        for i in indexes:
            count = int(c["pairs"][i])
            p = (1 + count) / (completed + 1) if completed else None
            bounds = [(1 + count) / (B + 1), (1 + count + unknown) / (B + 1)]
            cells[pair_names[i]] = {
                "exceedances": count,
                "p": p,
                "p_bounds": bounds,
                "p_adjusted": adjusted(p),
                "p_adjusted_bounds": [adjusted(bounds[0]), adjusted(bounds[1])],
                "reject": exact_decision(count, completed, B, pair_alpha, rule),
                "threshold": float(pair_alpha)}
        tests[name] = {"scheme": d["scheme"], "definition": d, "budget": B, "completed": completed,
                       "failed": c["failed"], "pairs": cells, "seconds": c["seconds"]}
    reject = {}
    for test in tests.values():
        reject.setdefault(test["scheme"], {}).update(
            {pair: cell["reject"] for pair, cell in test["pairs"].items()})
    decisions = {
        "rule": rule, "alpha": alpha, "n_pairs": n_pairs, "per_pair_threshold": float(pair_alpha),
        "convention": "P = (1 + E) / (B + 1), E = number of relabelings with distance >= observed; "
                      "a comparison is rejected when P <= alpha / n_pairs (Bonferroni); adjusted P = "
                      "min(1, n_pairs * P). p and p_adjusted use the completed relabelings, "
                      "(1 + E) / (completed + 1); reject and p_bounds refer to the planned B, with "
                      "failed or unfinished relabelings counted as exceedances for the upper bound, "
                      "and reject is None when those relabelings could still change the decision",
        "reject": reject,
        "reject_any_pair": {scheme: any(v is True for v in cells.values()) for scheme, cells in reject.items()},
        "stop_when_decided": bool(stop_when_decided)}

    orderings = {str(g): [names[i] for i in observed["orderings"][k]] for k, g in enumerate(group_values)}
    positions = {name: {str(g): int(np.flatnonzero(np.asarray(observed["orderings"][k]) == j)[0])
                        for k, g in enumerate(group_values)} for j, name in enumerate(names)}
    distance_block = {"pairs": dict(zip(pair_names, map(float, distances))),
                      "statistic": "normalized Kendall distance"}
    ess = observed["diagnostics"].get("effective_sample_size")
    if ess is None:
        sizes = frame[group_column].value_counts()
        ess = {str(g): float(sizes.get(k, 0)) for k, g in enumerate(group_values)}
    else:
        ess = {str(g): float(ess[str(k)]) for k, g in enumerate(group_values)}
    used, weights, largest = _weight_summary(observed.get("full_diagnostics", {}), spec, composition)
    paired = {"pairs": {}, "note": "P(within_diagnosis) - P(unrestricted) for each comparison"}
    if "within_diagnosis" in tests and "unrestricted" in tests:
        for pn in pair_names:
            a, b = tests["within_diagnosis"]["pairs"][pn]["p"], tests["unrestricted"]["pairs"][pn]["p"]
            paired["pairs"][pn] = None if a is None or b is None else a - b
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
        decisions=decisions, composition=composition, common_proportions=used, weights=weights,
        largest_weight=largest, effective_sample_size=ess, stability=stability,
        paired_difference=paired, fits=fits, timing=timing, provenance=provenance,
        observed_diagnostics=observed.get("full_diagnostics", {}))
    log("done")
    return result


compare_orderings = compare   # earlier name


def _describe(proportions, labels):
    if isinstance(proportions, tuple):
        return "fixed " + "/".join(f"{label} {p:.3f}" for label, p in zip(labels, proportions))
    return {"min": "minimum rule", "pooled": "pooled sample", None: "none (unweighted)"}.get(proportions,
                                                                                            str(proportions))


def _check_common_proportions(spec, composition):
    """Fixed proportions: every diagnosis with a positive proportion must be present in every group.
    Minimum rule: at least one diagnosis must be present in every group."""
    counts = composition["counts"]
    if isinstance(spec.common_proportions, tuple):
        for g in spec.group_values:
            lacking = [str(label) for label, p in zip(spec.labels, spec.common_proportions)
                       if p > 0 and counts[str(g)][str(label)] == 0]
            if lacking:
                raise ValueError(f"diagnosis {', '.join(lacking)} has a positive common proportion but no "
                                 f"participants in group {g}")
    elif spec.common_proportions == "min":
        if not any(all(counts[str(g)][str(label)] > 0 for g in spec.group_values) for label in spec.labels):
            raise ValueError("no diagnosis is present in every group, so the minimum rule is undefined")


def _weight_summary(diagnostics, spec, composition):
    """Common proportions used in the observed fit, weights per group and diagnosis, largest weight."""
    code_names = _code_names(spec.labels)
    proportions = diagnostics.get("common_proportions")
    cells = diagnostics.get("group_weights")
    used = None if proportions is None else {name: float(proportions[k]) for k, name in code_names}
    index_of = dict(zip(map(str, spec.labels), [0] + [1] * (len(spec.labels) - 2) + [2]))
    weights = {}
    for code, g in enumerate(spec.group_values):
        if cells is not None and str(code) in cells:
            row = cells[str(code)]
            weights[str(g)] = {name: (None if row[k] is None else float(row[k])) for k, name in code_names}
        else:       # unweighted estimator: every participant has weight one
            present = {index_of[label] for label, n in composition["counts"][str(g)].items() if n > 0}
            weights[str(g)] = {name: (1.0 if k in present else None) for k, name in code_names}
    values = [w for row in weights.values() for w in row.values() if w is not None]
    return used, weights, (max(values) if values else None)


def _failed_result(spec, group_values, names, observed, composition, provenance, fits, wall_started):
    return ComparisonResult(spec=asdict(spec), groups=[str(g) for g in group_values],
                            biomarkers=list(names), status=observed["status"], orderings={},
                            positions={}, distances={}, tests={}, decisions={}, composition=composition,
                            common_proportions=None, weights={}, largest_weight=None,
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
            flags.append(f"group {g} has no participants with diagnosis {', '.join(empty)}")
        if sum(row.values()) < 30:
            flags.append(f"group {g} has only {sum(row.values())} participants")
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
            "pooled": pooled, "min_rule": dict(zip(map(str, spec.labels), map(float, minimum))),
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
    # distance between the refitted orderings of different groups, to set against the observed one
    out["between_group_distance_mean"] = {}
    for a, b in combinations(range(len(group_values)), 2):
        out["between_group_distance_mean"][f"{group_values[a]}-{group_values[b]}"] = float(np.mean(
            [kendall_distance(r["orderings"][a], r["orderings"][b]) for r in ok]))
    return out


def _utc():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- command line

def _parse_common_proportions(text):
    """'min' -> 'min'; '0.5,0.25,0.25' -> list; 'CN=0.5,MCI=0.25,AD=0.25' -> dict."""
    if text is None:
        return None
    text = text.strip()
    if text == "min":
        return "min"
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if parts and all("=" in part for part in parts):
        return {key.strip(): float(value) for key, value in (part.split("=", 1) for part in parts)}
    return [float(part) for part in parts]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="concord",
        description="CONCORD: compare the event orderings of two or more groups at common diagnostic "
                    "proportions and test the differences by permuting group labels within diagnosis.")
    ap.add_argument("input", help="CSV file with one row per participant: PTID (optional), Diagnosis, "
                                  "the group column and the biomarker columns")
    ap.add_argument("--group", default="APOE", help="group column (default: APOE)")
    ap.add_argument("--labels", default="CN,MCI,AD", help="diagnosis labels in disease order (default: CN,MCI,AD)")
    ap.add_argument("--biomarkers", default=None, help="comma-separated biomarker columns (default: every other column)")
    ap.add_argument("--estimator", default="concord",
                    help="concord (default), pooled_score (pooled-score DEBM), separate (separately fitted "
                         "DEBM) or saebm (SA-EBM, needs pysaebm)")
    ap.add_argument("--common-proportions", default=None, metavar="P",
                    help="CONCORD's common proportions: 'min' (default: the smallest proportion of each "
                         "diagnosis across the compared groups, rescaled to sum to one) or fixed proportions "
                         "in label order, e.g. 0.5,0.25,0.25 or CN=0.5,MCI=0.25,AD=0.25, held fixed in every "
                         "relabeling and bootstrap refit. A diagnosis left out of CN=...,AD=... gets proportion "
                         "0 and its participants weight 0. Compute fixed proportions with "
                         "concord.common_proportions(...), e.g. over the six cohort-by-genotype groups of two "
                         "cohorts as in the paper, which gave 0.48926,0.28593,0.22481 (rounded)")
    ap.add_argument("--search", default=None,
                    help="ranking-consensus search: continued (default; continue adjacent swaps until none "
                         "lowers the loss) or pyebm (unmodified pyebm 2.0.3 search)")
    ap.add_argument("--consensus", default=None, help=argparse.SUPPRESS)       # earlier name of --search
    ap.add_argument("--schemes", default=None,
                    help="comma-separated permutation schemes: unrestricted, within_diagnosis, "
                         "pairwise_within_diagnosis (default: within_diagnosis for two groups, "
                         "pairwise_within_diagnosis for three or more; unrestricted only when requested)")
    ap.add_argument("--B", type=int, default=599, help="relabelings per scheme (default: 599)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="family-wise level; each comparison is tested at alpha / number of comparisons "
                         "(default: 0.05)")
    ap.add_argument("--stop-when-decided", action="store_true",
                    help="stop a scheme once no further relabeling can change its decisions")
    ap.add_argument("--stability", type=int, default=20,
                    help="bootstrap refits within group-by-diagnosis cells (default: 20; 0 skips them)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the relabelings and bootstrap resamples")
    ap.add_argument("--workers", type=int, default=1, help="worker processes (default: 1)")
    ap.add_argument("--fit-timeout", type=int, default=900, help="time limit per fit in seconds (default: 900)")
    ap.add_argument("--no-fast-likelihood", action="store_true",
                    help="evaluate pyebm's unmodified mixture objective")
    ap.add_argument("--saebm-iterations", type=int, default=10000, help="SA-EBM iterations (default: 10000)")
    ap.add_argument("--saebm-burn-in", type=int, default=2500, help="SA-EBM burn-in iterations (default: 2500)")
    ap.add_argument("--output", type=Path, default=None, help="write the full result as JSON to this path")
    ap.add_argument("--quiet", action="store_true", help="no progress messages")
    args = ap.parse_args(argv)
    search = args.search or args.consensus or "continued"
    schemes = (None if args.schemes is None
               else tuple(s.strip() for s in args.schemes.split(",") if s.strip()))
    labels = tuple(args.labels.split(","))
    try:
        proportions = _parse_common_proportions(args.common_proportions)
    except ValueError:
        ap.error("--common-proportions expects 'min' or numbers such as 0.5,0.25,0.25")
    try:                                   # every option is checked before the input is read
        estimator = _canonical_estimator(args.estimator)
        _canonical_search(search)
        for scheme in schemes or ():
            _canonical_scheme(scheme)
        _resolve_common_proportions(estimator, proportions, labels)
    except ValueError as exc:
        ap.error(str(exc))
    df = pd.read_csv(args.input)
    result = compare(
        df, group_column=args.group, labels=labels,
        biomarkers=args.biomarkers.split(",") if args.biomarkers else None,
        estimator=args.estimator, search=search, schemes=schemes,
        B=args.B, alpha=args.alpha, stop_when_decided=args.stop_when_decided,
        stability_resamples=args.stability, seed=args.seed, workers=args.workers,
        fit_timeout_s=args.fit_timeout, fast_likelihood=not args.no_fast_likelihood,
        saebm_iterations=args.saebm_iterations, saebm_burn_in=args.saebm_burn_in,
        verbose=not args.quiet, common_proportions=proportions)
    if args.output:
        result.to_json(args.output)
    print(result.summary())
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
