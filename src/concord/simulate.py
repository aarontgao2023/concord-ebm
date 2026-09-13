"""Versioned simulation design for ordering-null calibration studies.

This module does not import or mutate the legacy simulator.  In particular, the
``component_auc`` parameters compare latent pre/post-event Gaussian components;
they are not the realized CN-versus-AD AUCs, which are saved separately.

Public interface: ``resolve(name, **overrides)``, ``DesignConfig.to_dict()`` /
``from_dict()``, and ``simulate(config, seed) -> (dataframe, truth)``.  Truth is
JSON-serializable, includes one latent stage per dataframe row, and is intended to
be persisted by the runner.  No output files are written by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


DESIGN_VERSION = "2.0.0"
SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
GROUP_ORDER = ("e2", "e33", "e4")
GROUP_CODE = {name: index for index, name in enumerate(GROUP_ORDER)}
DX_ORDER = ("CN", "MCI", "AD")
REFERENCE_COUNTS = ((57, 6, 12), (244, 66, 101), (110, 156, 219))
BASE_STAGE_BETA = {"CN": (0.6, 9.0), "MCI": (2.0, 2.0), "AD": (9.0, 0.6)}
STRESS_STAGE_BETA = {"CN": (1.2, 6.0), "MCI": (2.6, 1.6), "AD": (12.0, 0.5)}
MISSING_MULTIPLIERS = {
    "e2": {"CN": 1.20, "MCI": 0.80, "AD": 0.70},
    "e4": {"CN": 0.70, "MCI": 1.15, "AD": 1.25},
}
CORR_BLOCKS = {"CSF": 0.55, "COG": 0.75, "IMG": 0.60}
STREAM_NAMES = ("base_order", "alternative", "group_assignment", "diagnosis",
                "latent_stage", "measurement_noise", "missingness")


@dataclass(frozen=True)
class BiomarkerSpec:
    name: str
    modality: str
    component_auc: float
    p_available: float


BIOMARKERS = (
    BiomarkerSpec("ABETA", "CSF", .88, .735),
    BiomarkerSpec("PTAU", "CSF", .85, .735),
    BiomarkerSpec("TAU", "CSF", .80, .725),
    BiomarkerSpec("NG", "CSF", .68, .275),
    BiomarkerSpec("NFL", "CSF", .72, .288),
    BiomarkerSpec("ADAS13", "COG", .92, .990),
    BiomarkerSpec("MMSE", "COG", .86, 1.),
    BiomarkerSpec("Hippocampus", "IMG", .82, .990),
    BiomarkerSpec("Entorhinal", "IMG", .80, .990),
    BiomarkerSpec("MidTemp", "IMG", .78, .990),
    BiomarkerSpec("Fusiform", "IMG", .76, .990),
    BiomarkerSpec("WholeBrain", "IMG", .74, .990),
    BiomarkerSpec("Ventricles", "IMG", .72, .990),
    BiomarkerSpec("Precuneus", "IMG", .70, .990),
)
_BIOMARKER_BY_NAME = {b.name: b for b in BIOMARKERS}


@dataclass(frozen=True)
class DesignConfig:
    name: str = "REF_H0"
    diagnosis_mode: str = "fixed"       # fixed margins or iid from pooled DX law
    composition_lambda: float = 1.0     # only used with fixed margins
    sample_scale: int = 1               # scales every reference row/column margin
    biomarker_names: tuple[str, ...] = tuple(b.name for b in BIOMARKERS)
    fixed_base_order: tuple[int, ...] | None = None  # position -> biomarker index
    alternative_group: str | None = None
    target_inversions: int = 0
    geometry: str = "exact_k"            # uniform fixed-K, single, disjoint, block
    moved_biomarker: str | None = None   # optional identity for single displacement
    stage_eta: float = 0.0              # linear interpolation of Beta shape parameters
    stage_group: str = "e4"
    component_auc_shift: float = 0.0
    group_pdf_shift: dict[str, dict[str, float]] = field(default_factory=dict)
    correlated: bool = False            # common covariance, not group heterogeneity
    missing: bool = True
    missing_stress_eta: float = 0.0
    missing_scope: str = "CSF"           # scope of stress multiplier; baseline is all

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native fields, including explicit versioning."""
        result = asdict(self)
        result["biomarker_names"] = list(self.biomarker_names)
        if self.fixed_base_order is not None:
            result["fixed_base_order"] = list(self.fixed_base_order)
        result["design_version"] = DESIGN_VERSION
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DesignConfig":
        values = dict(payload)
        version = values.pop("design_version", DESIGN_VERSION)
        if version != DESIGN_VERSION:
            raise ValueError(f"unsupported design version: {version!r}")
        if "biomarker_names" in values:
            values["biomarker_names"] = tuple(values["biomarker_names"])
        if values.get("fixed_base_order") is not None:
            values["fixed_base_order"] = tuple(values["fixed_base_order"])
        return cls(**values)


def _validate_order(order: Any) -> np.ndarray:
    raw = np.asarray(order)
    if raw.ndim != 1 or raw.size < 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("an ordering must be a one-dimensional integer permutation")
    if sorted(raw.tolist()) != list(range(raw.size)):
        raise ValueError("ordering must contain every index 0,...,I-1 exactly once")
    return raw.astype(int, copy=True)


def count_inversions(base: Any, other: Any) -> int:
    """Integer Kendall distance between position-to-biomarker permutations."""
    base, other = _validate_order(base), _validate_order(other)
    if base.size != other.size:
        raise ValueError("orderings have different lengths")
    ranks = np.empty(base.size, dtype=int)
    ranks[base] = np.arange(base.size)
    relative = ranks[other]
    return sum(int(np.count_nonzero(relative[i + 1:] < relative[i]))
               for i in range(relative.size))


@lru_cache(maxsize=32)
def _mahonian_table(n: int) -> tuple[tuple[int, ...], ...]:
    """M[m][k] counts m-permutations with exactly k inversions (Python ints)."""
    rows: list[tuple[int, ...]] = [(1,)]
    for m in range(1, n + 1):
        previous = rows[-1]
        row = []
        for k in range(m * (m - 1) // 2 + 1):
            row.append(sum(previous[k - a] for a in range(min(k, m - 1) + 1)
                           if k - a < len(previous)))
        rows.append(tuple(row))
    return tuple(rows)


def _randbelow(rng: np.random.Generator, high: int) -> int:
    """Uniform Python integer; unlike numpy.integers this permits large counts."""
    if high <= 0:
        raise ValueError("high must be positive")
    bits = high.bit_length()
    while True:
        value = int.from_bytes(rng.bytes((bits + 7) // 8), "little") & ((1 << bits) - 1)
        if value < high:
            return value


def ordering_at_inversions(base: Any, target_inversions: int,
                          rng: np.random.Generator, geometry: str = "exact_k",
                          moved_index: int | None = None) -> np.ndarray:
    """Construct an alternative with exactly K inversions or raise ValueError.

    ``exact_k`` samples uniformly from all I! permutations conditional on K using
    Mahonian counts. ``single_displacement`` chooses a feasible (source,dest)
    uniformly, optionally restricted to a specified biomarker identity.
    ``disjoint_adjacent`` samples a set of K nonoverlapping adjacent pairs.
    ``block_swap`` chooses feasible block lengths with product K and their start.
    These are different alternative families; none is silently substituted.
    """
    base = _validate_order(base)
    n = len(base)
    if isinstance(target_inversions, (bool, np.bool_)) or not isinstance(
            target_inversions, (int, np.integer)):
        raise ValueError("target_inversions must be an integer")
    k = int(target_inversions)
    if not 0 <= k <= n * (n - 1) // 2:
        raise ValueError("requested inversion count is outside the permutation space")
    if geometry not in {"exact_k", "single_displacement", "disjoint_adjacent", "block_swap"}:
        raise ValueError(f"unknown alternative geometry: {geometry}")
    if moved_index is not None and geometry != "single_displacement":
        raise ValueError("moved_index only applies to single_displacement")
    if moved_index is not None and moved_index not in base:
        raise ValueError("moved biomarker is absent from the ordering")
    if k == 0:
        return base.copy()

    if geometry == "exact_k":
        table = _mahonian_table(n)
        remaining = k
        insertion_inversions = {}
        for m in range(n, 0, -1):
            weights = [table[m - 1][remaining - a]
                       if 0 <= remaining - a < len(table[m - 1]) else 0
                       for a in range(m)]
            draw = _randbelow(rng, sum(weights))
            for a, weight in enumerate(weights):
                if draw < weight:
                    insertion_inversions[m] = a
                    remaining -= a
                    break
                draw -= weight
        relative: list[int] = []
        for m in range(1, n + 1):
            relative.insert(m - 1 - insertion_inversions[m], m - 1)
        out = base[relative]
    elif geometry == "single_displacement":
        candidates = [(i, j) for i in range(n) for j in range(n)
                      if abs(j - i) == k and (moved_index is None or base[i] == moved_index)]
        if not candidates:
            raise ValueError("no single displacement realizes K for this ordering/biomarker")
        source, destination = candidates[int(rng.integers(len(candidates)))]
        moved = base.tolist()
        value = moved.pop(source)
        moved.insert(destination, value)
        out = np.asarray(moved, dtype=int)
    elif geometry == "disjoint_adjacent":
        if k > n // 2:
            raise ValueError("K disjoint adjacent swaps require at least 2*K biomarkers")
        starts = np.sort(rng.choice(n - k, size=k, replace=False)) + np.arange(k)
        out = base.copy()
        for start in starts:
            out[start], out[start + 1] = out[start + 1], out[start]
    else:
        candidates = [(a, k // a, start) for a in range(1, n)
                      if k % a == 0 and a + k // a <= n
                      for start in range(n - a - k // a + 1)]
        if not candidates:
            raise ValueError("no adjacent block swap realizes this K")
        a, b, start = candidates[int(rng.integers(len(candidates)))]
        out = np.concatenate((base[:start], base[start + a:start + a + b],
                              base[start:start + a], base[start + a + b:]))
    realized = count_inversions(base, out)
    if realized != k:
        raise AssertionError(f"alternative construction requested {k}, produced {realized}")
    return out


def fixed_group_dx(sample_scale: int = 1, composition_lambda: float = 1.) -> dict[str, dict[str, int]]:
    """Mix fixed composition tables while preserving the correct two margins.

    Column totals are 411/228/332 (the included genotype groups), not 417/235/342.
    Balanced rounding chooses the greatest fractional remainder sum among feasible
    floor/ceiling tables, preserving all group sizes and pooled diagnosis counts.
    """
    if isinstance(sample_scale, bool) or not isinstance(sample_scale, (int, np.integer)) or sample_scale < 1:
        raise ValueError("sample_scale must be a positive integer")
    if not np.isfinite(composition_lambda) or not 0 <= composition_lambda <= 1:
        raise ValueError("composition_lambda must be between zero and one")
    reference = np.asarray(REFERENCE_COUNTS, dtype=int) * sample_scale
    rows, cols = reference.sum(axis=1), reference.sum(axis=0)
    target = (1 - composition_lambda) * np.outer(rows, cols) / reference.sum() + composition_lambda * reference
    floors = np.floor(target + 1e-12).astype(int)
    row_deficit, col_deficit = rows - floors.sum(axis=1), cols - floors.sum(axis=0)
    deficit = int(row_deficit.sum())
    best = None
    best_score = -float("inf")
    for chosen in combinations(range(9), deficit):
        increment = np.zeros((3, 3), dtype=int)
        increment.flat[list(chosen)] = 1
        if np.array_equal(increment.sum(axis=1), row_deficit) and np.array_equal(
                increment.sum(axis=0), col_deficit):
            score = float(np.sum(increment * (target - floors)))
            if score > best_score:
                best, best_score = floors + increment, score
    if best is None:
        raise AssertionError("could not balance integer diagnosis counts")
    return {g: {dx: int(best[i, j]) for j, dx in enumerate(DX_ORDER)}
            for i, g in enumerate(GROUP_ORDER)}


def resolve(name: str = "REF_H0", **overrides: Any) -> DesignConfig:
    """Resolve a small named design; factors stay explicit rather than a full grid.

    Names: REF_H0, IID_H0, BALANCED_H0, STAGE_H0, MISS_CSF_H0,
    MISS_ALL_H0, PDF_H0, and PWR_E2_K14/PWR_E4_K27/etc.  K6_SINGLE_E2,
    K6_DISJOINT_E4 (either group) are geometry controls.  Overrides are dataclass
    fields, e.g. ``sample_scale=4`` or ``stage_eta=.5``.
    """
    aliases = {"REF": "REF_H0", "IID": "IID_H0", "STRESS_STAGE": "STAGE_H0"}
    canonical = aliases.get(name.upper(), name.upper())
    named = {
        "REF_H0": {},
        "IID_H0": {"diagnosis_mode": "iid", "composition_lambda": 0.},
        "BALANCED_H0": {"composition_lambda": 0.},
        "STAGE_H0": {"stage_eta": 1.},
        "MISS_CSF_H0": {"missing_stress_eta": 1., "missing_scope": "CSF"},
        "MISS_ALL_H0": {"missing_stress_eta": 1., "missing_scope": "all"},
        "PDF_H0": {"group_pdf_shift": {"e4": {"ABETA": .45, "PTAU": .25}}},
    }
    if canonical in named:
        cfg = DesignConfig(name=canonical, **named[canonical])
    elif match := re.fullmatch(r"PWR_(E2|E33|E4)_K(\d+)", canonical):
        cfg = DesignConfig(name=canonical, alternative_group=match[1].lower(), target_inversions=int(match[2]))
    elif match := re.fullmatch(r"K6_(SINGLE|DISJOINT)_(E2|E33|E4)", canonical):
        cfg = DesignConfig(name=canonical, alternative_group=match[2].lower(), target_inversions=6,
                           geometry={"SINGLE": "single_displacement", "DISJOINT": "disjoint_adjacent"}[match[1]])
    else:
        raise KeyError(f"unknown v2 design: {name!r}")
    cfg = replace(cfg, **overrides)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: DesignConfig) -> None:
    fixed_group_dx(cfg.sample_scale, cfg.composition_lambda)
    if cfg.diagnosis_mode not in {"fixed", "iid"}:
        raise ValueError("diagnosis_mode must be fixed or iid")
    if cfg.diagnosis_mode == "iid" and cfg.composition_lambda != 0:
        raise ValueError("iid diagnosis uses the common pooled law; set composition_lambda=0")
    if len(cfg.biomarker_names) < 2 or len(set(cfg.biomarker_names)) != len(cfg.biomarker_names):
        raise ValueError("select at least two distinct biomarkers")
    if not set(cfg.biomarker_names) <= _BIOMARKER_BY_NAME.keys():
        raise ValueError("unknown biomarker name")
    if cfg.fixed_base_order is not None and len(_validate_order(cfg.fixed_base_order)) != len(cfg.biomarker_names):
        raise ValueError("fixed_base_order has the wrong length")
    if cfg.alternative_group not in (None, *GROUP_ORDER) or cfg.stage_group not in GROUP_ORDER:
        raise ValueError("unknown alternative or stage group")
    if (cfg.alternative_group is None) != (cfg.target_inversions == 0):
        raise ValueError("H0 requires no alternative group and K=0; H1 requires a group and K>0")
    if not np.isfinite(cfg.stage_eta) or not 0 <= cfg.stage_eta <= 1:
        raise ValueError("stage_eta must be between zero and one")
    if not np.isfinite(cfg.missing_stress_eta) or cfg.missing_stress_eta < 0:
        raise ValueError("missing_stress_eta must be finite and nonnegative")
    if cfg.missing_scope not in {"CSF", "all"}:
        raise ValueError("missing_scope must be CSF or all")
    if not cfg.missing and cfg.missing_stress_eta:
        raise ValueError("missing stress cannot be active when missing=False")
    for bm in cfg.biomarker_names:
        auc = _BIOMARKER_BY_NAME[bm].component_auc + cfg.component_auc_shift
        if not np.isfinite(auc) or not .5 < auc < 1:
            raise ValueError("all component AUCs must be strictly between .5 and 1")
    for group, values in cfg.group_pdf_shift.items():
        if group not in GROUP_ORDER or not set(values) <= set(cfg.biomarker_names):
            raise ValueError("group_pdf_shift references an absent group or biomarker")
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("PDF shifts must be finite")
    if cfg.moved_biomarker is not None and cfg.moved_biomarker not in cfg.biomarker_names:
        raise ValueError("moved_biomarker is absent")
    # Validate feasibility here even when generation is never reached.
    base = cfg.fixed_base_order if cfg.fixed_base_order is not None else np.arange(len(cfg.biomarker_names))
    ordering_at_inversions(base, cfg.target_inversions, np.random.default_rng(0), cfg.geometry)
    if cfg.moved_biomarker is not None and cfg.geometry != "single_displacement":
        raise ValueError("moved_biomarker requires single_displacement")


def _auc(cn: np.ndarray, ad: np.ndarray) -> float | None:
    cn, ad = cn[np.isfinite(cn)], ad[np.isfinite(ad)]
    if not len(cn) or not len(ad):
        return None
    values = np.concatenate((cn, ad))
    order = np.argsort(values, kind="mergesort")
    _, starts, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return float((ranks[len(cn):].sum() - len(ad) * (len(ad) + 1) / 2) / (len(cn) * len(ad)))


def _stage_parameters(cfg: DesignConfig) -> dict[str, dict[str, list[float]]]:
    return {g: {dx: [float(base + (cfg.stage_eta if g == cfg.stage_group else 0.) * (stress - base))
                     for base, stress in zip(BASE_STAGE_BETA[dx], STRESS_STAGE_BETA[dx])]
                for dx in DX_ORDER} for g in GROUP_ORDER}


def simulate(cfg: DesignConfig | dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate biomarkers and latent truth without fitting an EBM.

    In IID_H0, DX and group labels are independently randomized and all remaining
    nuisance laws are common.  This is an exchangeable design for the fitted
    biomarker/DX input (subject IDs are identifiers, not predictor variables).
    Activating group-specific nuisance or an alternative explicitly changes that
    null; the metadata reports the resulting guarantees rather than trusting names.
    """
    if isinstance(cfg, dict):
        cfg = DesignConfig.from_dict(cfg)
    _validate_config(cfg)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    rngs = {name: np.random.Generator(np.random.PCG64(child)) for name, child in zip(STREAM_NAMES, children)}
    bms = [_BIOMARKER_BY_NAME[name] for name in cfg.biomarker_names]
    p = len(bms)
    base = (_validate_order(cfg.fixed_base_order) if cfg.fixed_base_order is not None
            else rngs["base_order"].permutation(p))
    orders = {g: base.copy() for g in GROUP_ORDER}
    if cfg.alternative_group is not None:
        moved_index = cfg.biomarker_names.index(cfg.moved_biomarker) if cfg.moved_biomarker else None
        orders[cfg.alternative_group] = ordering_at_inversions(base, cfg.target_inversions,
                                                             rngs["alternative"], cfg.geometry, moved_index)
    planned = {g: cfg.target_inversions if g == cfg.alternative_group else 0 for g in GROUP_ORDER}
    realized = {g: count_inversions(base, orders[g]) for g in GROUP_ORDER}
    if planned != realized:
        raise AssertionError("realized group orderings disagree with the planned alternative")

    counts = fixed_group_dx(cfg.sample_scale, cfg.composition_lambda)
    group_sizes = [sum(counts[g].values()) for g in GROUP_ORDER]
    n = sum(group_sizes)
    dx = np.empty(n, dtype="<U3")
    if cfg.diagnosis_mode == "iid":
        groups = rngs["group_assignment"].permutation(np.repeat(np.arange(3), group_sizes))
        pooled = np.asarray(REFERENCE_COUNTS).sum(axis=0)
        dx[:] = rngs["diagnosis"].choice(DX_ORDER, size=n, p=pooled / pooled.sum())
    else:
        groups = np.repeat(np.arange(3), group_sizes)
        offset = 0
        for group in GROUP_ORDER:
            labels = np.repeat(DX_ORDER, [counts[group][d] for d in DX_ORDER])
            # Uniform assignments conditional on the fixed group x DX table.
            dx[offset:offset + len(labels)] = rngs["diagnosis"].permutation(labels)
            offset += len(labels)

    stage_parameters = _stage_parameters(cfg)
    stage = np.empty(n, dtype=int)
    stage_fraction = np.empty(n, dtype=float)
    for gi, group in enumerate(GROUP_ORDER):
        for diagnosis in DX_ORDER:
            mask = (groups == gi) & (dx == diagnosis)
            a, b = stage_parameters[group][diagnosis]
            values = rngs["latent_stage"].beta(a, b, int(mask.sum()))
            stage_fraction[mask] = values
            stage[mask] = np.rint(p * values).astype(int)

    normal = NormalDist()
    aucs = np.asarray([bm.component_auc + cfg.component_auc_shift for bm in bms])
    delta = np.asarray([round(np.sqrt(2) * normal.inv_cdf(float(auc)), 12) for auc in aucs])
    covariance = np.eye(p)
    if cfg.correlated:
        for i, left in enumerate(bms):
            for j, right in enumerate(bms):
                if i != j and left.modality == right.modality:
                    covariance[i, j] = CORR_BLOCKS[left.modality]
    complete = rngs["measurement_noise"].standard_normal((n, p)) @ np.linalg.cholesky(covariance).T
    component_parameters = {}
    for gi, group in enumerate(GROUP_ORDER):
        mask = groups == gi
        group_delta = delta + np.asarray([cfg.group_pdf_shift.get(group, {}).get(bm.name, 0.) for bm in bms])
        if np.any(group_delta <= 0):
            raise ValueError("PDF shifts must preserve abnormal mean > normal mean")
        ranks = np.argsort(orders[group])
        abnormal = ranks[None, :] < stage[mask, None]
        complete[mask] += abnormal * group_delta
        component_parameters[group] = {
            bm.name: {"normal_mean": 0., "abnormal_mean": float(group_delta[i]), "sd": 1.,
                      "component_auc": float(normal.cdf(group_delta[i] / np.sqrt(2)))}
            for i, bm in enumerate(bms)}

    probabilities = np.tile([bm.p_available if cfg.missing else 1. for bm in bms], (n, 1))
    if cfg.missing_stress_eta:
        for gi, group in enumerate(GROUP_ORDER):
            for diagnosis in DX_ORDER:
                rows = (groups == gi) & (dx == diagnosis)
                multiplier = 1 + cfg.missing_stress_eta * (MISSING_MULTIPLIERS.get(group, {}).get(diagnosis, 1.) - 1)
                for i, bm in enumerate(bms):
                    if cfg.missing_scope == "all" or bm.modality == "CSF":
                        probabilities[rows, i] = np.clip(bm.p_available * multiplier, 0., 1.)
    observed = complete.copy()
    observed[rngs["missingness"].random((n, p)) >= probabilities] = np.nan
    df = pd.DataFrame(observed, columns=cfg.biomarker_names)
    df.insert(0, "Diagnosis", dx)
    df.insert(0, "PTID", [f"v2_{i:07d}" for i in range(n)])
    df["APOE"] = groups

    realized_counts = {group: {diagnosis: int(np.sum((groups == gi) & (dx == diagnosis)))
                               for diagnosis in DX_ORDER} for gi, group in enumerate(GROUP_ORDER)}
    diagnosis_auc = {}
    availability = {}
    for gi, group in enumerate(GROUP_ORDER):
        in_group = groups == gi
        cn, ad = in_group & (dx == "CN"), in_group & (dx == "AD")
        diagnosis_auc[group] = {
            bm.name: {"complete": _auc(complete[cn, i], complete[ad, i]),
                      "observed": _auc(observed[cn, i], observed[ad, i])}
            for i, bm in enumerate(bms)}
        availability[group] = {}
        for diagnosis in DX_ORDER:
            rows = in_group & (dx == diagnosis)
            availability[group][diagnosis] = {
                bm.name: {"expected": float(probabilities[rows, i][0]) if np.any(rows) else None,
                          "observed_n": int(np.count_nonzero(np.isfinite(observed[rows, i])))}
                for i, bm in enumerate(bms)}
    pair_truth = {}
    for a, b in combinations(GROUP_ORDER, 2):
        inversions = count_inversions(orders[a], orders[b])
        pair_truth[f"{a}_vs_{b}"] = {"null": inversions == 0, "inversions": inversions,
                                    "normalized_distance": inversions / (p * (p - 1) // 2)}
    group_pdf_common = all(component_parameters[g] == component_parameters[GROUP_ORDER[0]] for g in GROUP_ORDER)
    conditional_exchangeability = (not cfg.target_inversions and not cfg.stage_eta
                                  and group_pdf_common and not cfg.missing_stress_eta)
    payload = cfg.to_dict()
    config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    truth = {
        "design_version": DESIGN_VERSION,
        "config": payload,
        "seed": seed,
        "manifest": {
            "seed": seed, "bit_generator": "PCG64", "numpy_version": np.__version__,
            "pandas_version": pd.__version__, "config_sha256": sha256(config_json.encode()).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "streams": {name: {"entropy": int(child.entropy), "spawn_key": list(child.spawn_key),
                                "initial_state_words": child.generate_state(4).tolist()}
                        for name, child in zip(STREAM_NAMES, children)},
            "data_sha256": sha256(df.to_csv(index=False, float_format="%.17g").encode()).hexdigest(),
        },
        "biomarker_names": list(cfg.biomarker_names), "group_order": list(GROUP_ORDER),
        "base_order": base.tolist(), "group_orderings": {g: orders[g].tolist() for g in GROUP_ORDER},
        "planned_inversions": planned, "realized_inversions": realized,
        "pair_truth": pair_truth, "common_order_null": cfg.target_inversions == 0,
        "conditional_exchangeability_by_design": bool(conditional_exchangeability),
        "unrestricted_exchangeability_by_design": bool(conditional_exchangeability and cfg.diagnosis_mode == "iid"),
        "group_dx_counts": realized_counts, "group_sizes": dict(zip(GROUP_ORDER, group_sizes)),
        "stage_beta_parameters": stage_parameters,
        "latent_stage": stage.tolist(), "latent_stage_fraction": stage_fraction.tolist(),
        "latent_row_ptid": df.PTID.tolist(),
        "component_parameters": component_parameters, "shared_covariance": covariance.tolist(),
        "realized_diagnosis_auc": diagnosis_auc, "availability_by_group_dx": availability,
        "missingness_scope_description": "baseline availability applies to every modality; missing_scope restricts only the group-by-DX stress multiplier",
        "alternative_sampling": {"exact_k": "uniform over permutations conditional on exact K",
                                 "single_displacement": "uniform over feasible source/destination pairs, optionally restricted to one biomarker",
                                 "disjoint_adjacent": "uniform over sets of K nonoverlapping adjacent pairs",
                                 "block_swap": "uniform over feasible (left length, right length, start) triples"}[cfg.geometry],
    }
    # Fail at generation rather than leaving latent truth that cannot be persisted.
    json.dumps(truth, allow_nan=False)
    return df, truth
