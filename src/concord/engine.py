"""DEBM fits through pyebm 2.0.3, with one optional change to the ranking-consensus search.

Mode ``original`` (``search='pyebm'`` in concord.compare) calls the unmodified pyebm 2.0.3 search,
which stops after the first accepted adjacent swap. Mode ``repaired`` (``search='continued'``, the
default of concord.compare) keeps pyebm's initialization, ranking loss, adjacent-swap neighborhood
and strict-improvement rule, but continues after an accepted swap until no adjacent swap lowers
the loss (or the 10,000-evaluation budget is reached). Mixture fitting and its stopping rule are
unchanged in both modes: for separately fitted DEBM, pyebm stops the mixture fitting when the mean
change in mixing proportions falls below 0.01 in the first group.

The temporary patches are process-local, serialized by a lock and restored even on exceptions;
run one fit at a time in each worker process. The caller sets fit timeouts and decides how to
handle quality flags; a failed SLSQP success flag is recorded, not silently excluded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Literal
import warnings

import numpy as np

from ._pyebm import verify_pyebm


ENGINE_VERSION = "consensus-v2.1"
_PATCH_LOCK = RLock()


@dataclass(frozen=True)
class EngineConfig:
    mode: Literal["original", "repaired"] = "original"
    expected_events: int | None = None
    group_column: str = "APOE"
    group_values: tuple = (0, 1, 2)
    labels: tuple[str, ...] = ("CN", "MCI", "AD")
    biomarker_names: tuple[str, ...] | None = None
    audit_original_neighbors: bool = False

    def __post_init__(self):
        if self.mode not in ("original", "repaired"):
            raise ValueError(f"Unknown engine mode: {self.mode!r}")
        if self.expected_events is not None and self.expected_events < 2:
            raise ValueError("expected_events must be at least 2")


@dataclass
class FitResult:
    orderings: list[np.ndarray] | None
    status: str
    diagnostics: dict

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {"orderings": None if self.orderings is None
                else [order.tolist() for order in self.orderings],
                "status": self.status, "diagnostics": self.diagnostics}


def _swap(ordering, index):
    out = list(ordering)
    out[index], out[index + 1] = out[index + 1], out[index]
    return out


def _neighbors(cls, ordering, data, probabilities):
    return [cls.totalconsensus(_swap(ordering, i), data, probabilities)
            for i in range(len(ordering) - 1)]


def _repaired_consensus(cls, n, data, probabilities, initial,
                        flag_only_init=None, *, records=None):
    """Same local-search objective as upstream; stop on lack of improvement.

    Upstream's budget is 10,000 sequence evaluations. A complete final-neighbor
    scan may exceed that search budget by at most n-1 evaluations; it provides
    scores at the returned sequence for upstream event-center calculation and
    an explicit residual-improvement diagnostic. No global optimum is claimed.
    """
    current = list(initial)
    if len(current) != n or sorted(current) != list(range(n)):
        raise ValueError("Consensus initial state is not a permutation of range(n)")
    best, heterogeneity, distances = cls.totalconsensus(current, data, probabilities)
    initial_score = float(best)
    if not np.isfinite(best):
        raise FloatingPointError("Nonfinite initial consensus objective")
    evaluations = accepted = 0
    scores = []
    reason = "initialization_only"
    local_optimal = None
    if flag_only_init != 1:
        while True:
            neighbors = _neighbors(cls, current, data, probabilities)
            scores = [item[0] for item in neighbors]
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite neighboring consensus objective")
            index = scores.index(min(scores))  # retain upstream tie-breaking
            candidate_score = scores[index]
            local_optimal = bool(best <= candidate_score)
            if local_optimal:
                reason = "no_improving_neighbor"
                break
            if evaluations + n - 1 > 10000:
                reason = "evaluation_limit"
                break
            evaluations += n - 1
            current = _swap(current, index)
            best, heterogeneity, distances = neighbors[index]
            accepted += 1
    if records is not None:
        records.append({"mode": "repaired", "initial_score": initial_score,
                        "final_score": float(best), "accepted_swaps": accepted,
                        "search_evaluations": evaluations,
                        "final_neighbor_evaluations": n - 1 if scores else 0,
                        "termination": reason, "local_optimal": local_optimal,
                        "best_neighbor_score": float(min(scores)) if scores else None,
                        "residual_improvement": float(max(0.0, best - min(scores)))
                        if scores else None})
    return current, best, scores, heterogeneity, distances


class _OptimizationRecorder:
    """Proxy only this package's optimization reference, not scipy globally."""
    def __init__(self, original, records):
        self.original, self.records = original, records

    def __getattr__(self, name):
        return getattr(self.original, name)

    def minimize(self, *args, **kwargs):
        result = self.original.minimize(*args, **kwargs)
        value = getattr(result, "fun", np.nan)
        self.records.append({"method": str(kwargs.get("method", "unspecified")),
                             "success": bool(result.success),
                             "status": int(result.status),
                             "nit": int(getattr(result, "nit", -1)),
                             "message": str(result.message),
                             "finite_x": bool(np.isfinite(result.x).all()),
                             "finite_objective": bool(np.isfinite(value).all())})
        return result


@contextmanager
def _temporary_engine(config: EngineConfig, diagnostics: dict):
    """Internal context; exposed for restoration and equivalence tests."""
    from pyebm.central_ordering.generalized_mallows import weighted_mallows
    from pyebm.mixture_model import gaussian_mixture_model as gmm

    with _PATCH_LOCK:
        original_descriptor = weighted_mallows.__dict__["consensus"]
        original_consensus = original_descriptor.__get__(None, weighted_mallows)
        original_opt = gmm.opt
        records = diagnostics.setdefault("consensus", [])
        optimizer_records = diagnostics.setdefault("optimizer", [])

        def wrapped(cls, n, data, probabilities, initial, flag_only_init=None):
            if config.mode == "repaired":
                return _repaired_consensus(cls, n, data, probabilities, initial,
                                          flag_only_init, records=records)
            result = original_consensus(n, data, probabilities, initial, flag_only_init)
            current, best, scores, _, _ = result
            changed = list(current) != list(initial)
            neighbor_scores = None
            if flag_only_init != 1:
                if not changed:
                    neighbor_scores = scores
                elif config.audit_original_neighbors:
                    neighbor_scores = [item[0] for item in
                                       _neighbors(cls, current, data, probabilities)]
            minimum = float(min(neighbor_scores)) if neighbor_scores else None
            records.append({"mode": "original", "final_score": float(best),
                            "accepted_swaps": int(changed),
                            "termination": "upstream_return",
                            "local_optimal": bool(best <= minimum)
                            if minimum is not None else None,
                            "best_neighbor_score": minimum,
                            "residual_improvement": float(max(0.0, best - minimum))
                            if minimum is not None else None})
            return result

        weighted_mallows.consensus = classmethod(wrapped)
        gmm.opt = _OptimizationRecorder(original_opt, optimizer_records)
        try:
            yield
        finally:
            weighted_mallows.consensus = original_descriptor
            gmm.opt = original_opt


def validate_orderings(orderings, events: int, groups: int) -> list[np.ndarray]:
    if not isinstance(orderings, list) or len(orderings) != groups:
        raise ValueError(f"Expected {groups} group orderings")
    checked = []
    for group, ordering in enumerate(orderings):
        values = np.asarray(ordering)
        if values.shape != (events,) or not np.isfinite(values).all():
            raise ValueError(f"Group {group}: wrong shape or nonfinite ordering")
        integers = values.astype(int)
        if not np.array_equal(values, integers) or not np.array_equal(
                np.sort(integers), np.arange(events)):
            raise ValueError(f"Group {group}: ordering is not a permutation")
        checked.append(integers)
    return checked


def map_orderings(orderings, model_names, expected_names, groups):
    """Return indices in the caller's biomarker order, checking names exactly."""
    if (len(model_names) != len(set(model_names))
            or len(expected_names) != len(set(expected_names))
            or set(model_names) != set(expected_names)):
        raise ValueError("Model and requested biomarker names do not match")
    checked = validate_orderings(orderings, len(expected_names), groups)
    mapping = np.asarray([expected_names.index(name) for name in model_names])
    return [mapping[ordering] for ordering in checked]


def _all_finite(value) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_all_finite(part) for part in value)
    return bool(np.isfinite(np.asarray(value)).all())


def fit_orderings(df, config: EngineConfig | None = None) -> FitResult:
    """Return validated orders plus diagnostics; never silently discard a fit.

    Missing dependencies/version mismatches raise immediately, before fitting.
    Runtime fit errors become explicit status='error'; malformed orderings become
    status='invalid'. Finite orders remain available despite optimizer quality
    flags, so this wrapper does not introduce unreported success conditioning.
    """
    config = config or EngineConfig()
    provenance = verify_pyebm()
    from pyebm import debm

    diagnostics = {"engine_version": ENGINE_VERSION, "mode": config.mode,
                   "provenance": provenance, "quality_flags": []}
    observed_groups = tuple(sorted(df[config.group_column].unique()))
    if observed_groups != config.group_values:
        raise ValueError(f"Expected groups {config.group_values}; found {observed_groups}")
    metadata_columns = {"PTID", "Diagnosis", "EXAMDATE", config.group_column}
    expected_names = list(config.biomarker_names) if config.biomarker_names is not None else [
        name for name in df.columns if name not in metadata_columns]
    if config.expected_events is not None and len(expected_names) != config.expected_events:
        raise ValueError("Input biomarker count differs from expected_events")
    started = monotonic()
    model = None
    orderings = None
    status = "error"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with _temporary_engine(config, diagnostics):
                model, _, _ = debm.fit(df, Factors=[], Labels=list(config.labels),
                                       Groups=[config.group_column])
            diagnostics["biomarkers"] = expected_names
            diagnostics["pyebm_biomarkers"] = list(model.BiomarkerList)
            try:
                orderings = map_orderings(model.MeanCentralOrdering,
                                          list(model.BiomarkerList), expected_names,
                                          len(config.group_values))
                status = "ok"
            except (ValueError, TypeError, OverflowError) as exc:
                diagnostics["error"] = f"{type(exc).__name__}: {exc}"
                status = "invalid"
            parameters_finite = all(
                _all_finite(getattr(parameters, field))
                for parameters in model.BiomarkerParameters
                for field in ("Control", "Disease", "Mixing"))
            diagnostics["parameters_finite"] = parameters_finite
            diagnostics["event_centers_finite"] = _all_finite(model.EventCenters)
            if not parameters_finite:
                diagnostics["quality_flags"].append("nonfinite_parameters")
            if not diagnostics["event_centers_finite"]:
                diagnostics["quality_flags"].append("nonfinite_event_centers")
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            status = "error"
    optimizers = diagnostics.get("optimizer", [])
    diagnostics["optimizer_calls"] = len(optimizers)
    diagnostics["optimizer_failures"] = sum(not item["success"] for item in optimizers)
    if diagnostics["optimizer_failures"]:
        diagnostics["quality_flags"].append("optimizer_nonconvergence")
    if any(item.get("termination") == "evaluation_limit"
           for item in diagnostics.get("consensus", [])):
        diagnostics["quality_flags"].append("consensus_evaluation_limit")
    diagnostics["warnings_count"] = len(captured)
    diagnostics["warnings"] = sorted({f"{item.category.__name__}: {item.message}"
                                      for item in captured})[:30]
    diagnostics["seconds"] = monotonic() - started
    return FitResult(orderings, status, diagnostics)
