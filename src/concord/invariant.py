"""Composition-invariant DEBM estimators for between-group ordering comparison.

Three variants, all implemented as process-local patches on pyebm 2.0.3 and executed
through concord.engine.fit_orderings (repaired consensus), so validation, provenance and
diagnostics are the frozen pipeline's own:

  shared            one two-component mixture per biomarker fitted on the POOLED sample
                    (Groups=[]), replicated to every group; per-group consensus unchanged.
  invariant_pooled  shared mixture + consensus standardised to the pooled diagnostic
                    composition (weights pi_ref[d]/pi_g[d] as a weighted mean over subjects).
  invariant_min     shared mixture + consensus standardised to the renormalised element-wise
                    minimum composition over groups (lowest weight variance; the default).

The pooled mixture depends on the biomarker values only, never on the group labels, so it is
identical under every relabelling of the same dataset and may be cached: a small
process-local LRU keyed by a hash of the biomarker matrix holds the fitted parameters and
subject posteriors. Every label-dependent step (weights, consensus) is recomputed for every
call. Uniform weights reproduce the standard pooled-mixture fit exactly.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict, namedtuple
from time import monotonic

import numpy as np

from .engine import EngineConfig, FitResult, fit_orderings

INVARIANT_ENGINE_VERSION = "invariant-v1.0"
VARIANTS = {"shared": None, "invariant_pooled": "pooled", "invariant_min": "min"}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 6
CODES = np.array([1, 2, 3])          # pyebm diagnosis codes: CN, MCI, AD


def _matrix_key(df, names):
    values = np.ascontiguousarray(df[names].to_numpy(dtype=float))
    h = hashlib.sha256(values.tobytes()); h.update("|".join(names).encode())
    return h.hexdigest()


def composition_weights(diag, gv, mask, ref):
    """diag: int codes per row; gv: group value per row; mask: rows used. Returns
    (group values in np.unique order, {g: weight vector over g's masked rows}, ESS by group)."""
    gvals = np.unique(gv)
    comp = {}
    for g in gvals:
        rows = (gv == g) & mask
        comp[g] = np.array([(diag[rows] == c).mean() for c in CODES])
    if ref is None:
        pi_ref = None
    elif ref == "pooled":
        pi_ref = np.array([(diag[mask] == c).mean() for c in CODES])
    elif ref == "min":
        m = np.min(np.stack([comp[g] for g in gvals]), axis=0); pi_ref = m / m.sum()
    else:
        raise ValueError(f"Unknown reference composition: {ref!r}")
    weights, ess = {}, {}
    for g in gvals:
        rows = (gv == g) & mask; d = diag[rows]
        if pi_ref is None:
            w = np.ones(int(rows.sum()))
        else:
            w = np.zeros(int(rows.sum()))
            for k, c in enumerate(CODES):
                sel = d == c
                if sel.any():
                    w[sel] = pi_ref[k] / comp[g][k]
        weights[g] = w
        ess[str(g)] = float(w.sum() ** 2 / (w ** 2).sum())
    return gvals, weights, ess


def fit_invariant_orderings(df, engine_config: EngineConfig | None = None, variant: str = "invariant_min",
                            use_cache: bool = True) -> FitResult:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    ref = VARIANTS[variant]
    config = engine_config or EngineConfig(mode="repaired")
    import pyebm.core_utilities as cu
    from pyebm.central_ordering import generalized_mallows as gm

    metadata = {"PTID", "Diagnosis", "EXAMDATE", config.group_column}
    names = list(config.biomarker_names) if config.biomarker_names is not None else [
        c for c in df.columns if c not in metadata]
    key = _matrix_key(df, names) if use_cache else None
    cache = _CACHE.get(key) if key else None
    if cache is None:
        cache = {}
    holder, queue, cur, info = {}, [], {}, {"ess": None, "mixture_cached": cache.get("params") is not None}

    orig_mm, orig_parse, orig_fco = cu.do_mixturemodel, cu.parse_inputs, cu.find_central_ordering
    fm_desc, fm_orig = gm.weighted_mallows.__dict__["fitMallows"], gm.weighted_mallows.fitMallows
    tc_desc, tc_orig = gm.weighted_mallows.__dict__["totalconsensus"], gm.weighted_mallows.totalconsensus

    def shared_mm(DMO, data_AD_raw, data_CN_raw, Data_all, Groups, GroupValues, GroupValues_cn, GroupValues_ad,
                  HyperParams=1, flag_init_together=1, only_init=0):
        if "params" not in cache:
            t0 = monotonic()
            BP, p_yes, p_no, lpost, lpre = orig_mm(DMO, data_AD_raw, data_CN_raw, Data_all, [], [], [], [],
                                                   HyperParams=HyperParams, flag_init_together=1, only_init=only_init)
            cache["params"] = (np.array(BP.Control, copy=True), np.array(BP.Disease, copy=True), np.array(BP.Mixing, copy=True))
            cache["post"] = tuple(np.array(x, copy=True) for x in (p_yes, p_no, lpost, lpre))
            cache["n"] = Data_all.shape[0]; cache["seconds"] = monotonic() - t0
        if cache["n"] != Data_all.shape[0]:
            raise RuntimeError("cached pooled mixture does not match this dataset")
        C, D, M = cache["params"]; p_yes, p_no, lpost, lpre = cache["post"]
        k = len(np.unique(GroupValues[0])) if len(Groups) else 1
        BPn = namedtuple("BiomarkerParams", "Control Disease Mixing")
        BPn.Control = [C.copy() for _ in range(k)]; BPn.Disease = [D.copy() for _ in range(k)]
        BPn.Mixing = [M.copy() for _ in range(k)]
        return BPn, p_yes.copy(), p_no.copy(), lpost.copy(), lpre.copy()

    def parse_cap(*a, **k):
        out = orig_parse(*a, **k)
        holder["diag"] = out[7]["Diagnosis"].to_numpy().astype(int); holder["Data_all"] = out[6]
        return out

    def fco(Data_all, p_yes, BP, Groups, GroupValues, DMO, algo_type, maskidx):
        A = holder["Data_all"]
        if A.shape != Data_all.shape or not np.allclose(np.nan_to_num(A[:, :, 0], nan=-9e9),
                                                        np.nan_to_num(Data_all[:, :, 0], nan=-9e9)):
            raise RuntimeError("row order of Data_all differs from parse_inputs output")
        gv = np.asarray(GroupValues[0])
        mask = np.asarray(maskidx, bool) if len(maskidx) else np.ones(len(gv), bool)
        gvals, W, ess = composition_weights(holder["diag"], gv, mask, ref)
        queue[:] = [W[g] for g in gvals]; info["ess"] = ess
        return orig_fco(Data_all, p_yes, BP, Groups, GroupValues, DMO, algo_type, maskidx)

    def fm(p_yes, mixing):
        w = queue.pop(0)
        if len(w) != p_yes.shape[0]:
            raise RuntimeError(f"weight length {len(w)} != group rows {p_yes.shape[0]}")
        cur["w"] = w
        try:
            return fm_orig(p_yes, mixing)
        finally:
            cur.pop("w", None)

    def tc(pi0, D, prob):
        tscore, score, score_indv = tc_orig(pi0, D, prob)
        w = cur.get("w")
        if w is None:
            raise RuntimeError("totalconsensus called outside a weighted fitMallows")
        if len(w) != len(score):
            raise RuntimeError("weight/score length mismatch")
        return np.float64(np.sum(w * np.asarray(score, float)) / np.sum(w)), score, score_indv

    cu.do_mixturemodel = shared_mm
    if ref is not None:
        cu.parse_inputs = parse_cap; cu.find_central_ordering = fco
        gm.weighted_mallows.fitMallows = staticmethod(fm); gm.weighted_mallows.totalconsensus = staticmethod(tc)
    try:
        result = fit_orderings(df, config)
    finally:
        cu.do_mixturemodel, cu.parse_inputs, cu.find_central_ordering = orig_mm, orig_parse, orig_fco
        gm.weighted_mallows.fitMallows = fm_desc; gm.weighted_mallows.totalconsensus = tc_desc
    if ref is not None and queue:
        result = FitResult(None, "error", {**result.diagnostics, "error": "unconsumed group weights"})
    if key and "params" in cache:
        _CACHE[key] = cache; _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    result.diagnostics.update(invariant_engine_version=INVARIANT_ENGINE_VERSION, variant=variant,
                              reference_composition=ref, effective_sample_size=info["ess"],
                              pooled_mixture_cached=info["mixture_cached"],
                              pooled_mixture_seconds=cache.get("seconds"))
    return result
