"""Pooled abnormality model and diagnosis weights (CONCORD and pooled-score DEBM).

Both estimators fit one two-component Gaussian mixture per biomarker to the pooled sample of the
compared groups (pyebm 2.0.3 with no group argument), so that the same measurement receives the
same probability of being abnormal in every group. Each group's ordering is then estimated with
the DEBM ranking consensus, run through concord.engine.fit_orderings. The variants differ only in
the participant weights applied to the ranking loss:

  shared            pooled-score DEBM: every participant has weight one.
  invariant_min     CONCORD: a participant with diagnosis d in group g has weight
                    w = pi*(d) / pihat_g(d), where pihat_g(d) is the proportion of diagnosis d in
                    group g and pi*(d) are the common proportions. Each group's ordering minimizes
                    the weighted mean ranking loss, so after weighting every diagnosis accounts
                    for the same share pi*(d) of every group. By default pi*(d) is the smallest
                    proportion of diagnosis d across the compared groups, rescaled to sum to one
                    (the minimum rule); this gives the smallest possible largest weight, 1/kappa
                    with kappa = sum_d min_g pihat_g(d). Fixed common proportions can be passed
                    instead (``reference=``); every diagnosis with a positive proportion must then
                    be present in every group.
  invariant_pooled  weights to the diagnostic proportions of the pooled sample (not used in the
                    paper).

The pooled mixture depends on the measurements and the diagnoses (its two components are
initialized from the CN and AD participants), not on the group labels, so it is the same after
every relabeling of a dataset and is cached: a small process-local cache keyed by a hash of the
biomarker matrix, the biomarker names and the pyebm diagnosis codes holds the fitted parameters and
the participants' posterior probabilities. Everything that depends on the group labels (weights,
orderings) is recomputed at every call. With all weights equal to one, CONCORD reproduces
pooled-score DEBM exactly.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict, namedtuple
from time import monotonic

import numpy as np

from .engine import EngineConfig, FitResult, fit_orderings

INVARIANT_ENGINE_VERSION = "invariant-v1.1"
VARIANTS = {"shared": None, "invariant_pooled": "pooled", "invariant_min": "min"}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 6
CODES = np.array([1, 2, 3])          # pyebm diagnosis codes: CN, MCI, AD
_CODE_NAMES = ("CN", "MCI", "AD")


def _diagnosis_codes(df, labels):
    """pyebm diagnosis codes per row: 1 for the first label (CN), 3 for the last (AD), 2 otherwise."""
    labels = [str(label) for label in labels]
    diagnosis = df["Diagnosis"].astype(str).to_numpy()
    return np.where(diagnosis == labels[0], 1, np.where(diagnosis == labels[-1], 3, 2)).astype(np.int8)


def _matrix_key(df, names, labels=("CN", "MCI", "AD")):
    """Cache key of the pooled mixture: the biomarker matrix, the biomarker names and the diagnosis
    codes (the mixture is initialized from the CN and AD participants). The group column is left
    out, so the key is the same after every relabeling."""
    names = [str(name) for name in names]
    values = np.ascontiguousarray(df[names].to_numpy(dtype=float))
    h = hashlib.sha256(values.tobytes()); h.update("|".join(names).encode())
    h.update(b"|diagnosis|"); h.update(np.ascontiguousarray(_diagnosis_codes(df, labels)).tobytes())
    return h.hexdigest()


def _weights(diag, gv, mask, ref):
    """Common proportions, participant weights, effective sample sizes and weights per diagnosis.

    Returns (group values in np.unique order, {g: weight vector over g's masked rows}, ESS by group,
    common proportions over CODES or None, {g: [weight of each diagnosis code or None if absent]}).
    """
    gvals = np.unique(gv)
    comp = {}
    for g in gvals:
        rows = (gv == g) & mask
        comp[g] = np.array([(diag[rows] == c).mean() for c in CODES])
    if ref is None:
        pi_common = None
    elif isinstance(ref, str) and ref == "pooled":
        pi_common = np.array([(diag[mask] == c).mean() for c in CODES])
    elif isinstance(ref, str) and ref == "min":
        m = np.min(np.stack([comp[g] for g in gvals]), axis=0)
        if not m.sum() > 0:
            raise ValueError("no diagnosis is present in every group, so the minimum rule is undefined")
        pi_common = m / m.sum()
    elif isinstance(ref, (tuple, list, np.ndarray)):
        pi_common = np.asarray(ref, dtype=float)                 # fixed common proportions
        if (pi_common.shape != CODES.shape or not np.isfinite(pi_common).all() or (pi_common < 0).any()
                or pi_common.sum() <= 0):
            raise ValueError("fixed common proportions must be three nonnegative values (CN, MCI, AD)")
        pi_common = pi_common / pi_common.sum()
        for g in gvals:
            lacking = [_CODE_NAMES[k] for k in range(len(CODES)) if pi_common[k] > 0 and not comp[g][k] > 0]
            if lacking:
                raise ValueError(f"diagnosis {', '.join(lacking)} has a positive common proportion but no "
                                 f"participants in group {g}")
    else:
        raise ValueError(f"Unknown common proportions: {ref!r}")
    weights, ess, cells = {}, {}, {}
    for g in gvals:
        rows = (gv == g) & mask; d = diag[rows]
        if pi_common is None:
            w = np.ones(int(rows.sum()))
            cells[str(g)] = [1.0 if comp[g][k] > 0 else None for k in range(len(CODES))]
        else:
            w = np.zeros(int(rows.sum()))
            cell = [None] * len(CODES)
            for k, c in enumerate(CODES):
                sel = d == c
                if sel.any():
                    w[sel] = pi_common[k] / comp[g][k]
                    cell[k] = float(pi_common[k] / comp[g][k])
            cells[str(g)] = cell
        weights[g] = w
        ess[str(g)] = float(w.sum() ** 2 / (w ** 2).sum())
    return gvals, weights, ess, (None if pi_common is None else [float(x) for x in pi_common]), cells


def composition_weights(diag, gv, mask, ref):
    """diag: int codes per row; gv: group value per row; mask: rows used; ref: None (unit weights),
    'min', 'pooled' or three fixed common proportions (CN, MCI, AD). Returns
    (group values in np.unique order, {g: weight vector over g's masked rows}, ESS by group)."""
    gvals, weights, ess, _, _ = _weights(diag, gv, mask, ref)
    return gvals, weights, ess


def fit_invariant_orderings(df, engine_config: EngineConfig | None = None, variant: str = "invariant_min",
                            use_cache: bool = True, reference=None) -> FitResult:
    """Fit every group's ordering with the pooled abnormality model and the variant's weights.

    ``reference``: optional fixed common proportions (CN, MCI, AD) that replace the variant's rule
    (the pooled abnormality model is kept). Every diagnosis with a positive proportion must be
    present in every group; otherwise the fit returns status 'error'. A proportion of 0 gives the
    participants with that diagnosis weight 0 (they still enter the pooled mixture). The
    diagnostics record the common proportions, the weight of each diagnosis in each group and the
    effective sample sizes.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    ref = VARIANTS[variant]
    rule = ref
    if reference is not None:
        ref = tuple(float(x) for x in reference)
        if len(ref) != len(CODES) or not np.isfinite(ref).all() or any(x < 0 for x in ref) or sum(ref) <= 0:
            raise ValueError("reference must be three nonnegative proportions (CN, MCI, AD)")
        rule = "fixed"
    config = engine_config or EngineConfig(mode="repaired")
    import pyebm.core_utilities as cu
    from pyebm.central_ordering import generalized_mallows as gm

    metadata = {"PTID", "Diagnosis", "EXAMDATE", config.group_column}
    names = list(config.biomarker_names) if config.biomarker_names is not None else [
        c for c in df.columns if c not in metadata]
    key = _matrix_key(df, names, config.labels) if use_cache else None
    cache = _CACHE.get(key) if key else None
    if cache is None:
        cache = {}
    holder, queue, cur = {}, [], {}
    info = {"ess": None, "proportions": None, "cells": None, "mixture_cached": cache.get("params") is not None}

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
        gvals, W, ess, proportions, cells = _weights(holder["diag"], gv, mask, ref)
        queue[:] = [W[g] for g in gvals]
        info.update(ess=ess, proportions=proportions, cells=cells)
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
                              common_proportions_rule=rule, common_proportions=info["proportions"],
                              group_weights=info["cells"], effective_sample_size=info["ess"],
                              pooled_mixture_cached=info["mixture_cached"],
                              pooled_mixture_seconds=cache.get("seconds"))
    return result
