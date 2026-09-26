"""CONCORD: composition-normalized consensus for ordering comparison.

Compares the event orderings of two or more patient groups whose diagnostic composition differs.
CONCORD fits one abnormality model to the pooled sample, weights every group to the same
diagnostic proportions, and tests group differences by permuting labels within diagnosis.

    import concord
    result = concord.compare(df, group_column="APOE")
    print(result.summary())
"""
from ._version import __version__
from .core import (CompareSpec, ComparisonResult, ESTIMATORS, SCHEMES, SEARCHES, common_proportions,
                   compare, compare_orderings, exact_decision, fit_once, kendall_distance, prepare_data)
from .engine import EngineConfig, FitResult, fit_orderings
from .invariant import VARIANTS as INVARIANT_VARIANTS, composition_weights, fit_invariant_orderings
from .likelihood import fast_likelihood_context
from ._pyebm import verify_pyebm

__all__ = ["__version__", "compare", "compare_orderings", "common_proportions", "ComparisonResult",
           "CompareSpec", "ESTIMATORS", "SCHEMES", "SEARCHES", "exact_decision", "fit_once",
           "kendall_distance", "prepare_data", "EngineConfig", "FitResult", "fit_orderings",
           "INVARIANT_VARIANTS", "composition_weights", "fit_invariant_orderings",
           "fast_likelihood_context", "verify_pyebm"]
