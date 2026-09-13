"""CONCORD — COmposition-Normalised Consensus for ORDering comparison.

Compare event-based model (DEBM) orderings between groups with an estimator whose target does
not depend on each group's diagnostic composition, and a permutation test whose reference
distribution preserves it.

    import concord
    result = concord.compare(df, group_column="APOE")
    print(result.summary())
"""
from ._version import __version__
from .core import (CompareSpec, ComparisonResult, ESTIMATORS, SCHEMES, compare, compare_orderings,
                      exact_decision, fit_once, kendall_distance, prepare_data)
from .engine import EngineConfig, FitResult, fit_orderings
from .invariant import VARIANTS as INVARIANT_VARIANTS, composition_weights, fit_invariant_orderings
from .likelihood import fast_likelihood_context
from ._pyebm import verify_pyebm

__all__ = ["__version__", "compare", "compare_orderings", "ComparisonResult", "CompareSpec", "ESTIMATORS",
           "SCHEMES", "exact_decision", "fit_once", "kendall_distance", "prepare_data", "EngineConfig",
           "FitResult", "fit_orderings", "INVARIANT_VARIANTS", "composition_weights",
           "fit_invariant_orderings", "fast_likelihood_context", "verify_pyebm"]
