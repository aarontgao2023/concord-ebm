"""Faster evaluation of pyebm's Gaussian-mixture objective (used by default).

At every evaluation of the mixture log-likelihood, pyebm 2.0.3 constructs two frozen
scipy.stats.norm distributions. calculate_likelihood_gmm below computes the same objective by
calling scipy.stats.norm.pdf with the same location and scale. Removal of missing values, the
order of the arithmetic, the 1e-100 term, the sum over groups, the parameterization and the
function name are unchanged; gradients, tolerances, bounds, the optimizer and every other pyebm
function are untouched. concord.compare(fast_likelihood=False) uses the unmodified objective.

The replacement is process-local and the original objective is always restored.

Objective adapted from pyebm 2.0.3 gaussian_mixture_model.py, copyright Erasmus MC Rotterdam and
contributors, licensed under the GNU General Public License version 3; the adapted objective is
licensed under the same terms.
SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import scipy.stats

from ._pyebm import verify_pyebm
from .engine import _PATCH_LOCK


FAST_LIKELIHOOD_VERSION = "direct-normal-pdf-v2.1"


def calculate_likelihood_gmm(param, data, Groups, GroupValues, Mixing):
    """The upstream scalar objective with only frozen-RV construction removed."""
    if len(Mixing) == 0:
        param_mix = param[4]
        invalid_indices = np.isnan(data)
        valid_indices = np.logical_not(invalid_indices)
        likeli_pre = scipy.stats.norm.pdf(data[valid_indices], loc=param[0], scale=param[1])
        likeli_post = scipy.stats.norm.pdf(data[valid_indices], loc=param[2], scale=param[3])
        likeli = np.multiply(param_mix, likeli_pre) + np.multiply(1-param_mix, likeli_post) + 1e-100
        loglikeli = -np.sum(np.log(likeli))
    else:
        gval = np.unique(GroupValues[0])
        idx_valid = ~np.isnan(gval)
        gval = gval[idx_valid]
        loglikeli_list = []
        for g in range(len(gval)):
            param_mix = Mixing[g]
            idx = GroupValues[0] == gval[g]
            data_g = data[idx]
            invalid_indices = np.isnan(data_g)
            valid_indices = np.logical_not(invalid_indices)
            data_g = data_g[valid_indices]
            likeli_pre = scipy.stats.norm.pdf(data_g, loc=param[0], scale=param[1])
            likeli_post = scipy.stats.norm.pdf(data_g, loc=param[2], scale=param[3])
            likeli = np.multiply(param_mix, likeli_pre) + np.multiply(1-param_mix, likeli_post) + 1e-100
            loglikeli = -np.sum(np.log(likeli))
            loglikeli_list.append(loglikeli)
        loglikeli = np.sum(loglikeli_list)
    return loglikeli


@contextmanager
def fast_likelihood_context(enabled: bool = True):
    """Temporarily replace only the verified upstream GMM objective.

    Uses the reentrant engine lock, so nesting with any DEBM fit of this package
    is safe. The objective's __name__ is preserved for diagnostics.
    """
    if not enabled:
        yield {"enabled": False, "version": FAST_LIKELIHOOD_VERSION}
        return
    provenance = verify_pyebm()
    from pyebm.mixture_model import gaussian_mixture_model as gmm
    with _PATCH_LOCK:
        previous = gmm.calculate_likelihood_gmm
        try:
            gmm.calculate_likelihood_gmm = calculate_likelihood_gmm
            yield {"enabled": True, "version": FAST_LIKELIHOOD_VERSION,
                   "upstream_source_sha256": provenance["source_sha256"]["mixture_model/gaussian_mixture_model.py"],
                   "change": "normal PDF calls avoid frozen distribution construction only"}
        finally:
            gmm.calculate_likelihood_gmm = previous
