"""Identify the unmodified upstream pyebm implementation the engines patch.

This module never installs packages or edits upstream source. Install the pinned
wheel (pyebm==2.0.3) before running CONCORD. Source fingerprints also detect a
locally edited installation that still reports version 2.0.3.
"""
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform


PYEBM_VERSION = "2.0.3"
PYEBM_WHEEL_URL = "https://files.pythonhosted.org/packages/13/53/c7c2987dd8ab14a9035e2d2f80a4f607c01f14607b44db393c6fb6503d51/pyebm-2.0.3-py3-none-any.whl"
PYEBM_WHEEL_SHA256 = "246e1bb93db4ad0ed0c701f3e241b9c7ceda112599acff1c86291abaf6cb8df4"
SOURCE_SHA256 = {
    "central_ordering/generalized_mallows.py": "9c05900c77e81d9499e6f02a6d62879bb1de1fea51095834897cac2042aba431",
    "core_utilities.py": "e74650bfabea67885151142e142082feeeb944af50380db86256168ee0927972",
    "mixture_model/gaussian_mixture_model.py": "591fddc3646822f8c045e71ff930c3f1f080cce63dc7121ae34dc9b6fad9b8b6",
}


@lru_cache(maxsize=1)
def verify_pyebm() -> dict:
    """Fail before a run if version or the three relevant source files differ."""
    import pyebm

    version = metadata.version("pyebm")
    if version != PYEBM_VERSION:
        raise RuntimeError(f"Expected pyebm {PYEBM_VERSION}; found {version}")
    root = Path(pyebm.__file__).resolve().parent
    observed = {name: sha256((root / name).read_bytes()).hexdigest()
                for name in SOURCE_SHA256}
    if observed != SOURCE_SHA256:
        mismatches = [name for name in SOURCE_SHA256
                      if observed[name] != SOURCE_SHA256[name]]
        raise RuntimeError(f"pyebm source differs from the pinned wheel: {mismatches}")
    versions = {"python": platform.python_version(), "pyebm": version}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels"):
        versions[package] = metadata.version(package)
    return {"versions": versions, "pyebm_path": str(root),
            "source_sha256": observed, "wheel_sha256": PYEBM_WHEEL_SHA256}
