"""BRUNCH: branching structure inference of hybrid multivariate Hawkes processes.

Implementation of the PAKDD-2020 paper:
    Li, Li, Bhowmick. "BRUNCH: Branching Structure Inference of Hybrid Multivariate
    Hawkes Processes with Application to Social Media."
"""

from .events import EventCollection
from .kernels import apply_link, exp_kernel, exp_kernel_integral, link_names
from .likelihood import log_likelihood
from .media import run_media
from .mle import mle_update
from .model import BRUNCH, BRUNCHConfig, BRUNCHResult
from .params import HawkesParams
from .state import BranchingState
from .synthetic import SyntheticData, f1_parent_recovery, generate

__all__ = [
    "BRUNCH",
    "BRUNCHConfig",
    "BRUNCHResult",
    "BranchingState",
    "EventCollection",
    "HawkesParams",
    "SyntheticData",
    "apply_link",
    "exp_kernel",
    "exp_kernel_integral",
    "f1_parent_recovery",
    "generate",
    "link_names",
    "log_likelihood",
    "mle_update",
    "run_media",
]
