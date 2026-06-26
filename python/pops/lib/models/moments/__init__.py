"""pops.lib.models.moments -- provided moment models (HyQMOM15, Gaussian).

Provided models are PURE compositions of the Spec-4 moment facade
(:mod:`pops.lib.moments`); they wrap the generic builder, never re-implement it.
"""
from .hyqmom15 import HyQMOM15
from .gaussian import Gaussian

__all__ = ["HyQMOM15", "Gaussian"]
