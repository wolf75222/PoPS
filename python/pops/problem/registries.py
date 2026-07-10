"""Typed Problem registries, split by declaration family.

This module is the stable import surface.  Implementation modules separate model-instance
qualification, case declarations, runtime consumers, and structural constraints so no registry
family can grow back into the former monolith.
"""

from pops.problem._block_registry import BlockRegistry
from pops.problem._declaration_registries import FieldRegistry, ParamRegistry, TimeRegistry
from pops.problem._registry_support import NO_KIND as _NO_KIND  # noqa: F401 -- stable private seam
from pops.problem._runtime_registries import ConstraintRegistry, RuntimePolicyRegistry

__all__ = [
    "BlockRegistry",
    "ConstraintRegistry",
    "FieldRegistry",
    "ParamRegistry",
    "RuntimePolicyRegistry",
    "TimeRegistry",
]
