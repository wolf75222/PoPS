"""Typed Problem registries, split by declaration family.

This module is the stable import surface.  Implementation modules separate model-instance
qualification and case declarations so no registry family can grow back into a monolith.
"""

from pops.problem._block_registry import BlockRegistry
from pops.problem._declaration_registries import FieldRegistry, ParamRegistry, TimeRegistry
from pops.problem._initial_registry import InitialConditionRegistry

__all__ = [
    "BlockRegistry",
    "FieldRegistry",
    "InitialConditionRegistry",
    "ParamRegistry",
    "TimeRegistry",
]
