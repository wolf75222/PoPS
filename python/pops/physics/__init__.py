"""Physical-model authoring through the single canonical :class:`Model` facade.

The implementation engines remain in their private modules for lowering existing operator
registries; they are not alternate authoring APIs. Generic composition happens through typed
operators and small protocols returned by ``Model``.
"""

from .board import Model

__all__ = ["Model"]
