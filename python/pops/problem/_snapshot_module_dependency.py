"""Bounded structural projections for module attributes used by callbacks."""
from __future__ import annotations

import hashlib
import json
import types
from typing import Any

from pops.problem._snapshot_module_fingerprint import module_implementation_fingerprint


def module_dependency_projection(
    module: types.ModuleType,
    *,
    attribute_paths: tuple[tuple[str, ...], ...],
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    canonical: Any,
) -> dict[str, Any]:
    """Project observed module attributes and authenticate every traversed module.

    The artifact projection (``artifact=True``) authenticates a module outside PoPS by the values
    the callback reads, not by the bytes of its source file: a comment or a line shift in a user
    module is provenance, not artifact identity (spec 0.3 CC2, criterion (i) of G07; N11 lot 2b).
    The full snapshot keeps the source digest. A PoPS module keeps it in both, as engine identity.
    """
    def fingerprint(name: str) -> dict[str, Any]:
        if artifact and name != "pops" and not name.startswith("pops."):
            return {"module": name, "authority": "observed-values"}
        return module_implementation_fingerprint(name, path=path)

    dependencies = []
    for parts in sorted(set(attribute_paths)):
        current: Any = module
        module_chain = []
        for part in parts:
            if isinstance(current, types.ModuleType):
                module_chain.append(fingerprint(current.__name__))
            if not hasattr(current, part):
                raise AttributeError("AuthoringSnapshot module dependency %s has no attribute %s"
                                     % (".".join(parts), part))
            current = getattr(current, part)
        dependencies.append({
            "path": list(parts),
            "modules": module_chain,
            "value": canonical(
                current, path="%s.%s" % (path, ".".join(parts)), active=active,
                handle_resolver=handle_resolver, artifact=artifact),
        })
    full = {
        "module": module.__name__,
        "implementation": fingerprint(module.__name__),
        "dependencies": dependencies,
    }
    encoded = json.dumps(full, sort_keys=False, separators=(",", ":"), allow_nan=False)
    return {
        "module": module.__name__,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def is_cross_module_framework_dependency(function: Any, value: Any) -> bool:
    """Whether a PoPS symbol can be authenticated at its defining module boundary."""
    if not isinstance(value, type) and not callable(value):
        return False
    owner = getattr(function, "__module__", None)
    dependency = getattr(value, "__module__", None)
    return isinstance(dependency, str) and dependency.startswith("pops.") \
        and dependency != owner


def framework_dependency_projection(value: Any, *, path: str) -> dict[str, Any]:
    """Bound a framework dependency by source/binary plus qualified symbol."""
    module_name = value.__module__
    return {
        "module": module_implementation_fingerprint(module_name, path=path),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "kind": "class" if isinstance(value, type) else "callable",
    }


__all__ = [
    "framework_dependency_projection",
    "is_cross_module_framework_dependency",
    "module_dependency_projection",
]
