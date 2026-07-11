"""Authenticated implementation identity for Python and native modules."""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any


def module_implementation_fingerprint(
    module_name: str | None,
    *,
    path: str,
) -> dict[str, Any]:
    """Authenticate a module by immutable source, binary, or interpreter build identity."""
    if not module_name:
        raise TypeError("AuthoringSnapshot callable at %s has no implementation module" % path)
    module = sys.modules.get(module_name)
    if module is None:
        raise TypeError("AuthoringSnapshot cannot inspect unloaded module %s at %s"
                        % (module_name, path))
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if origin in {"built-in", "frozen"}:
        return {
            "module": module_name,
            "origin": origin,
            "runtime": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        }
    if not isinstance(origin, str) or not os.path.isfile(origin):
        raise TypeError(
            "AuthoringSnapshot cannot authenticate module %s at %s: no readable source/binary"
            % (module_name, path))
    with open(origin, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return {"module": module_name, "sha256": digest}


__all__ = ["module_implementation_fingerprint"]
