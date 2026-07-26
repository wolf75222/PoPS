"""Source-only compiled Program stub for exact artifact-contract tests."""
from __future__ import annotations

from typing import Any

from pops.identity import make_identity


class CompiledProgramStub:
    """Minimum immutable metadata carried by a compiled per-layout Program."""

    def __init__(
        self,
        *,
        target: str,
        block_names: Any,
        abi_key: str,
        name: str = "program",
    ) -> None:
        names = tuple(block_names)
        self.name = name
        self.program = None
        self.program_name = name
        self.program_hash = None
        self.cache_key = None
        self.so_path = "/tmp/%s.so" % name
        self.target = target
        self.backend = "production"
        self.abi_key = abi_key
        self.cxx = "clang++"
        self.std = "c++23"
        self.program_block_routes = tuple(enumerate(names))
        self.artifact_identity = make_identity(
            "artifact",
            {"component": name, "target": target, "blocks": list(names)},
        )


__all__ = ["CompiledProgramStub"]
