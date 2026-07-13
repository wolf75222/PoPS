"""Executable cell-local linear solver descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.descriptors import Descriptor
from pops.identity import Identity, make_identity


@dataclass(frozen=True, slots=True)
class _PreparedDenseLU:
    identity: Identity

    def build_program_solve(self, *, program: Any, problem: Any,
                            name: Any = None) -> Any:
        build = getattr(problem, "build_with", None)
        if not callable(build):
            raise TypeError("DenseLU requires a typed Program solve problem")
        return build(program=program, prepared_solver=self, name=name)


class DenseLU(Descriptor):
    """Exact per-cell dense factorization for a typed ``LocalLinear`` problem."""

    category = "local_linear_solver"
    name = "dense_lu"
    native_id = "pops::detail::mat_inverse"
    scheme = "dense_lu"

    def prepare_program_solve(self) -> _PreparedDenseLU:
        payload = {"schema_version": 1, "scheme": self.scheme}
        return _PreparedDenseLU(make_identity("prepared-dense-lu", payload))

    def to_data(self) -> dict[str, Any]:
        return {"scheme": self.scheme}


__all__ = ["DenseLU"]
