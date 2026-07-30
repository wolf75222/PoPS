"""Bind-time installation of the immutable cadence carried by a compiled Program."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast


def install_program_cadence(engine: Any, program: Any) -> None:
    """Install one authenticated cadence before the native Program and runtime freeze."""
    from pops.time._cadence import ProgramCadence
    from pops.time._program.contract import require_program

    require_program(program, exact=True, where="pops.bind Program cadence")
    if getattr(program, "_compiled_detached", False) is not True \
            or getattr(program, "_frozen", False) is not True:
        raise TypeError(
            "pops.bind Program cadence requires the frozen compiled Program authority"
        )
    contract = program.cadence_contract()
    if type(contract) is not ProgramCadence:
        raise TypeError("pops.bind Program cadence is not an exact ProgramCadence")
    setter = getattr(engine, "set_program_cadence", None)
    if not callable(setter):
        raise RuntimeError("pops.bind runtime cannot install the authored Program cadence")
    setter(contract.substeps, contract.stride)

    substeps = getattr(engine, "program_substeps", None)
    stride = getattr(engine, "program_stride", None)
    if not callable(substeps) or not callable(stride):
        raise RuntimeError("pops.bind runtime cannot authenticate the installed Program cadence")
    installed_substeps = cast(Callable[[], int], substeps)
    installed_stride = cast(Callable[[], int], stride)
    actual = (int(installed_substeps()), int(installed_stride()))
    expected = (contract.substeps, contract.stride)
    if actual != expected:
        raise RuntimeError(
            "pops.bind runtime Program cadence differs from the compiled contract: "
            "expected=%r actual=%r" % (expected, actual)
        )


__all__ = ["install_program_cadence"]
