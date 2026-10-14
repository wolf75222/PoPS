"""Validation of a complete finite state on the existing whole-System Program path."""
from __future__ import annotations


def require_finite_state_program(program):
    from pops.time import Program
    from pops.time._program.versioned_values import validate_continuations
    if type(program) is not Program:
        raise TypeError("a block-free Case requires a complete finite-state Program")
    validate_continuations(program, target="system")
    roots = [v for v in program._values if v.op == "continuation_open"]
    if len(roots) != 1 or roots[0].attrs["schema"]["fields"]:
        raise ValueError("a block-free Case requires one field-free complete Continuation")
    if program._block_indices():
        raise ValueError("a finite-state Program cannot reference undeclared native blocks")
    def check(sig):
        if sig["vtype"] in ("state", "rhs", "scalar_field"):
            raise ValueError("a field-free Continuation cannot hide native fields in its packet")
        for _, child in sig.get("members", ()):
            check(child)
        if "element" in sig:
            check(sig["element"])
    check(roots[0].attrs["schema"]["read"])
    return program
