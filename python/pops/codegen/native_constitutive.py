"""One authenticated joint law at an existing numerical consumer's sampling location."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeConstitutiveEmission:
    lines: tuple[str, ...]
    values: tuple[str, ...]
    status: str
    reason: str
    functions: tuple[Any, ...]


def emit_native_constitutive(expressions, *, indent="    "):
    """Materialize each selected native call once; the consumer owns status aggregation.

    This helper does not choose cell/face interpolation, numerical stencils, quadrature,
    communicator or publication. The existing prepared consumer supplies that context and
    must check ``status`` before reading a failed result's components.
    """
    from pops._ir.native_call import native_functions
    from .cpp_writer import _cse_emit
    expressions = tuple(expressions)
    lines, values, statuses = _cse_emit(expressions, "pops::Real", indent,
                                       return_native_statuses=True)
    functions = native_functions(expressions)
    if not statuses:
        return NativeConstitutiveEmission(tuple(lines), tuple(values), "0", "0", functions)
    status, reason = "constitutive_status_", "constitutive_reason_"
    lines += [indent + "int %s = 0;" % status, indent + "unsigned %s = 0;" % reason]
    for name in statuses:
        lines += [indent + "{",
            indent + "  const int raw_ = static_cast<int>(%s.status);" % name,
            indent + "  const int selected_ = raw_ >= 0 && raw_ <= 3 ? raw_ : 3;",
            indent + "  if (selected_ > %s) { %s = selected_; %s = %s.reason; }" %
                (status, status, reason, name),
            indent + "  else if (selected_ == %s && %s.reason > %s) %s = %s.reason;" %
                (status, name, reason, reason, name),
            indent + "}"]
    return NativeConstitutiveEmission(tuple(lines), tuple(values), status, reason, functions)
