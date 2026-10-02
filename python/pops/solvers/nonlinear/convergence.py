"""Authenticated stopping policy of the complete Original residual."""
import math
from collections.abc import Mapping
from pops.identity.scalar import scalar_data
from pops.solvers.tolerances import Relative, Absolute, AbsoluteFloor
from pops.solvers._numeric import exact_positive_real
CONTRACT = "pops.newton.original-residual-convergence@1"

def lower_tolerance(value):
    if type(value) is Relative:
        if value.floor is not None and type(value.floor) is not AbsoluteFloor:
            raise ValueError("relative Newton floor must be an AbsoluteFloor")
        rel, absolute, kind = float(exact_positive_real(value.rel, where="Newton relative tolerance")), 0.0 if value.floor is None else float(exact_positive_real(value.floor.abs_floor, where="Newton absolute floor")), "relative"
    elif type(value) is Absolute:
        rel, absolute, kind = 0.0, float(exact_positive_real(value.abs_tol, where="Newton absolute tolerance")), "absolute"
    else:
        return None
    data = {"contract": CONTRACT, "kind": kind, "relative": scalar_data(rel), "absolute": scalar_data(absolute)}
    validate_convergence(data)
    return data

def validate_convergence(data):
    if not isinstance(data, Mapping) or set(data) != {"contract", "kind", "relative", "absolute"} or data["contract"] != CONTRACT:
        raise ValueError("invalid Original Newton convergence contract")
    values = []
    for key in ("relative", "absolute"):
        item = data[key]
        if not isinstance(item, Mapping) or set(item) != {"kind", "value"}:
            raise ValueError("convergence requires canonical binary64")
        scalar = item
        if not isinstance(scalar, Mapping) or set(scalar) != {"kind", "value"} or scalar["kind"] != "binary64" or type(scalar["value"]) is not str:
            raise ValueError("convergence requires canonical binary64")
        value = float.fromhex(scalar["value"])
        if not math.isfinite(value) or value < 0 or (value == 0 and math.copysign(1, value) < 0) or scalar_data(value) != item:
            raise ValueError("invalid convergence coefficient")
        values.append(value)
    rel, absolute = values
    if data["kind"] == "relative" and rel > 0:
        return rel, absolute
    if data["kind"] == "absolute" and rel == 0 and absolute > 0:
        return rel, absolute
    raise ValueError("invalid Original Newton convergence kind or coefficients")

def convergence_cpp(data, scalar_cpp):
    rel, absolute = validate_convergence(data)
    kind = "kRelative" if data["kind"] == "relative" else "kAbsolute"
    return "pops::FieldNewtonConvergence{pops::FieldNewtonConvergenceKind::%s, %s, %s}" % (kind, scalar_cpp(rel), scalar_cpp(absolute))
