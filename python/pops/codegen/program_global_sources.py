"""Bind physical global leaves to authorized PODs before the source cell kernel."""
from collections.abc import Mapping
from pops._ir.expr import Expr, Var
from pops.model.global_quantity import GlobalQuantityRef, global_references


def bind_source_globals(expressions, evaluation, variables, *, source_module=None):
    import json
    from pops.time._program.integrals import integral_identity, integral_units_bytes
    references = global_references(expressions)
    rows = () if evaluation is None else evaluation.attrs.get("physical_global_inputs_v1", ())
    if not references and not rows: return expressions, []
    if evaluation is None or evaluation.op != "source":
        raise NotImplementedError("physical global inputs require a bound named Equation source evaluation")
    wanted = {ref.handle._resolved(): ref.units for ref in references}
    bound, prelude = {}, []
    for row in rows:
        if (set(row) != {"port", "units", "input", "version"}
                or type(row["version"]) is not int or row["version"] != 1):
            raise ValueError("physical global source binding metadata changed")
        port = row["port"]
        if not port.is_instance or port.block_ref != evaluation.state_ref.block_ref:
            raise ValueError("physical global source binding block owner changed")
        if getattr(evaluation.prog, "_compiled_detached", False):
            from pops.time._program.global_source_plan import require_detached_source_global
            if source_module is None:
                raise ValueError("detached physical global source requires Module/body authority")
            require_detached_source_global(evaluation, module=source_module)
        elif port.block_ref[port.declaration_ref] is not port:
            raise ValueError("physical global source input is not the registry-issued port")
        declaration = port.declaration_ref._resolved()
        if declaration not in wanted or declaration in bound or port.units != wanted[declaration]:
            raise ValueError("physical global source binding declaration/units changed")
        index = row["input"]
        if type(index) is not int or not 1 <= index < len(evaluation.inputs):
            raise ValueError("physical global source binding input index changed")
        capture = evaluation.prog._canonical_value(evaluation.inputs[index])
        units = integral_units_bytes(wanted[declaration])
        if (capture.op != "integral_candidate" or capture.vtype != "scalar"
                or capture.point != evaluation.point or row["units"] != units
                or capture.attrs.get("units") != units or capture.attrs.get("scope") != "candidate"):
            raise ValueError("physical global source binding capture point/units/scope changed")
        symbol = "physical_global_%d_%d" % (evaluation.id, index)
        prelude.append("const pops::Real %s = ctx.integral_candidate_value(%s,%s,%s);" %
            (symbol, variables[capture.id], json.dumps(integral_identity(capture.prog,capture.attrs["integral"])),
             json.dumps(units)))
        bound[declaration] = Var(symbol, "physical_global")
    if set(bound) != set(wanted):
        raise ValueError("physical global source requires every declared input before its kernel")
    memo = {}
    def clone(value):
        if isinstance(value, GlobalQuantityRef): return bound[value.handle._resolved()]
        if isinstance(value, Expr):
            if id(value) in memo: return memo[id(value)]
            result = object.__new__(type(value)); memo[id(value)] = result
            for base in reversed(type(value).__mro__):
                slots = base.__dict__.get("__slots__", ())
                for key in (slots,) if isinstance(slots, str) else slots:
                    if key not in ("__dict__", "__weakref__", "_pops_symbolic_initializing") and hasattr(value,key):
                        object.__setattr__(result,key,clone(getattr(value,key)))
            for key, child in getattr(value,"__dict__",{}).items(): object.__setattr__(result,key,clone(child))
            object.__setattr__(result,"_pops_symbolic_initializing",False)
            return result
        if isinstance(value, Mapping): return {key:clone(child) for key,child in value.items()}
        if isinstance(value,(tuple,list)): return tuple(clone(child) for child in value)
        return value
    return clone(expressions), prelude
