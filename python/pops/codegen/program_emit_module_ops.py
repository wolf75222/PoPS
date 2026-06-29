"""GeneratedModule::Operators emission for Program ``P.call`` nodes.

The Program IR owns sequencing only.  A typed ``P.call(handle, ...)`` lowers to a
first-class ``call`` node carrying ``operator_id`` and ``output_type``; the
numerical route attached to that operator is emitted here as a generated module
operator function.  Runtime primitives such as ``ctx.rhs_into`` and
``ctx.solve_fields_from_state`` remain the C++ execution seams, but they are now
called by ``GeneratedModule::Operators::<op>``, not by the Program dispatcher.
"""

import re

from pops.codegen.program_emit_model_kernels import (
    _emit_flux_kernel,
    _emit_source_kernel,
)
from pops.codegen.program_emit_kernels import _model_impl


def operator_function_name(operator_id, operator_name):
    """Stable C++ function name for an operator registry id/name pair."""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", str(operator_name))
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return safe


def _operator_enum_name(operator_id, operator_name):
    safe = operator_function_name(operator_id, operator_name)
    return "%s_%d" % (safe, int(operator_id))


def _is_default(op, family):
    if op.capabilities.get("default"):
        return True
    if family == "fields":
        return op.name in ("fields", "fields_from_state")
    if family == "flux":
        return op.name in ("flux", "flux_default")
    if family == "source":
        return op.name in ("source", "source_default", "default")
    return False


def _named_fluxes_from_lowering(lowering):
    fluxes = lowering.get("fluxes")
    if not fluxes or fluxes == ["default"]:
        return None
    named = [f for f in fluxes if f != "default"]
    if len(named) != len(fluxes):
        raise ValueError(
            "operator lowering mixes 'default' with named fluxes %r; request either the default "
            "Riemann flux or a named-flux sum" % named)
    return named


def _rate_lines(model, op, state_var="state", out_var="out", block_expr="b"):
    """C++ body of a Rate(State) operator function."""
    lines = []
    if op.kind == "grid_operator":
        if _is_default(op, "flux"):
            lines.append("ctx.neg_div_flux_default_into(%s, %s, %s);" %
                         (block_expr, state_var, out_var))
        else:
            fx = "%s_fx" % out_var
            fy = "%s_fy" % out_var
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fx, state_var))
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fy, state_var))
            lines += _emit_flux_kernel(model, [op.name], state_var, fx, fy, block_expr)
            lines.append("ctx.neg_div_flux_into(%s, %s, %s);" % (out_var, fx, fy))
        return lines

    if op.kind == "local_source":
        if _is_default(op, "source"):
            lines.append("ctx.source_default_into(%s, %s, %s);" %
                         (block_expr, state_var, out_var))
        else:
            lines += _emit_source_kernel(model, op.name, state_var, out_var, block_expr)
        return lines

    if op.kind != "local_rate":
        raise ValueError(
            "GeneratedModule::Operators cannot emit Rate operator %r of kind %r"
            % (op.name, op.kind))

    lowering = dict(op.lowering)
    named_fluxes = _named_fluxes_from_lowering(lowering)
    requested = lowering.get("sources")
    want_flux = lowering.get("flux", True)
    default_sources = ("default", "source_default", "source")
    want_default_source = requested is None or any(s in default_sources for s in requested)
    if not want_flux:
        if want_default_source:
            lines.append("ctx.source_default_into(%s, %s, %s);" %
                         (block_expr, state_var, out_var))
    elif named_fluxes is None:
        if want_default_source:
            lines.append("ctx.rhs_into(%s, %s, %s);" % (block_expr, state_var, out_var))
        else:
            lines.append("ctx.neg_div_flux_default_into(%s, %s, %s);" %
                         (block_expr, state_var, out_var))
    else:
        fx = "%s_fx" % out_var
        fy = "%s_fy" % out_var
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fx, state_var))
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fy, state_var))
        lines += _emit_flux_kernel(model, named_fluxes, state_var, fx, fy, block_expr)
        lines.append("ctx.neg_div_flux_into(%s, %s, %s);" % (out_var, fx, fy))
    for source in [s for s in (requested or []) if s not in default_sources]:
        scratch = "%s_%s" % (out_var, source)
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (scratch, state_var))
        lines += _emit_source_kernel(model, source, state_var, scratch, block_expr)
        lines.append("ctx.axpy(%s, static_cast<pops::Real>(1), %s);" % (out_var, scratch))
    return lines


def _emit_one_operator(model, op, operator_id):
    fn = operator_function_name(operator_id, op.name)
    header = []
    body = []
    if op.kind == "field_operator":
        if len(op.signature.inputs) > 1:
            header.append(
                "static void %s(const pops::runtime::program::ProgramContext& ctx, "
                "const std::vector<const pops::MultiFab*>& u_stages) {" % fn)
            body.append("ctx.solve_fields_from_blocks(u_stages);")
        else:
            header.append(
                "static void %s(const pops::runtime::program::ProgramContext& ctx, int b, "
                "pops::MultiFab& state) {" % fn)
            if _is_default(op, "fields"):
                body.append("ctx.solve_fields_from_state(b, state);")
            else:
                body.append('ctx.solve_fields_from_state("%s", b, state);' % op.name)
    elif op.kind in ("grid_operator", "local_source", "local_rate"):
        header.append(
            "static void %s(const pops::runtime::program::ProgramContext& ctx, int b, "
            "pops::MultiFab& state, pops::MultiFab& out) {" % fn)
        body += _rate_lines(model, op)
    elif op.kind == "projection":
        header.append(
            "static void %s(const pops::runtime::program::ProgramContext& ctx, int b, "
            "pops::MultiFab& state) {" % fn)
        body.append("ctx.apply_projection(b, state);")
    elif op.kind == "local_linear_operator":
        header.append(
            "static LocalLinearOperatorView %s(const pops::runtime::program::ProgramContext& ctx, int b, "
            "pops::MultiFab& fields) {" % fn)
        body.append("(void)ctx;")
        body.append("(void)b;")
        body.append("(void)fields;")
        body.append(
            "return LocalLinearOperatorView{LocalLinearOperatorView::Id::%s};"
            % _operator_enum_name(operator_id, op.name))
    else:
        return [
            "// Operator %s (%s) is metadata-only for this Program codegen path."
            % (op.name, op.kind)
        ]
    return header + ["  " + line for line in body] + ["}"]


def emit_generated_module_operators(program, model=None):
    """C++ namespace containing one function per typed Module operator."""
    registry = None
    if model is not None and hasattr(model, "operator_registry"):
        registry = model.operator_registry()
    elif getattr(program, "_registry", None) is not None:
        registry = program._registry
    if registry is None:
        return (
            "namespace GeneratedModule {\n"
            "namespace Operators {\n"
            "}  // namespace Operators\n"
            "}  // namespace GeneratedModule\n"
        )
    # Validate the model can be viewed by the module-native codegen when any emitted operator needs
    # model formulas.  Pure field/projection operator wrappers can be emitted from the bound registry
    # alone (multi-species field solve tests use this path).
    if any(op.kind in ("grid_operator", "local_source", "local_rate", "local_linear_operator")
           for op in registry):
        if model is None:
            raise ValueError(
                "GeneratedModule::Operators: rate/source/linear operators require model= so their "
                "C++ bodies can be emitted; only field/projection operators can lower from a bound "
                "registry alone")
        _model_impl(model)
    lines = [
        "namespace GeneratedModule {",
        "namespace Operators {",
    ]
    linear_ops = [registry.get(name) for name in registry.names()
                  if registry.get(name).kind == "local_linear_operator"]
    if linear_ops:
        lines += [
            "struct LocalLinearOperatorView {",
            "  enum class Id {",
        ]
        for op in linear_ops:
            lines.append(
                "    %s," % _operator_enum_name(registry.id_of(op.name), op.name))
        lines += [
            "  };",
            "  Id id;",
            "};",
        ]
    for name in registry.names():
        op = registry.get(name)
        operator_id = registry.id_of(name)
        lines.append("// OperatorId %d: %s (%s)" % (operator_id, op.name, op.kind))
        lines += _emit_one_operator(model, op, operator_id)
    lines += [
        "}  // namespace Operators",
        "}  // namespace GeneratedModule",
    ]
    return "\n".join(lines)
