"""Generated device launchers for dynamic field boundary laws.

The authoring Expr graph is lowered once into named C++ functors.  Runtime iteration sees direct
function pointers and POD captures only: no Python callback, string lookup, virtual dispatch or host
expression interpreter occurs in a face-cell loop.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_FACE_INDEX = {(0, "lo"): 0, (0, "hi"): 1, (1, "lo"): 2, (1, "hi"): 3}


def _raw_faces(plan: Any) -> tuple[Any, ...]:
    """Return the already-validated condition object for each Cartesian physical face."""
    from pops.fields.bcs import AllPhysicalBoundaries, AxisBoundary

    faces = [None, None, None, None]
    for binding in plan.discretization.boundaries:
        selector = binding.selector
        if isinstance(selector, AllPhysicalBoundaries):
            selected = range(4)
        elif isinstance(selector, AxisBoundary):
            selected = (_FACE_INDEX[(selector.axis, selector.side)],)
        else:  # resolve_field_install_plan already rejected this route
            raise TypeError("dynamic field boundary selector has no Cartesian lowering")
        for face in selected:
            if faces[face] is not None:
                raise ValueError("dynamic field boundary face is assigned more than once")
            faces[face] = binding.condition
    if any(item is None for item in faces):
        raise ValueError("dynamic field boundary plan is incomplete")
    return tuple(faces)


def _coefficients(condition: Any) -> tuple[Any, Any, Any]:
    from pops.fields.bcs import Dirichlet, Mixed, Neumann, Periodic

    if isinstance(condition, Dirichlet):
        return 1, 0, condition.value
    if isinstance(condition, Neumann):
        return 0, 1, condition.flux
    if isinstance(condition, Mixed):
        return condition.alpha, condition.beta, condition.value
    if isinstance(condition, Periodic):
        return 0, 0, 0
    raise TypeError("dynamic field boundary condition has no residual lowering")


def _as_expr(value: Any) -> Any:
    from pops.ir.expr import Expr, _wrap
    from pops.model import Handle
    from pops.ir.handle_expr import ValueExpr

    if isinstance(value, Handle):
        return ValueExpr(value)
    return value if isinstance(value, Expr) else _wrap(value)


class _ExprCpp:
    def __init__(self, *, unknown: Any, parameter_indices: Mapping[str, int],
                 state_indices: Mapping[tuple[str, int], int],
                 field_indices: Mapping[tuple[str, int], int]) -> None:
        self.unknown = unknown
        self.parameter_indices = parameter_indices
        self.state_indices = state_indices
        self.field_indices = field_indices
        self.used_parameters: set[int] = set()
        self.used_states: set[int] = set()
        self.used_fields: set[int] = set()
        self.used_times: set[str] = set()

    def _param_index(self, ref: Any) -> int:
        handle = getattr(ref, "handle", None)
        if handle is None:
            raise ValueError(
                "dynamic boundary RuntimeParamRef requires an owner-qualified ParamHandle")
        try:
            index = self.parameter_indices[handle.qualified_id]
        except KeyError:
            raise ValueError(
                "dynamic boundary parameter %s is absent from its resolved dependency pack"
                % handle.qualified_id) from None
        self.used_parameters.add(index)
        return index

    def emit(self, value: Any) -> str:
        from pops.ir.expr import Const, Var, _Bin, Neg, Sqrt, Abs, Sign, Pow
        from pops.ir.handle_expr import ValueExpr
        from pops.ir.values import RuntimeParamRef
        from pops.fields.boundary_values import BoundaryValue, LogicalTimeValue

        value = _as_expr(value)
        if isinstance(value, Const):
            return value.to_cpp()
        if isinstance(value, RuntimeParamRef):
            return "p%d" % self._param_index(value)
        if isinstance(value, BoundaryValue):
            if value.handle == self.unknown:
                return "u"
            key = (value.handle.qualified_id, value.component)
            indices = self.state_indices if value.handle.kind == "state" else self.field_indices
            try:
                index = indices[key]
            except KeyError:
                raise ValueError(
                    "boundary value %s[%d] is absent from its resolved direct-buffer pack"
                    % key) from None
            if value.handle.kind == "state":
                self.used_states.add(index)
                return "state%d(i, j, %d)" % (index, value.component)
            self.used_fields.add(index)
            return "field%d(i, j, %d)" % (index, value.component)
        if isinstance(value, LogicalTimeValue):
            coordinate = value.coordinate
            self.used_times.add(coordinate)
            return "logical_%s" % coordinate
        if isinstance(value, ValueExpr):
            if value.handle.kind == "parameter":
                proxy = type("BoundaryParamRef", (), {
                    "handle": value.handle,
                    "name": value.handle.local_id,
                })()
                return "p%d" % self._param_index(proxy)
            if value.handle.qualified_id != self.unknown.qualified_id:
                raise NotImplementedError(
                    "dynamic boundary value dependency %s is not yet routed to a prepared "
                    "state/field buffer" % value.handle.qualified_id)
            return "u"
        if isinstance(value, Var):
            raise ValueError(
                "dynamic boundary expression contains unqualified Var(%r); use a typed Handle"
                % value.name)
        if isinstance(value, Neg):
            return "(-%s)" % self.emit(value.a)
        if isinstance(value, Sqrt):
            return "std::sqrt(%s)" % self.emit(value.a)
        if isinstance(value, Abs):
            return "std::fabs(%s)" % self.emit(value.a)
        if isinstance(value, Sign):
            inner = self.emit(value.a)
            return "(pops::Real(%s > 0) - pops::Real(%s < 0))" % (inner, inner)
        if isinstance(value, Pow):
            return "std::pow(%s, %s)" % (self.emit(value.a), self.emit(value.b))
        if isinstance(value, _Bin):
            return "(%s %s %s)" % (self.emit(value.a), value.op, self.emit(value.b))
        raise TypeError("dynamic boundary expression node %s has no C++ lowering" %
                        type(value).__name__)


def _parameter_handles(plan: Any) -> tuple[Any, ...]:
    return plan.boundary_parameter_handles()


def _face_struct(face: int, condition: Any, *, symbol: int, cpp: _ExprCpp,
                 unknown: Any) -> str:
    from pops.ir.lowering import diff

    alpha, beta, value = (_as_expr(item) for item in _coefficients(condition))
    a = cpp.emit(alpha)
    b = cpp.emit(beta)
    v = cpp.emit(value)
    da = cpp.emit(diff(alpha, unknown))
    db = cpp.emit(diff(beta, unknown))
    dv = cpp.emit(diff(value, unknown))
    params = "\n".join("  pops::Real p%d;" % index for index in sorted(cpp.used_parameters))
    if params:
        params += "\n"
    dependencies = "\n".join(
        ["  pops::ConstArray4 state%d;" % index for index in sorted(cpp.used_states)] +
        ["  pops::ConstArray4 field%d;" % index for index in sorted(cpp.used_fields)] +
        ["  pops::Real logical_%s;" % name for name in sorted(cpp.used_times)])
    if dependencies:
        dependencies += "\n"
    axis = 0 if face < 2 else 1
    boundary_i = "geometry.domain.lo[0]" if face == 0 else (
        "geometry.domain.hi[0]" if face == 1 else "i")
    boundary_j = "geometry.domain.lo[1]" if face == 2 else (
        "geometry.domain.hi[1]" if face == 3 else "j")
    h = "geometry.dx()" if axis == 0 else "geometry.dy()"
    mirror_i = ("2 * geometry.domain.lo[0] - i - 1" if face == 0 else
                "2 * geometry.domain.hi[0] - i + 1" if face == 1 else "i")
    mirror_j = ("2 * geometry.domain.lo[1] - j - 1" if face == 2 else
                "2 * geometry.domain.hi[1] - j + 1" if face == 3 else "j")
    layer = ("geometry.domain.lo[0] - i" if face == 0 else
             "i - geometry.domain.hi[0]" if face == 1 else
             "geometry.domain.lo[1] - j" if face == 2 else
             "j - geometry.domain.hi[1]")
    boundary_test = (
        "box.lo[0] == geometry.domain.lo[0]" if face == 0 else
        "box.hi[0] == geometry.domain.hi[0]" if face == 1 else
        "box.lo[1] == geometry.domain.lo[1]" if face == 2 else
        "box.hi[1] == geometry.domain.hi[1]")
    valid_box = (
        "pops::Box2D{{geometry.domain.lo[0], box.lo[1]}, {geometry.domain.lo[0], box.hi[1]}}"
        if face == 0 else
        "pops::Box2D{{geometry.domain.hi[0], box.lo[1]}, {geometry.domain.hi[0], box.hi[1]}}"
        if face == 1 else
        "pops::Box2D{{box.lo[0], geometry.domain.lo[1]}, {box.hi[0], geometry.domain.lo[1]}}"
        if face == 2 else
        "pops::Box2D{{box.lo[0], geometry.domain.hi[1]}, {box.hi[0], geometry.domain.hi[1]}}")
    ghost_box = (
        "pops::Box2D{{geometry.domain.lo[0] - ng, box.lo[1]}, "
        "{geometry.domain.lo[0] - 1, box.hi[1]}}" if face == 0 else
        "pops::Box2D{{geometry.domain.hi[0] + 1, box.lo[1]}, "
        "{geometry.domain.hi[0] + ng, box.hi[1]}}" if face == 1 else
        "pops::Box2D{{box.lo[0], geometry.domain.lo[1] - ng}, "
        "{box.hi[0], geometry.domain.lo[1] - 1}}" if face == 2 else
        "pops::Box2D{{box.lo[0], geometry.domain.hi[1] + 1}, "
        "{box.hi[0], geometry.domain.hi[1] + ng}}")
    param_args = ", ".join("params[%d]" % index for index in sorted(cpp.used_parameters))
    if param_args:
        param_args = ", " + param_args
    parameter_setup = ""
    if cpp.used_parameters:
        parameter_setup = f"""  if (context.parameters == nullptr ||
      context.parameter_count <= {max(cpp.used_parameters)})
    throw std::runtime_error("dynamic field boundary parameter carrier is incomplete");
  const auto& params = *context.parameters;
"""
    dependency_checks = []
    if cpp.used_states:
        dependency_checks.append(
            "  if (context.states == nullptr || context.state_count <= %d)\n"
            "    throw std::runtime_error(\"dynamic field boundary state carrier is incomplete\");"
            % max(cpp.used_states))
    if cpp.used_fields:
        dependency_checks.append(
            "  if (context.fields == nullptr || context.field_count <= %d)\n"
            "    throw std::runtime_error(\"dynamic field boundary field carrier is incomplete\");"
            % max(cpp.used_fields))
    dependency_setup = "\n".join(dependency_checks)

    def law_args(local_index: str) -> str:
        args = ["iterate.fab(%s).const_array()" % local_index]
        args.extend("context.states[%d]->fab(%s).const_array()" % (index, local_index)
                    for index in sorted(cpp.used_states))
        args.extend("context.fields[%d]->fab(%s).const_array()" % (index, local_index)
                    for index in sorted(cpp.used_fields))
        point_names = {
            "time": "time", "dt": "dt", "step": "step", "substep": "substep",
            "iteration": "iteration", "stage": "stage_slot",
            "partition": "partition_slot",
        }
        args.extend("static_cast<pops::Real>(context.point.%s)" % point_names[name]
                    for name in sorted(cpp.used_times))
        args.extend("params[%d]" % index for index in sorted(cpp.used_parameters))
        return ", ".join(args)
    return f"""
struct FieldBoundaryFace{symbol} {{
  pops::ConstArray4 iterate;
{dependencies}{params}  POPS_HD pops::Real alpha(int i, int j) const {{ const pops::Real u = iterate(i, j); return {a}; }}
  POPS_HD pops::Real beta(int i, int j) const {{ const pops::Real u = iterate(i, j); return {b}; }}
  POPS_HD pops::Real value(int i, int j) const {{ const pops::Real u = iterate(i, j); return {v}; }}
  POPS_HD pops::Real dalpha(int i, int j) const {{ const pops::Real u = iterate(i, j); return {da}; }}
  POPS_HD pops::Real dbeta(int i, int j) const {{ const pops::Real u = iterate(i, j); return {db}; }}
  POPS_HD pops::Real dvalue(int i, int j) const {{ const pops::Real u = iterate(i, j); return {dv}; }}
  POPS_HD pops::Real denominator(int i, int j, pops::Real distance) const {{
    return alpha(i, j) / pops::Real(2) + beta(i, j) / distance;
  }}
}};

struct FieldBoundaryValidate{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::Geometry geometry;
  POPS_HD void operator()(int i, int j, pops::Real& out) const {{
    const pops::Real aa = law.alpha(i, j);
    const pops::Real bb = law.beta(i, j);
    const pops::Real vv = law.value(i, j);
    const pops::Real denom = aa / pops::Real(2) + bb / {h};
    const pops::Real scale = std::fmax(pops::Real(1),
        std::fmax(std::fabs(aa / pops::Real(2)), std::fabs(bb / {h})));
    const bool invalid = !std::isfinite(aa) || !std::isfinite(bb) || !std::isfinite(vv) ||
        !std::isfinite(denom) || std::fabs(denom) <=
        pops::Real(64) * std::numeric_limits<pops::Real>::epsilon() * scale;
    if (invalid) {{
      const int linear = (j - geometry.domain.lo[1]) * geometry.domain.nx() +
                         (i - geometry.domain.lo[0]);
      const pops::Real encoded = pops::Real(geometry.domain.nx() * geometry.domain.ny() - linear);
      if (encoded > out) out = encoded;
    }}
  }}
}};

struct FieldBoundaryResidualGhost{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::Array4 output;
  pops::Geometry geometry;
  POPS_HD void operator()(int i, int j) const {{
    const int bi = {boundary_i};
    const int bj = {boundary_j};
    const pops::Real distance = pops::Real(2 * ({layer}) - 1) * {h};
    const pops::Real aa = law.alpha(bi, bj);
    const pops::Real bb = law.beta(bi, bj);
    const pops::Real vv = law.value(bi, bj);
    const pops::Real inner = law.iterate({mirror_i}, {mirror_j});
    output(i, j) = (vv - inner * (aa / pops::Real(2) - bb / distance)) /
                   (aa / pops::Real(2) + bb / distance);
  }}
}};

struct FieldBoundaryJvpGhost{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::ConstArray4 direction;
  pops::Array4 output;
  pops::Geometry geometry;
  POPS_HD void operator()(int i, int j) const {{
    const int bi = {boundary_i};
    const int bj = {boundary_j};
    const pops::Real distance = pops::Real(2 * ({layer}) - 1) * {h};
    const pops::Real aa = law.alpha(bi, bj), bb = law.beta(bi, bj);
    const pops::Real vv = law.value(bi, bj);
    const pops::Real daa = law.dalpha(bi, bj), dbb = law.dbeta(bi, bj);
    const pops::Real dvv = law.dvalue(bi, bj);
    const pops::Real inner = law.iterate({mirror_i}, {mirror_j});
    const pops::Real dinner = direction({mirror_i}, {mirror_j});
    const pops::Real du = direction(bi, bj);
    const pops::Real denom = aa / pops::Real(2) + bb / distance;
    const pops::Real numer = vv - inner * (aa / pops::Real(2) - bb / distance);
    const pops::Real ddenom = du * (daa / pops::Real(2) + dbb / distance);
    const pops::Real dnumer = dvv * du - dinner * (aa / pops::Real(2) - bb / distance)
        - inner * du * (daa / pops::Real(2) - dbb / distance);
    output(i, j) = (dnumer * denom - numer * ddenom) / (denom * denom);
  }}
}};

static void prepare_field_boundary_residual_{symbol}(
    int requested_face, const pops::MultiFab& iterate, pops::MultiFab& output,
    const pops::Geometry& geometry, const pops::FieldBoundaryExecutionContext& context) {{
  if (requested_face != {face}) return;
  if (context.failure == nullptr)
    throw std::runtime_error("dynamic field boundary has no fallible execution channel");
  if (context.failure->failed()) return;
{parameter_setup.rstrip()}
{dependency_setup}
  pops::Real best = pops::Real(0);
  int best_li = -1;
  for (int li = 0; li < iterate.local_size(); ++li) {{
    const pops::Box2D box = iterate.box(li);
    if (!({boundary_test})) continue;
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    const pops::Real encoded = pops::reduce_max_cell(
        {valid_box}, FieldBoundaryValidate{symbol}{{law, geometry}});
    if (encoded > best) {{ best = encoded; best_li = li; }}
  }}
  if (best > pops::Real(0)) {{
    const int linear = geometry.domain.nx() * geometry.domain.ny() - static_cast<int>(best);
    const int i = geometry.domain.lo[0] + linear % geometry.domain.nx();
    const int j = geometry.domain.lo[1] + linear / geometry.domain.nx();
    const FieldBoundaryFace{symbol} law{{{law_args("best_li")}}};
    context.failure->code = 1; context.failure->face = {face};
    context.failure->i = i; context.failure->j = j;
    context.failure->value = law.denominator(i, j, {h});
    return;
  }}
  const int ng = output.n_grow();
  for (int li = 0; li < output.local_size(); ++li) {{
    const pops::Box2D box = output.box(li);
    if (!({boundary_test})) continue;
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    pops::for_each_cell({ghost_box},
        FieldBoundaryResidualGhost{symbol}{{law, output.fab(li).array(), geometry}});
  }}
}}

static void prepare_field_boundary_jvp_{symbol}(
    int requested_face, const pops::MultiFab& iterate, const pops::MultiFab& direction,
    pops::MultiFab& output, const pops::Geometry& geometry,
    const pops::FieldBoundaryExecutionContext& context) {{
  if (requested_face != {face}) return;
  if (context.failure == nullptr)
    throw std::runtime_error("dynamic field boundary JVP has no fallible execution channel");
  if (context.failure->failed()) return;
{parameter_setup.rstrip()}
{dependency_setup}
  const int ng = output.n_grow();
  for (int li = 0; li < output.local_size(); ++li) {{
    const pops::Box2D box = output.box(li);
    if (!({boundary_test})) continue;
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    pops::for_each_cell({ghost_box}, FieldBoundaryJvpGhost{symbol}{{
        law, direction.fab(li).const_array(), output.fab(li).array(), geometry}});
  }}
}}
"""


def emit_field_boundaries(program: Any, authority: Any, field_plans: Any, target: str) -> str:
    """Emit optional boundary launchers + the target-specific install entry."""
    dynamic = [(name, plan) for name, plan in sorted((field_plans or {}).items())
               if plan.native_options.get("boundary_kernel_required")]
    if not dynamic:
        return ""
    chunks = [
        "// Generated dynamic field boundary residual/JVP launchers.",
        "#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>",
        "#include <pops/mesh/execution/for_each.hpp>",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <limits>",
        "#include <stdexcept>",
        "namespace {",
        "static void field_boundary_noop(int, const pops::MultiFab&, pops::MultiFab&, "
        "const pops::Geometry&, const pops::FieldBoundaryExecutionContext&) {}",
        "static void field_boundary_jvp_noop(int, const pops::MultiFab&, const pops::MultiFab&, "
        "pops::MultiFab&, const pops::Geometry&, const pops::FieldBoundaryExecutionContext&) {}",
    ]
    installs = []
    for ordinal, (_, plan) in enumerate(dynamic):
        faces = _raw_faces(plan)
        parameter_indices = {
            handle.qualified_id: index
            for index, handle in enumerate(_parameter_handles(plan))
        }
        dependency_pack = plan.native_options["boundary_dependencies"]
        state_indices = {
            (row["qualified_id"], row["component"]): index
            for index, row in enumerate(dependency_pack["states"])
        }
        field_indices = {
            (row["qualified_id"], row["component"]): index
            for index, row in enumerate(dependency_pack["fields"])
        }
        dynamic_faces = []
        iterate_dependent = bool(plan.native_options["boundary_iterate_dependent"])
        for face, condition in enumerate(faces):
            if not plan.native_options["boundary_faces"][face]["dynamic"]:
                continue
            cpp = _ExprCpp(unknown=plan.operator.unknown,
                           parameter_indices=parameter_indices,
                           state_indices=state_indices, field_indices=field_indices)
            chunks.append(_face_struct(face, condition, symbol=face + 4 * ordinal, cpp=cpp,
                                       unknown=plan.operator.unknown))
            dynamic_faces.append((face, face + 4 * ordinal))
        residual_dispatch = "\n".join(
            "  if (face == %d) return prepare_field_boundary_residual_%d("
            "face, iterate, output, geometry, context);" % (face, symbol)
            for face, symbol in dynamic_faces)
        jvp_dispatch = "\n".join(
            "  if (face == %d) return prepare_field_boundary_jvp_%d("
            "face, iterate, direction, output, geometry, context);" % (face, symbol)
            for face, symbol in dynamic_faces)
        chunks.append(f"""
static void prepare_field_boundary_residual_route_{ordinal}(
    int face, const pops::MultiFab& iterate, pops::MultiFab& output,
    const pops::Geometry& geometry, const pops::FieldBoundaryExecutionContext& context) {{
{residual_dispatch}
}}
static void prepare_field_boundary_jvp_route_{ordinal}(
    int face, const pops::MultiFab& iterate, const pops::MultiFab& direction,
    pops::MultiFab& output, const pops::Geometry& geometry,
    const pops::FieldBoundaryExecutionContext& context) {{
{jvp_dispatch}
}}
""")
        slot = json.dumps(plan.native_options["provider_slot"])
        identity = json.dumps(plan.identity.token + ":boundary")
        jvp = ("prepare_field_boundary_jvp_route_%d" % ordinal
               if iterate_dependent else "nullptr")
        jvp_identity = json.dumps(plan.identity.token + ":boundary-jvp") if iterate_dependent else '""'
        installs.append(
            "  ctx.set_field_boundary_kernel(%s, pops::CompiledFieldBoundaryKernel{%s, %s, %s, "
            "%s, %s, field_boundary_noop, %s, %s});" %
            (slot, identity, json.dumps(plan.identity.token + ":boundary-residual"),
             jvp_identity, "prepare_field_boundary_residual_route_%d" % ordinal, jvp,
             "field_boundary_jvp_noop" if iterate_dependent else "nullptr",
             "true" if iterate_dependent else "false"))
    chunks += ["}  // namespace"]
    context_type = ("pops::runtime::program::AmrProgramContext" if target == "amr_system"
                    else "pops::runtime::program::ProgramContext")
    entry = ("pops_install_field_boundaries_amr" if target == "amr_system"
             else "pops_install_field_boundaries")
    if target == "amr_system":
        chunks.append("#include <pops/runtime/program/amr_program_context.hpp>")
    chunks += [
        'extern "C" void %s(void* sys) {' % entry,
        "  %s ctx(sys);" % context_type,
        *installs,
        "}",
    ]
    return "\n".join(chunks) + "\n"


__all__ = ["emit_field_boundaries"]
