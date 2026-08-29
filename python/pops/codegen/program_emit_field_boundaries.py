"""Generated exact-ranked device launchers for dynamic field boundary laws.

The authoring Expr graph is lowered once into named C++ functors. Runtime iteration sees direct
function pointers and POD captures only: no Python callback, string lookup, virtual dispatch or host
expression interpreter occurs in a face-cell loop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _face_ordinal(axis: int, side: str, dimension: int) -> int:
    if type(axis) is not int or axis < 0 or axis >= dimension or side not in ("lo", "hi"):
        raise ValueError("dynamic field boundary selector is outside its exact Cartesian rank")
    return 2 * axis + (side == "hi")


def _raw_faces(plan: Any, dimension: int) -> tuple[Any, ...]:
    """Return the validated condition object for every exact-ranked physical face."""
    from pops.fields.bcs import AllPhysicalBoundaries, AxisBoundary

    faces = [None] * (2 * dimension)
    for binding in plan.discretization.boundaries:
        selector = binding.selector
        if isinstance(selector, AllPhysicalBoundaries):
            selected = range(len(faces))
        elif isinstance(selector, AxisBoundary):
            selected = (_face_ordinal(selector.axis, selector.side, dimension),)
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
    from pops._ir.expr import Expr, _wrap
    from pops._ir.handle_expr import ValueExpr
    from pops.model import Handle

    if isinstance(value, Handle):
        return ValueExpr(value)
    return value if isinstance(value, Expr) else _wrap(value)


class _ExprCpp:
    def __init__(
        self,
        *,
        unknown: Any,
        parameter_indices: Mapping[str, int],
        state_indices: Mapping[tuple[str, int], int],
        field_indices: Mapping[tuple[str, int], int],
    ) -> None:
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
                "dynamic boundary RuntimeParamRef requires an owner-qualified ParamHandle"
            )
        try:
            index = self.parameter_indices[handle.qualified_id]
        except KeyError:
            raise ValueError(
                "dynamic boundary parameter %s is absent from its resolved dependency pack"
                % handle.qualified_id
            ) from None
        self.used_parameters.add(index)
        return index

    def emit(self, value: Any) -> str:
        from pops._ir.expr import Abs, Const, Neg, Pow, Sign, Sqrt, Var, _Bin
        from pops._ir.handle_expr import ValueExpr
        from pops._ir.values import RuntimeParamRef
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
            handle_kind = getattr(value.handle, "kind", None)
            indices = self.state_indices if handle_kind == "state" else self.field_indices
            try:
                index = indices[key]
            except KeyError:
                raise ValueError(
                    "boundary value %s[%d] is absent from its resolved direct-buffer pack" % key
                ) from None
            if handle_kind == "state":
                self.used_states.add(index)
                return "state%d(index, %d)" % (index, value.component)
            self.used_fields.add(index)
            return "field%d(index, %d)" % (index, value.component)
        if isinstance(value, LogicalTimeValue):
            self.used_times.add(value.coordinate)
            return "logical_%s" % value.coordinate
        if isinstance(value, ValueExpr):
            if getattr(value.handle, "kind", None) == "parameter":
                proxy = type(
                    "BoundaryParamRef",
                    (),
                    {"handle": value.handle, "name": value.handle.local_id},
                )()
                return "p%d" % self._param_index(proxy)
            if value.handle.qualified_id != self.unknown.qualified_id:
                raise NotImplementedError(
                    "dynamic boundary value dependency %s is not routed to a prepared "
                    "state/field buffer" % value.handle.qualified_id
                )
            return "u"
        if isinstance(value, Var):
            raise ValueError(
                "dynamic boundary expression contains unqualified Var(%r); use a typed Handle"
                % value.name
            )
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
        raise TypeError(
            "dynamic boundary expression node %s has no C++ lowering" % type(value).__name__
        )


def _parameter_handles(plan: Any) -> tuple[Any, ...]:
    return plan.provider_parameter_handles("boundary-kernel")


def _face_struct(face: int, condition: Any, *, symbol: int, cpp: _ExprCpp) -> str:
    from pops._ir.lowering import diff

    alpha, beta, value = (_as_expr(item) for item in _coefficients(condition))
    a = cpp.emit(alpha)
    b = cpp.emit(beta)
    v = cpp.emit(value)
    da = cpp.emit(diff(alpha, cpp.unknown))
    db = cpp.emit(diff(beta, cpp.unknown))
    dv = cpp.emit(diff(value, cpp.unknown))
    params = "\n".join("  pops::Real p%d;" % index for index in sorted(cpp.used_parameters))
    if params:
        params += "\n"
    dependencies = "\n".join(
        [
            "  pops::FieldView<const pops::Real, pops::kNativeDimension> state%d;" % index
            for index in sorted(cpp.used_states)
        ]
        + [
            "  pops::FieldView<const pops::Real, pops::kNativeDimension> field%d;" % index
            for index in sorted(cpp.used_fields)
        ]
        + ["  pops::Real logical_%s;" % name for name in sorted(cpp.used_times)]
    )
    if dependencies:
        dependencies += "\n"
    axis = face // 2
    upper = bool(face % 2)
    bound = "geometry.domain().hi[%d]" % axis if upper else "geometry.domain().lo[%d]" % axis
    touch = "box.hi[%d] == %s" % (axis, bound) if upper else "box.lo[%d] == %s" % (axis, bound)
    mirror = (
        "2 * geometry.domain().hi[%d] - index[%d] + 1" % (axis, axis)
        if upper
        else "2 * geometry.domain().lo[%d] - index[%d] - 1" % (axis, axis)
    )
    layer = (
        "index[%d] - geometry.domain().hi[%d]" % (axis, axis)
        if upper
        else "geometry.domain().lo[%d] - index[%d]" % (axis, axis)
    )
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
            "  if (context.states == nullptr || context.state_distributions == nullptr || "
            "context.state_count <= %d)\n"
            '    throw std::runtime_error("dynamic field boundary state carrier is incomplete");'
            % max(cpp.used_states)
        )
    if cpp.used_fields:
        dependency_checks.append(
            "  if (context.fields == nullptr || context.field_distributions == nullptr || "
            "context.field_count <= %d)\n"
            '    throw std::runtime_error("dynamic field boundary field carrier is incomplete");'
            % max(cpp.used_fields)
        )
    dependency_setup = "\n".join(dependency_checks)

    def law_args(local_index: str) -> str:
        args = ["iterate.fab(%s).view()" % local_index]
        args.extend(
            "context.states[%d]->fab(context.states[%d]->local_index_of("
            "iterate.global_index(%s))).view()" % (index, index, local_index)
            for index in sorted(cpp.used_states)
        )
        args.extend(
            "context.fields[%d]->fab(context.fields[%d]->local_index_of("
            "iterate.global_index(%s))).view()" % (index, index, local_index)
            for index in sorted(cpp.used_fields)
        )
        point_names = {
            "time": "time",
            "dt": "dt",
            "step": "step",
            "substep": "substep",
            "iteration": "iteration",
            "stage": "stage_slot",
            "partition": "partition_slot",
        }
        args.extend(
            "static_cast<pops::Real>(context.point.%s)" % point_names[name]
            for name in sorted(cpp.used_times)
        )
        args.extend("params[%d]" % index for index in sorted(cpp.used_parameters))
        return ", ".join(args)

    return f"""
struct FieldBoundaryFace{symbol} {{
  pops::FieldView<const pops::Real, pops::kNativeDimension> iterate;
{dependencies}{params}  POPS_HD pops::Real alpha(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {a};
  }}
  POPS_HD pops::Real beta(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {b};
  }}
  POPS_HD pops::Real value(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {v};
  }}
  POPS_HD pops::Real dalpha(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {da};
  }}
  POPS_HD pops::Real dbeta(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {db};
  }}
  POPS_HD pops::Real dvalue(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real u = iterate(index); return {dv};
  }}
  POPS_HD pops::Real denominator(
      const pops::CellIndex<pops::kNativeDimension>& index, pops::Real distance) const {{
    return alpha(index) / pops::Real(2) + beta(index) / distance;
  }}
}};

struct FieldBoundaryValidate{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::Geometry<pops::kNativeDimension> geometry;
  pops::Real domain_points;
  POPS_HD pops::Real operator()(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    const pops::Real aa = law.alpha(index);
    const pops::Real bb = law.beta(index);
    const pops::Real vv = law.value(index);
    const pops::Real h = geometry.spacing({axis});
    const pops::Real denom = aa / pops::Real(2) + bb / h;
    const pops::Real scale = std::fmax(
        pops::Real(1), std::fmax(std::fabs(aa / pops::Real(2)), std::fabs(bb / h)));
    const bool invalid = !std::isfinite(aa) || !std::isfinite(bb) || !std::isfinite(vv) ||
        !std::isfinite(denom) || std::fabs(denom) <=
        pops::Real(64) * std::numeric_limits<pops::Real>::epsilon() * scale;
    if (!invalid) return pops::Real(0);
    pops::Real linear = pops::Real(0);
    pops::Real stride = pops::Real(1);
    for (int ranked_axis = 0; ranked_axis < pops::kNativeDimension; ++ranked_axis) {{
      linear += pops::Real(index[ranked_axis] - geometry.domain().lo[ranked_axis]) * stride;
      stride *= pops::Real(geometry.domain().length(ranked_axis));
    }}
    return domain_points - linear;
  }}
}};

struct FieldBoundaryResidualGhost{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::FieldView<pops::Real, pops::kNativeDimension> output;
  pops::Geometry<pops::kNativeDimension> geometry;
  POPS_HD void operator()(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    pops::CellIndex<pops::kNativeDimension> boundary = index;
    boundary[{axis}] = {bound};
    pops::CellIndex<pops::kNativeDimension> mirrored = index;
    mirrored[{axis}] = {mirror};
    const pops::Real distance = pops::Real(2 * ({layer}) - 1) * geometry.spacing({axis});
    const pops::Real aa = law.alpha(boundary);
    const pops::Real bb = law.beta(boundary);
    const pops::Real vv = law.value(boundary);
    const pops::Real inner = law.iterate(mirrored);
    output(index) = (vv - inner * (aa / pops::Real(2) - bb / distance)) /
                    (aa / pops::Real(2) + bb / distance);
  }}
}};

struct FieldBoundaryJvpGhost{symbol} {{
  FieldBoundaryFace{symbol} law;
  pops::FieldView<const pops::Real, pops::kNativeDimension> direction;
  pops::FieldView<pops::Real, pops::kNativeDimension> output;
  pops::Geometry<pops::kNativeDimension> geometry;
  POPS_HD void operator()(
      const pops::CellIndex<pops::kNativeDimension>& index) const {{
    pops::CellIndex<pops::kNativeDimension> boundary = index;
    boundary[{axis}] = {bound};
    pops::CellIndex<pops::kNativeDimension> mirrored = index;
    mirrored[{axis}] = {mirror};
    const pops::Real distance = pops::Real(2 * ({layer}) - 1) * geometry.spacing({axis});
    const pops::Real aa = law.alpha(boundary), bb = law.beta(boundary);
    const pops::Real vv = law.value(boundary);
    const pops::Real daa = law.dalpha(boundary), dbb = law.dbeta(boundary);
    const pops::Real dvv = law.dvalue(boundary);
    const pops::Real inner = law.iterate(mirrored);
    const pops::Real dinner = direction(mirrored);
    const pops::Real du = direction(boundary);
    const pops::Real denom = aa / pops::Real(2) + bb / distance;
    const pops::Real numer = vv - inner * (aa / pops::Real(2) - bb / distance);
    const pops::Real ddenom = du * (daa / pops::Real(2) + dbb / distance);
    const pops::Real dnumer = dvv * du - dinner * (aa / pops::Real(2) - bb / distance) -
        inner * du * (daa / pops::Real(2) - dbb / distance);
    output(index) = (dnumer * denom - numer * ddenom) / (denom * denom);
  }}
}};

static void prepare_field_boundary_residual_{symbol}(
    int requested_face, const pops::MultiFab<pops::kNativeDimension>& iterate,
    pops::MultiFab<pops::kNativeDimension>& output,
    const pops::Geometry<pops::kNativeDimension>& geometry,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>& context) {{
  if (requested_face != {face}) return;
  if (context.failure == nullptr)
    throw std::runtime_error("dynamic field boundary has no fallible execution channel");
  if (context.failure->failed()) return;
{parameter_setup.rstrip()}
{dependency_setup}
  const std::int64_t domain_points = geometry.domain().numPts();
  const double exact_limit = std::ldexp(1.0, std::numeric_limits<pops::Real>::digits);
  if (static_cast<double>(domain_points) > exact_limit)
    throw std::overflow_error("dynamic field boundary witness exceeds exact Real indexing");
  pops::Real best = pops::Real(0);
  std::size_t best_li = pops::MultiFab<pops::kNativeDimension>::not_local;
  for (std::size_t li = 0; li < iterate.local_size(); ++li) {{
    const pops::Box<pops::kNativeDimension> box = iterate.box(li);
    if (!({touch})) continue;
    pops::Box<pops::kNativeDimension> validation = box;
    validation.lo[{axis}] = {bound};
    validation.hi[{axis}] = {bound};
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    const pops::Real encoded = pops::for_each_cell_reduce_max(
        validation, FieldBoundaryValidate{symbol}{{
            law, geometry, static_cast<pops::Real>(domain_points)}});
    if (encoded > best) {{ best = encoded; best_li = li; }}
  }}
  if (best > pops::Real(0)) {{
    std::int64_t linear = domain_points - static_cast<std::int64_t>(best);
    pops::CellIndex<pops::kNativeDimension> failed{{}};
    for (int ranked_axis = 0; ranked_axis < pops::kNativeDimension; ++ranked_axis) {{
      const std::int64_t extent = geometry.domain().length(ranked_axis);
      failed[ranked_axis] = geometry.domain().lo[ranked_axis] +
                            static_cast<int>(linear % extent);
      linear /= extent;
    }}
    const FieldBoundaryFace{symbol} law{{{law_args("best_li")}}};
    context.failure->code = 1;
    context.failure->face = {face};
    context.failure->cell = failed;
    context.failure->value = law.denominator(failed, geometry.spacing({axis}));
    return;
  }}
  const int ng = static_cast<int>(output.ghosts()[{axis}]);
  if (ng <= 0)
    throw std::runtime_error("dynamic field boundary requires a normal ghost layer");
  for (std::size_t li = 0; li < output.local_size(); ++li) {{
    const pops::Box<pops::kNativeDimension> box = output.box(li);
    if (!({touch})) continue;
    pops::Box<pops::kNativeDimension> ghost = box.grow({axis}, ng);
    ghost.lo[{axis}] = {"geometry.domain().hi[%d] + 1" % axis if upper else "geometry.domain().lo[%d] - ng" % axis};
    ghost.hi[{axis}] = {"geometry.domain().hi[%d] + ng" % axis if upper else "geometry.domain().lo[%d] - 1" % axis};
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    pops::for_each_cell(ghost, FieldBoundaryResidualGhost{symbol}{{
        law, output.fab(li).view(), geometry}});
  }}
}}

static void prepare_field_boundary_jvp_{symbol}(
    int requested_face, const pops::MultiFab<pops::kNativeDimension>& iterate,
    const pops::MultiFab<pops::kNativeDimension>& direction,
    pops::MultiFab<pops::kNativeDimension>& output,
    const pops::Geometry<pops::kNativeDimension>& geometry,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>& context) {{
  if (requested_face != {face}) return;
  if (context.failure == nullptr)
    throw std::runtime_error("dynamic field boundary JVP has no fallible execution channel");
  if (context.failure->failed()) return;
{parameter_setup.rstrip()}
{dependency_setup}
  const int ng = static_cast<int>(output.ghosts()[{axis}]);
  if (ng <= 0)
    throw std::runtime_error("dynamic field boundary JVP requires a normal ghost layer");
  for (std::size_t li = 0; li < output.local_size(); ++li) {{
    const pops::Box<pops::kNativeDimension> box = output.box(li);
    if (!({touch})) continue;
    pops::Box<pops::kNativeDimension> ghost = box.grow({axis}, ng);
    ghost.lo[{axis}] = {"geometry.domain().hi[%d] + 1" % axis if upper else "geometry.domain().lo[%d] - ng" % axis};
    ghost.hi[{axis}] = {"geometry.domain().hi[%d] + ng" % axis if upper else "geometry.domain().lo[%d] - 1" % axis};
    const FieldBoundaryFace{symbol} law{{{law_args("li")}}};
    pops::for_each_cell(ghost, FieldBoundaryJvpGhost{symbol}{{
        law, direction.fab(li).view(), output.fab(li).view(), geometry}});
  }}
}}
"""


def emit_field_boundaries(program: Any, authority: Any, field_plans: Any, target: str) -> str:
    """Emit exact-ranked launchers plus one preparation-image registration helper.

    The helper accepts the already image-owned ``ProgramExecutionServices`` selected by the sole
    ABI-v5 candidate prepare callback.  It never receives a ``System``/``AmrSystem`` facade and
    never constructs a second provider or installer.
    """
    del program, authority
    dynamic = [
        (name, plan)
        for name, plan in sorted((field_plans or {}).items())
        if plan.native_options.get("boundary_kernel_required")
    ]
    if not dynamic:
        return ""
    dimensions = {plan.native_options["output_route"]["dimension"] for _, plan in dynamic}
    if len(dimensions) != 1:
        raise ValueError("dynamic field boundaries disagree on the resolved native rank")
    dimension = next(iter(dimensions))
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise TypeError("dynamic field boundaries require one exact native rank")
    exact = "pops::kNativeDimension"
    chunks = [
        "// Generated exact-ranked dynamic field boundary residual/JVP launchers.",
        "#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>",
        "#include <pops/mesh/execution/for_each.hpp>",
        "#include <cmath>",
        "#include <cstdint>",
        "#include <limits>",
        "#include <stdexcept>",
        "#include <utility>",
        "static_assert(pops::kNativeDimension == %d);" % dimension,
        "namespace {",
        "static void field_boundary_noop(int, const pops::MultiFab<%s>&, "
        "pops::MultiFab<%s>&, const pops::Geometry<%s>&, "
        "const pops::FieldBoundaryExecutionContext<%s>&) {}" % (exact, exact, exact, exact),
        "static void field_boundary_jvp_noop(int, const pops::MultiFab<%s>&, "
        "const pops::MultiFab<%s>&, pops::MultiFab<%s>&, const pops::Geometry<%s>&, "
        "const pops::FieldBoundaryExecutionContext<%s>&) {}" % (exact, exact, exact, exact, exact),
    ]
    installs = []
    symbol_stride = 2 * dimension
    for ordinal, (_, plan) in enumerate(dynamic):
        faces = _raw_faces(plan, dimension)
        boundary_faces = tuple(plan.native_options["boundary_faces"])
        if len(boundary_faces) != symbol_stride:
            raise ValueError("dynamic field boundary facts lost an exact-ranked face")
        parameter_indices = {
            handle.qualified_id: index for index, handle in enumerate(_parameter_handles(plan))
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
            if not boundary_faces[face]["dynamic"]:
                continue
            cpp = _ExprCpp(
                unknown=plan.operator.unknown,
                parameter_indices=parameter_indices,
                state_indices=state_indices,
                field_indices=field_indices,
            )
            symbol = face + symbol_stride * ordinal
            chunks.append(_face_struct(face, condition, symbol=symbol, cpp=cpp))
            dynamic_faces.append((face, symbol))
        residual_dispatch = "\n".join(
            "  if (face == %d) return prepare_field_boundary_residual_%d("
            "face, iterate, output, geometry, context);" % (face, symbol)
            for face, symbol in dynamic_faces
        )
        jvp_dispatch = "\n".join(
            "  if (face == %d) return prepare_field_boundary_jvp_%d("
            "face, iterate, direction, output, geometry, context);" % (face, symbol)
            for face, symbol in dynamic_faces
        )
        chunks.append(
            f"""
static void prepare_field_boundary_residual_route_{ordinal}(
    int face, const pops::MultiFab<{exact}>& iterate, pops::MultiFab<{exact}>& output,
    const pops::Geometry<{exact}>& geometry,
    const pops::FieldBoundaryExecutionContext<{exact}>& context) {{
{residual_dispatch}
}}
static void prepare_field_boundary_jvp_route_{ordinal}(
    int face, const pops::MultiFab<{exact}>& iterate,
    const pops::MultiFab<{exact}>& direction, pops::MultiFab<{exact}>& output,
    const pops::Geometry<{exact}>& geometry,
    const pops::FieldBoundaryExecutionContext<{exact}>& context) {{
{jvp_dispatch}
}}
"""
        )
        slot = json.dumps(plan.native_options["provider_slot"])
        identity = json.dumps(plan.identity.token + ":boundary")
        # Every dynamic Robin law is affine in the unknown only after its coefficients have been
        # evaluated.  Krylov therefore always needs the exact homogeneous/JVP ghost launcher, even
        # when those coefficients depend only on time, parameters or another prepared field.
        jvp = "prepare_field_boundary_jvp_route_%d" % ordinal
        jvp_identity = json.dumps(plan.identity.token + ":boundary-jvp")
        installs.append(
            "  ctx.set_field_boundary_kernel(%s, pops::CompiledFieldBoundaryKernel<%s>{"
            "%s, %s, %s, %s, %s, field_boundary_noop, %s, %s});"
            % (
                slot,
                exact,
                identity,
                json.dumps(plan.identity.token + ":boundary-residual"),
                jvp_identity,
                "prepare_field_boundary_residual_route_%d" % ordinal,
                jvp,
                "field_boundary_jvp_noop",
                "true" if iterate_dependent else "false",
            )
        )
    chunks += ["}  // namespace"]
    if target not in {"system", "amr_system"}:
        raise ValueError("dynamic field boundaries require one exact runtime target")
    entry = "program_candidate_prepare_field_boundaries"
    chunks.append("#include <pops/runtime/program/program_execution_services.hpp>")
    chunks += [
        "static void %s(" % entry,
        "    pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>& ctx) {",
        *installs,
        "}",
    ]
    return "\n".join(chunks) + "\n"


__all__ = ["emit_field_boundaries"]
