"""Lower an authenticated Python face formula to one native NumericalFlux policy."""
from __future__ import annotations

from typing import Any


def prepare_user_face_carrier(emitter: Any, numerics: Any) -> Any:
    if numerics is None:
        return emitter
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children
    from pops.numerics.riemann.user import authenticated_user_face
    from pops.numerics.spatial import FiniteVolume

    selected = []
    for row in numerics.rates:
        method = row.method
        if not isinstance(method, FiniteVolume) or \
                getattr(method.riemann, "scheme", None) != "source_face":
            continue
        descriptor = authenticated_user_face(method.riemann)
        if descriptor.options["state"] != method.variables.options.get("state"):
            raise ValueError("user face state differs from the selected finite-volume state")
        selected.append(descriptor)
    if not selected:
        return emitter
    identities = {item.options["source_identity"] for item in selected}
    if len(identities) != 1:
        raise ValueError("one native model package cannot install different user face bodies")
    impl = getattr(emitter, "_m", emitter)
    if any(item.options["width"] != len(impl.cons_names) for item in selected):
        raise ValueError("user face width differs from the compiled model state")
    registry = getattr(emitter, "_param_registry", None)
    for descriptor in selected:
        pending, seen = list(descriptor.expression), set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, RuntimeParamRef):
                if registry is None:
                    raise ValueError("user face runtime read has no model parameter authority")
                handle = node.handle.declaration_ref if node.handle.is_instance else node.handle
                try:
                    registered = registry.handle(handle)
                except (KeyError, ValueError) as exc:
                    raise ValueError("user face runtime parameter belongs to another model: %s"
                                     % node.name) from exc
                if registered.param_kind != "runtime":
                    raise ValueError("user face capture is not a RuntimeParam")
            pending.extend(_children(node))
    prior = getattr(impl, "_user_face", None)
    if prior is not None and prior.options["source_identity"] not in identities:
        raise ValueError("native model user face body changed across resolved rates")
    object.__setattr__(impl, "_user_face", selected[0])
    return emitter


def user_face_source_identity(emitter: Any) -> str | None:
    impl = getattr(emitter, "_m", emitter)
    descriptor = getattr(impl, "_user_face", None)
    if descriptor is None:
        return None
    from pops.numerics.riemann.user import authenticated_user_face

    return authenticated_user_face(descriptor).options["source_identity"]


def emit_user_face_policy(emitter: Any) -> str:
    from pops.codegen.cpp_writer import _cse_emit
    from pops.numerics.riemann.user import authenticated_user_face

    impl = getattr(emitter, "_m", emitter)
    descriptor = getattr(impl, "_user_face", None)
    if descriptor is None:
        return ""
    descriptor = authenticated_user_face(descriptor)
    options = descriptor.options
    if options["runtime_captures"]:
        impl.assign_runtime_indices()
    width = options["width"]
    bindings = {"pops_face_speed": "bound.value"}
    for index in range(width):
        bindings.update({
            "pops_face_left_%d" % index: "left.state[%d]" % index,
            "pops_face_right_%d" % index: "right.state[%d]" % index,
            "pops_face_flux_left_%d" % index: "left_density.value[%d]" % index,
            "pops_face_flux_right_%d" % index: "right_density.value[%d]" % index,
        })
    expression_lines, rendered, observed = _cse_emit(
        descriptor.expression, "pops::Real", "    ", materialize_all=True,
        return_names=True, scalar_bindings=bindings)
    finite = " && ".join([*("std::isfinite(%s)" % item for item in observed),
                          *("std::isfinite(density[%d])" % index for index in range(width))])
    lines = [
        "namespace pops_generated {",
        "struct UserFacePolicy {",
        '  static constexpr const char* source_identity = "%s";' % options["source_identity"],
        *(["  pops::RuntimeParams params{};"] if options["runtime_captures"] else []),
        "  template <pops::OrdinaryPhysicalFlux Physical>",
        "  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(",
        "      const Physical& physical, const typename Physical::Trace& left,",
        "      const typename Physical::Trace& right, const pops::FaceContext& face) const {",
        "    using State = typename Physical::State;",
        "    if (face.orientation == pops::FaceOrientation::kNegative)",
        "      return pops::detail::canonical_evaluation(*this, physical, left, right, face);",
        "    pops::StabilityBound bound{};",
        "    if (!pops::detail::max_normal_stability_bound(",
        "            physical.stability(left, face), physical.stability(right, face), bound))",
        "      return pops::FluxEvaluation<State>::reject(",
        "          pops::RiemannFailureCause::kUserInvalidStability);",
        "    const auto left_density = physical.evaluate(left, face);",
        "    const auto right_density = physical.evaluate(right, face);",
        "    auto physical_failure = pops::FluxEvaluation<State>::failed(0);",
        "    if (pops::detail::physical_pair_failed(",
        "            left_density, right_density, physical_failure))",
        "      return physical_failure;",
        "    State density{};",
        *expression_lines,
        *("    density[%d] = %s;" % (index, value)
          for index, value in enumerate(rendered[:width])),
        "    const pops::Real numerical_speed = %s;" % rendered[width],
        "    if (!std::isfinite(numerical_speed) || numerical_speed < pops::Real(0))",
        "      return pops::FluxEvaluation<State>::reject(",
        "          pops::RiemannFailureCause::kUserInvalidStability);",
        "    if (!(%s))" % finite,
        "      return pops::FluxEvaluation<State>::reject(",
        "          pops::RiemannFailureCause::kUserNonFiniteFlux);",
        "    return pops::FluxEvaluation<State>::ok(",
        "        density, pops::StabilityBound{numerical_speed, pops::StabilityUnit::kLengthPerTime,",
        "                                      pops::StabilityConvention::kNormalSpectralRadius});",
        "  }",
        "};",
        "}  // namespace pops_generated",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["prepare_user_face_carrier", "user_face_source_identity", "emit_user_face_policy"]
