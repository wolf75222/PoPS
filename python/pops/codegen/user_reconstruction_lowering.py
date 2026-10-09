"""Bind one source-authored reconstruction to its compiled native model package."""
from __future__ import annotations

from typing import Any


def prepare_user_reconstruction_carrier(emitter: Any, numerics: Any) -> Any:
    """Retain an authenticated method body on the private compiler view.

    A generated model package has one reconstruction policy type. Several rate
    occurrences may share it, but different bodies require distinct packages.
    """
    if numerics is None:
        return emitter
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    selected = []
    owners = []
    for row in numerics.rates:
        method = row.method
        if not isinstance(method, FiniteVolume) or method.sampling:
            continue
        descriptor = method.reconstruction
        if getattr(descriptor, "scheme", None) == "source_stencil":
            authored = authenticated_user_reconstruction(descriptor)
            if authored.capabilities.get("vector_row"):
                if authored.options["state"] != method.variables.options.get("state") or authored.options["sampling"]:
                    raise ValueError("joint reconstruction inputs differ from the selected finite-volume state")
            selected.append(authored)
            owners.append(method.variables.options.get("state"))
    if not selected:
        return emitter
    identities = {item.options["source_identity"] for item in selected}
    if len(identities) != 1:
        raise ValueError("one native model package cannot install different user reconstructions")
    impl = getattr(emitter, "_m", emitter)
    registry = getattr(emitter, "_param_registry", None)
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children
    for descriptor, owner in zip(selected, owners, strict=True):
        if descriptor.capabilities.get("vector_row") and descriptor.options["component_counts"][0] != len(impl.cons_names):
            raise ValueError("joint reconstruction width differs from the compiled model state")
        pending, seen = list(descriptor.expression) if isinstance(descriptor.expression, tuple) else [descriptor.expression], set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, RuntimeParamRef):
                if node.handle.is_instance and (owner is None or
                        node.handle.block_ref._resolved() != owner.block_ref._resolved()):
                    raise ValueError("user reconstruction runtime capture belongs to another block instance")
                handle = node.handle.declaration_ref if node.handle.is_instance else node.handle
                if registry is None:
                    raise ValueError("user reconstruction runtime read has no model parameter authority")
                try:
                    registered = registry.handle(handle)
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        "user reconstruction runtime parameter belongs to another model: %s"
                        % node.name) from exc
                if registered.param_kind != "runtime":
                    raise ValueError("user reconstruction capture is not a RuntimeParam")
            pending.extend(_children(node))
    prior = getattr(impl, "_user_reconstruction", None)
    if prior is not None and prior.options["source_identity"] not in identities:
        raise ValueError("native model reconstruction body changed across resolved rates")
    object.__setattr__(impl, "_user_reconstruction", selected[0])
    return emitter


def emit_user_reconstruction_policy(emitter: Any) -> str:
    """Emit the exact scalar policy selected by resolved numerical authority."""
    from pops.codegen.cpp_writer import _cse_emit
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    impl = getattr(emitter, "_m", emitter)
    descriptor = getattr(impl, "_user_reconstruction", None)
    if descriptor is None:
        return ""
    descriptor = authenticated_user_reconstruction(descriptor)
    options = descriptor.options
    if options["runtime_captures"]:
        impl.assign_runtime_indices()
    joint = descriptor.capabilities.get("vector_row", False)
    if joint:
        return _emit_joint_policy(impl, descriptor)
    sample_bindings = {}
    for offset in options["sample_offsets"]:
        spelling = "m%d" % -offset if offset < 0 else "p%d" % offset
        sample_bindings["pops_recon_sample_%s" % spelling] = "sample(%d)" % offset
    expression_lines, rendered, observed = _cse_emit(
        [descriptor.expression], "pops::Real", "    ",
        materialize_all=True, return_names=True, scalar_bindings=sample_bindings)
    finite = " && ".join(
        [*("std::isfinite(%s)" % item for item in observed),
         "std::isfinite(%s)" % rendered[0]])
    lines = [
        "namespace pops_generated {",
        "struct UserReconstructionPolicy {",
        *(["  pops::RuntimeParams params{};"] if options["runtime_captures"] else []),
        "  static constexpr int formal_order = %d;" % options["formal_order"],
        "  static constexpr int n_ghost = %d;" % options["ghost_depth"],
        "  static constexpr int stencil_min_offset = %d;" % options["stencil_min_offset"],
        "  static constexpr int stencil_max_offset = %d;" % options["stencil_max_offset"],
        '  static constexpr const char* source_identity = "%s";' % options["source_identity"],
        "  template <class Sample>",
        "  POPS_HD pops::Real stencil_face_value(const Sample& sample) const {",
        *expression_lines,
        "    return (%s) ? (%s) : std::numeric_limits<pops::Real>::quiet_NaN();"
        % (finite, rendered[0]),
        "  }",
        "};",
        "static_assert(pops::ReconstructionPolicy<UserReconstructionPolicy>);",
        "static_assert(pops::stencil_envelope_fits_storage<UserReconstructionPolicy>);",
        "}  // namespace pops_generated",
    ]
    return "\n".join(lines) + "\n"


def user_reconstruction_source_identity(emitter: Any) -> str | None:
    impl = getattr(emitter, "_m", emitter)
    descriptor = getattr(impl, "_user_reconstruction", None)
    if descriptor is None:
        return None
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    return authenticated_user_reconstruction(descriptor).options["source_identity"]


def _emit_joint_policy(impl, descriptor):
    from pops.codegen.cpp_writer import _cse_emit

    options = descriptor.options
    counts = options["component_counts"]
    starts, cursor = [], 0
    for count in counts:
        starts.append(cursor)
        cursor += count
    bindings = {}
    for offset, row, component in options["sample_reads"]:
        spelling = "m%d" % -offset if offset < 0 else "p%d" % offset
        bindings["pops_recon_joint_%s_r%d_c%d" % (spelling, row, component)] = "sample(%d, %d)" % (
            offset,
            starts[row] + component,
        )
    expression_lines, rendered, observed = _cse_emit(
        descriptor.expression,
        "pops::Real",
        "    ",
        materialize_all=True,
        return_names=True,
        scalar_bindings=bindings,
    )
    finite = " && ".join("std::isfinite(%s)" % value for value in (*observed, *rendered))
    lines = [
        "namespace pops_generated {",
        "struct UserReconstructionPolicy {",
        *(["  pops::RuntimeParams params{};"] if options["runtime_captures"] else []),
        "  static constexpr int n_components = %d;" % counts[0],
        "  static constexpr int formal_order = %d;" % options["formal_order"],
        "  static constexpr int n_ghost = %d;" % options["ghost_depth"],
        "  static constexpr int stencil_min_offset = %d;" % options["stencil_min_offset"],
        "  static constexpr int stencil_max_offset = %d;" % options["stencil_max_offset"],
        '  static constexpr const char* source_identity = "%s";' % options["source_identity"],
        "  template<class Sample>",
        "  POPS_HD std::array<pops::Real,n_components> stencil_face_state(const Sample& sample) const {",
        *expression_lines,
        "    if (!(%s)) {" % finite,
        "      std::array<pops::Real,n_components> invalid{};",
        "      for (auto& value : invalid) value = std::numeric_limits<pops::Real>::quiet_NaN();",
        "      return invalid;",
        "    }",
        "    return {{%s}};" % ", ".join(rendered),
        "  }",
        "};",
        "static_assert(pops::ReconstructionPolicy<UserReconstructionPolicy>);",
        "static_assert(pops::stencil_envelope_fits_storage<UserReconstructionPolicy>);",
        "} // namespace pops_generated",
    ]
    return "\n".join(lines) + "\n"
