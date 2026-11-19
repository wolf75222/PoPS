"""Lower the generated ghosts of a boundary policy to one native ghost policy (G04-a).

The SourceGhost faces of a block's resolved boundary carry the expressions of the ghost cells at
each depth (``pops.boundary.transport.SourceGhost``). This module attaches them to the emitter of
the model package and writes ``pops_generated::UserGhostPolicy``, which the prepared boundary
calls on the device for those faces (``FillSourceGhostFace``). The generated System block passes
it to every physical fill and the boundary authenticates it face by face, so a package and a
boundary that disagree on a face are refused, never filled.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# The face of the transport flux that a generated ghost needs: the source-authored face of the
# same package. The generated System block has no catalogue face route with a ghost policy.
_SOURCE_FACE = "source_face"


def _resolved_ghost_faces(numerics: Any) -> tuple[dict[int, Any], int | None]:
    """{face ordinal: SourceGhost} of the resolved boundaries of ``numerics``, and the rank."""
    from pops.boundary.transport import ResolvedTransportBoundarySet

    faces: dict[int, Any] = {}
    dimension = None
    for plan in getattr(numerics, "boundaries", ()):
        # A resolved plan holds the ghost producer plan of each boundary set, whose execution
        # authority is the resolved transport boundary.
        authority = getattr(plan, "execution_authority", plan)
        conditions = getattr(authority, "conditions", ())
        rows = [row for row in conditions if getattr(row, "source_ghost", None) is not None]
        if not rows:
            continue
        if type(authority) is not ResolvedTransportBoundarySet:
            raise TypeError("a source ghost face requires an exact resolved transport boundary")
        _, _, _, _, rank = authority._native_contract()
        if dimension is not None and rank != dimension:
            raise ValueError("source ghost faces of one package have several ranks")
        dimension = rank
        for row in rows:
            geometry = row.geometry
            ordinal = 2 * geometry.axis.index + (0 if geometry.side.value == "lower" else 1)
            if ordinal in faces:
                raise ValueError("one package cannot generate two ghosts for face %d" % ordinal)
            faces[ordinal] = row.source_ghost
    return faces, dimension


def _check_captures(emitter: Any, ghost: Any) -> None:
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children

    registry = getattr(emitter, "_param_registry", None)
    pending: list[Any] = [root for row in ghost.ghosts for root in row]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RuntimeParamRef):
            if registry is None:
                raise ValueError("generated ghost runtime read has no model parameter authority")
            handle = node.handle
            if handle is None:
                raise ValueError("generated ghost runtime capture requires its declaring Handle")
            if handle.is_instance:
                handle, _ = handle._instance_refs()
            try:
                registered = registry.handle(handle)
            except (KeyError, ValueError) as exc:
                raise ValueError("generated ghost runtime parameter belongs to another model: %s"
                                 % node.name) from exc
            if registered.param_kind != "runtime":
                raise ValueError("generated ghost capture is not a RuntimeParam")
        pending.extend(_children(node))


def prepare_user_ghost_carrier(emitter: Any, numerics: Any) -> Any:
    """Attach the generated ghosts of the resolved boundaries of ``numerics`` to the emitter."""
    if numerics is None:
        return emitter
    faces, dimension = _resolved_ghost_faces(numerics)
    if not faces:
        return emitter
    from pops.numerics.spatial import FiniteVolume

    methods = [row.method for row in numerics.rates if isinstance(row.method, FiniteVolume)]
    native = sorted({str(getattr(method.riemann, "scheme", None) or
                         getattr(method.riemann, "name", "?"))
                     for method in methods
                     if getattr(method.riemann, "scheme", None) != _SOURCE_FACE})
    if not methods or native:
        raise ValueError(
            "a source ghost face needs the generated face of its package (riemann.User), but the "
            "transport face of this block is %s; write the face of the block with riemann.User, "
            "or its boundary with a native condition" % (", ".join(native) or "absent"))
    impl = getattr(emitter, "_m", emitter)
    ghosts = list(faces.values())
    width = len(impl.cons_names)
    if any(ghost.width != width for ghost in ghosts):
        raise ValueError("generated ghost width differs from the compiled model state")
    depths = {ghost.depth for ghost in ghosts}
    if len(depths) != 1:
        raise ValueError("the generated ghosts of one package must cover one set of depths, got %s"
                         % sorted(depths))
    for ghost in ghosts:
        _check_captures(emitter, ghost)
    carrier = {"faces": dict(sorted(faces.items())), "dimension": dimension,
               "depth": depths.pop(), "reads": max(ghost.reads for ghost in ghosts),
               "width": width,
               "stencil_depth": max(method.ghost_depth for method in methods)}
    data = {"schema_version": 1, "kind": "source_ghost_policy",
            "faces": [[ordinal, ghost.source_identity] for ordinal, ghost in carrier["faces"].items()],
            "dimension": dimension, "depth": carrier["depth"], "reads": carrier["reads"],
            "width": width}
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    carrier["source_identity"] = hashlib.sha256(encoded).hexdigest()
    prior = getattr(impl, "_user_ghost", None)
    if prior is not None and prior["source_identity"] != carrier["source_identity"]:
        raise ValueError("native model generated ghosts changed across resolved boundaries")
    object.__setattr__(impl, "_user_ghost", carrier)
    return emitter


def user_ghost_source_identity(emitter: Any) -> str | None:
    carrier = getattr(getattr(emitter, "_m", emitter), "_user_ghost", None)
    return None if carrier is None else carrier["source_identity"]


def user_ghost_captures(emitter: Any) -> bool:
    carrier = getattr(getattr(emitter, "_m", emitter), "_user_ghost", None)
    return carrier is not None and any(ghost.runtime_captures
                                       for ghost in carrier["faces"].values())


def require_user_ghost_storage(emitter: Any) -> None:
    """Refuse a package whose block stores ghost cells deeper than its generated depths.

    The depth D that the elaborator derives covers the stencils of the rates and the storage
    that they ask; the composition of the halos of resolved operations and the principal groups
    are known only at compilation, and can ask more. The block would then leave the deeper ghost
    cells to no law."""
    impl = getattr(emitter, "_m", emitter)
    carrier = getattr(impl, "_user_ghost", None)
    if carrier is None:
        return
    storage = max(carrier["stencil_depth"], int(getattr(impl, "_program_state_ghost_depth", 1)))
    if storage > carrier["depth"]:
        raise ValueError(
            "the generated ghosts of this package fill the ghost depths 1..%d, but its block "
            "stores %d ghost cells (the halos of its program operations); G04 generates the "
            "deeper ghosts in a later lot" % (carrier["depth"], storage))


def emit_user_ghost_policy(emitter: Any) -> str:
    from pops.codegen.cpp_writer import _cse_emit

    impl = getattr(emitter, "_m", emitter)
    carrier = getattr(impl, "_user_ghost", None)
    if carrier is None:
        return ""
    require_user_ghost_storage(emitter)
    captures = user_ghost_captures(emitter)
    if captures:
        impl.assign_runtime_indices()
    width, depth, faces = carrier["width"], carrier["depth"], carrier["faces"]
    face_count = 2 * carrier["dimension"]
    bindings = {"pops_ghost_inner_%d_%d" % (j, component): "inner(%d, %d)" % (j, component)
                for j in range(1, depth + 1) for component in range(width)}
    generated = " || ".join("ordinal == %d" % ordinal for ordinal in faces) or "false"
    lines = [
        "namespace pops_generated {",
        "struct UserGhostPolicy {",
        "  static constexpr bool present = true;",
        '  static constexpr const char* source_identity = "%s";' % carrier["source_identity"],
        "  static constexpr int width = %d;" % width,
        "  static constexpr int depth = %d;" % depth,
        "  static constexpr int reads = %d;" % carrier["reads"],
        "  static constexpr int face_count = %d;" % face_count,
        "  static constexpr bool face_generated(int ordinal) { return %s; }" % generated,
        "  static constexpr const char* face_identity(int ordinal) {",
        "    switch (ordinal) {",
        *('      case %d: return "%s";' % (ordinal, ghost.source_identity)
          for ordinal, ghost in faces.items()),
        "      default: return \"\";",
        "    }",
        "  }",
        *(["  using pops_device_capture_members = pops::DeviceCaptureMembers<pops::RuntimeParams>;",
           "  pops::RuntimeParams params{};"] if captures else []),
        "  template <int Ordinal, class Inner>",
        "  POPS_HD void value(int k, const Inner& inner, pops::Real* ghost) const {",
        "    (void)k;",
        "    (void)inner;",
    ]
    for ordinal, ghost in faces.items():
        lines.append("    if constexpr (Ordinal == %d) {" % ordinal)
        for k, row in enumerate(ghost.ghosts, start=1):
            declarations, rendered = _cse_emit(list(row), "pops::Real", "        ",
                                               scalar_bindings=bindings)
            lines.append("      if (k == %d) {" % k)
            lines.extend(declarations)
            lines.extend("        ghost[%d] = %s;" % (component, value)
                         for component, value in enumerate(rendered))
            lines.append("        return;")
            lines.append("      }")
        lines.append("    }")
    lines += [
        "    // A depth or a face that the policy does not generate: the face kernel refuses NaN.",
        "    for (int component = 0; component < width; ++component)",
        "      ghost[component] = std::numeric_limits<pops::Real>::quiet_NaN();",
        "  }",
        "};",
        "}  // namespace pops_generated",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["emit_user_ghost_policy", "prepare_user_ghost_carrier", "require_user_ghost_storage",
           "user_ghost_captures", "user_ghost_source_identity"]
