"""Resolve-time capability proof for shared conservative block interfaces."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

from pops.codegen._rhs_coherence import (
    groupable_default_rhs,
    plan_rhs_coherence,
    uses_default_flux,
)

_CONTROL_BLOCK_KEYS = (
    "apply_block", "residual_block", "true_block", "false_block", "cond_block", "body_block",
)


def _block_name(value: Any) -> str | None:
    block = getattr(value, "block", None)
    name = getattr(block, "local_id", None)
    return name if isinstance(name, str) and name else None


def _nested_values(values: Any):
    for value in values:
        yield value
        attrs = getattr(value, "attrs", {})
        for key in _CONTROL_BLOCK_KEYS:
            nested = attrs.get(key)
            if nested:
                yield from _nested_values(nested)


def _nested_control_values(values: Any, path: tuple[str, ...] = ()):
    """Yield values below control-flow nodes together with their exact nesting path."""
    for parent in values:
        attrs = getattr(parent, "attrs", {})
        for key in _CONTROL_BLOCK_KEYS:
            nested = attrs.get(key)
            if not nested:
                continue
            control = getattr(parent, "op", None) or "control"
            nested_path = path + ("%s.%s" % (control, key),)
            for value in nested:
                yield value, nested_path
            yield from _nested_control_values(nested, nested_path)


def _values_with_paths(values: Any):
    """Yield every Program value once with its exact enclosing control/apply path."""
    for value in values:
        yield value, ()
    yield from _nested_control_values(values)


def _prepared_component_templates(block: Any) -> list[dict[str, Any]]:
    """Read the canonical prepared component rows of one resolved block boundary plan."""
    rows: list[dict[str, Any]] = []
    numerics = getattr(block, "numerics", None)
    for boundary in (() if numerics is None else numerics.boundaries):
        compile_data = getattr(boundary, "compile_boundary_data", None)
        if not callable(compile_data):
            raise TypeError(
                "prepared boundary rhs_jacvec validation requires compile_boundary_data()")
        data = compile_data()
        templates = data.get("component_region_templates") if isinstance(data, dict) else None
        if not isinstance(templates, list) or any(not isinstance(row, dict) for row in templates):
            raise TypeError(
                "prepared boundary compile data has no canonical component_region_templates")
        rows.extend(
            row for row in templates if row.get("operation") in {"residual", "jvp"}
        )
    return rows


def _component_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    region = row.get("region")
    region_identity = None if not isinstance(region, dict) else (
        region.get("region_identity") or region.get("identity"))
    values = (
        row.get("producer_identity"), row.get("state_identity"),
        row.get("ghost_identity"), region_identity,
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise TypeError("prepared boundary residual/JVP row has incomplete qualified identities")
    # The guard above proves every entry is a non-empty string.  Preserve the compact validation
    # while exposing its exact post-condition to static consumers.
    return cast(tuple[str, str, str, str], values)


def _qualified_table(row: dict[str, Any], name: str) -> tuple[str, ...]:
    values = row.get(name)
    if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in values):
        raise TypeError(
            "prepared boundary residual/JVP %s table must contain qualified identities" % name)
    return tuple(values)


def _jacvec_location(value: Any, path: tuple[str, ...]) -> str:
    location = "top level" if not path else " -> ".join(path)
    return "rhs_jacvec %r on block %r at %s" % (
        getattr(value, "name", "<unnamed>"), _block_name(value.inputs[2]), location)


def validate_prepared_boundary_jacvec(blocks: tuple[Any, ...], program: Any) -> None:
    """Fail closed when an external boundary JVP cannot execute the authored ``rhs_jacvec``.

    The current matrix-free runtime supplies one direction for the owning conservative state and
    one mutable output.  It can keep solved fields frozen, but it has no tangent-field materializer
    for a field-coupled total derivative.  Validate those facts at resolve rather than after the
    first Krylov matvec.
    """
    if program is None:
        return
    by_name = {getattr(block, "name", None): block for block in blocks}
    rows_by_block: dict[str, list[dict[str, Any]]] = {}
    for value, path in _values_with_paths(program._values):
        if getattr(value, "op", None) != "rhs_jacvec":
            continue
        block_name = _block_name(value.inputs[2])
        if block_name is None:
            raise ValueError(
                "%s has no owner-qualified block identity" % _jacvec_location(value, path))
        block = by_name.get(block_name)
        if block is None:
            raise ValueError(
                "%s has no matching resolved block" % _jacvec_location(value, path))
        if block_name not in rows_by_block:
            rows_by_block[block_name] = _prepared_component_templates(block)
        rows = rows_by_block[block_name]
        if not rows:
            continue

        grouped: dict[
            tuple[str, str, str, str], dict[str, list[dict[str, Any]]]
        ] = defaultdict(lambda: {"residual": [], "jvp": []})
        for row in rows:
            grouped[_component_pair_key(row)][row["operation"]].append(row)

        for pair_key, operations in sorted(grouped.items()):
            residuals, jvps = operations["residual"], operations["jvp"]
            where = _jacvec_location(value, path)
            if len(residuals) != 1 or len(jvps) != 1:
                raise ValueError(
                    "%s requires one exact external FieldBoundaryClosure residual/JVP pair for "
                    "%r; found residual=%d, jvp=%d"
                    % (where, pair_key, len(residuals), len(jvps)))
            residual, jvp = residuals[0], jvps[0]
            exact_fields = (
                "component_id", "component_manifest_identity", "native_interface",
                "interface_version", "states", "fields", "parameters", "rate",
                "nonlinear_iterate",
            )
            changed = [name for name in exact_fields if residual.get(name) != jvp.get(name)]
            if changed:
                raise ValueError(
                    "%s requires its residual/JVP pair to preserve the exact component and "
                    "dependency contract; changed %s" % (where, sorted(changed)))

            state_identity = jvp["state_identity"]
            states = _qualified_table(jvp, "states")
            if states != (state_identity,):
                raise NotImplementedError(
                    "%s reads boundary state dependencies %s, but rhs_jacvec is a local "
                    "single-block linearization and supplies no coupled iterate/direction; "
                    "only the owning state %r is executable"
                    % (where, list(states), state_identity))
            residual_directions = _qualified_table(residual, "directions")
            jvp_directions = _qualified_table(jvp, "directions")
            if residual_directions or jvp_directions != (state_identity,):
                raise NotImplementedError(
                    "%s supports exactly one external boundary JVP direction equal to the "
                    "owning state %r; got residual=%s, jvp=%s"
                    % (where, state_identity, residual_directions, jvp_directions))
            residual_outputs = _qualified_table(residual, "outputs")
            jvp_outputs = _qualified_table(jvp, "outputs")
            if len(residual_outputs) != 1 or len(jvp_outputs) != 1:
                raise NotImplementedError(
                    "%s supports exactly one mutable external boundary output per residual/JVP; "
                    "got residual=%d, jvp=%d"
                    % (where, len(residual_outputs), len(jvp_outputs)))
            fields = _qualified_table(residual, "fields")
            field_coupled = value.attrs.get("field_coupled")
            if not isinstance(field_coupled, bool):
                raise TypeError("%s requires a boolean field_coupled contract" % where)
            if field_coupled and fields:
                raise NotImplementedError(
                    "%s reads solved boundary field(s) %s, but the native matrix-free runtime "
                    "has no field-tangent materializer for field_coupled=True"
                    % (where, list(fields)))


def validate_shared_interface_program(
        blocks: tuple[Any, ...], layout_plan: Any, program: Any, *,
        target: str, resolved_hierarchy: Any = None) -> bool:
    """Prove that every interface is installed and evaluated as one atomic RHS group.

    This runs during resolve, before code generation or engine construction.  The runtime
    scheduler keeps defensive checks, but never discovers an unsupported authored Program on its
    first residual evaluation.
    """
    declarations: dict[str, Any] = {}
    owners: dict[str, set[str]] = defaultdict(set)
    endpoint_owners: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"left": set(), "right": set()})
    for block in blocks:
        numerics = getattr(block, "numerics", None)
        for boundary in (() if numerics is None else numerics.boundaries):
            owned_boundaries = {
                production.region.boundary
                for production in getattr(boundary, "productions", ())
                if production.region.boundary is not None
            }
            for interface in getattr(boundary, "interfaces", ()):
                identity = interface.qualified_id
                previous = declarations.setdefault(identity, interface)
                if previous.canonical_identity() != interface.canonical_identity():
                    raise ValueError(
                        "shared interface %s has competing canonical declarations" % identity)
                owners[identity].add(block.name)
                for side_name, side in (("left", interface.left), ("right", interface.right)):
                    if side.boundary in owned_boundaries:
                        endpoint_owners[identity][side_name].add(block.name)
    if not declarations:
        return False
    if program is None:
        raise ValueError("shared block interfaces require one explicit whole-system Program")

    block_layouts: dict[str, str] = {}
    for assignment in layout_plan.assignments:
        if assignment.subject_kind == "block":
            name = assignment.subject.local_id
            if name in block_layouts:
                raise ValueError("one shared-interface block has multiple LayoutPlan assignments")
            block_layouts[name] = assignment.layout.qualified_id

    neighbours: dict[str, set[str]] = defaultdict(set)
    for identity, interface in declarations.items():
        endpoint_names = []
        for side_name, side in (("left", interface.left), ("right", interface.right)):
            matches = endpoint_owners[identity][side_name]
            if len(matches) != 1:
                raise ValueError(
                    "shared interface %s %s BoundaryHandle must be owned by exactly one runtime "
                    "block; matched %s" % (identity, side_name, sorted(matches)))
            endpoint = next(iter(matches))
            assigned_layout = block_layouts.get(endpoint)
            if assigned_layout != side.layout.qualified_id:
                raise ValueError(
                    "shared interface %s %s endpoint layout differs from block %r LayoutPlan "
                    "assignment" % (identity, side_name, endpoint))
            endpoint_names.append(endpoint)
        left, right = endpoint_names
        if left == right:
            raise ValueError("shared interface %s endpoints resolve to the same block" % identity)
        expected_owners = {left, right}
        if owners[identity] != expected_owners:
            raise ValueError(
                "shared interface %s must be present in both endpoint boundary plans %s; got %s"
                % (identity, sorted(expected_owners), sorted(owners[identity])))
        if block_layouts[left] != block_layouts[right]:
            raise NotImplementedError(
                "shared interface %s crosses LayoutHandles %s and %s; the native atomic "
                "NumericalFlux scheduler requires both endpoint blocks in one co-located runtime "
                "layout. Cross-layout coupling requires an explicit Mapping/Transfer provider."
                % (identity, block_layouts[left], block_layouts[right]))
        neighbours[left].add(right)
        neighbours[right].add(left)

    if target == "amr_system":
        from pops.mesh._amr import FrozenHierarchy

        if resolved_hierarchy is None:
            raise TypeError("shared-interface AMR validation requires a resolved hierarchy")
        hierarchy = resolved_hierarchy.plan
        if hierarchy.level_count != 1 or type(hierarchy.regrid) is not FrozenHierarchy:
            raise NotImplementedError(
                "shared block interfaces on AMR require a prepared interface-flux reflux ledger; "
                "the installed scheduler supports only one frozen level and refuses refined or "
                "regridded hierarchies during resolve"
            )

    participant_names = frozenset(neighbours)
    for value, path in _nested_control_values(program._values):
        block = _block_name(value)
        if uses_default_flux(value) and block in participant_names:
            raise NotImplementedError(
                "shared interface RHS %r on block %r is nested under control flow %s; "
                "the native NumericalFlux scheduler cannot form its required atomic rhs_group "
                "there. Move every endpoint RHS to one top-level coherence round at the same "
                "StagePoint."
                % (value.name, block, " -> ".join(path))
            )
    for value in _nested_values(program._values):
        if getattr(value, "op", None) == "rhs_jacvec" \
                and _block_name(value.inputs[2]) in participant_names:
            raise NotImplementedError(
                "shared NumericalFlux implicit JVP requires a coupled two-sided trace "
                "linearization; the current NumericalFlux scheduler is explicit-only"
            )

    values = list(program._values)
    covered: set[int] = set()
    for value in values:
        block = _block_name(value)
        if uses_default_flux(value) and block in participant_names \
                and not groupable_default_rhs(value):
            raise NotImplementedError(
                "shared interface RHS %r on block %r mixes the default shared flux with named "
                "source work; split the named source into a separate Program node"
                % (value.name, block))

    coherence = plan_rhs_coherence(program, values, block_key=_block_name)
    for round_ in coherence.rounds:
        group = round_.values
        names = [_block_name(row) for row in group]
        present = set(names)
        for row in group:
            name = _block_name(row)
            if name not in participant_names:
                continue
            missing = neighbours[name] - present
            if missing:
                raise ValueError(
                    "shared interface RHS %r at %r is one-sided: block %r requires simultaneous "
                    "endpoint(s) %s in coherence round %d"
                    % (row.name, row.point, name, sorted(missing), round_.occurrence))
            covered.add(row.id)

    ungrouped = [
        value.name for value in values
        if uses_default_flux(value) and _block_name(value) in participant_names
        and value.id not in covered
    ]
    if ungrouped:
        raise ValueError(
            "shared interface default-flux evaluations were not proved simultaneous: %s"
            % sorted(ungrouped))
    return True


__all__ = ["validate_prepared_boundary_jacvec", "validate_shared_interface_program"]
