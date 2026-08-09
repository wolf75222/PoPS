"""Install resolved runtime authorities before native block construction.

This seam is intentionally protocol-driven: layout selection stays in ``_runtime_executor`` while
authorities describe the data the chosen engine must install.  A provider that cannot execute an
authority rejects it here, before native blocks freeze their configuration.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast


def _boundary_face_ordinal(value: Any, *, dimension: int, where: str) -> int:
    if not isinstance(value, dict):
        raise TypeError("%s must be one canonical BoundaryHandle identity" % where)
    orientation = value.get("orientation")
    if not isinstance(orientation, dict) or set(orientation) != {
            "schema_version", "axis", "side", "outward_sign"}:
        raise TypeError("%s has no canonical boundary orientation" % where)
    axis = orientation["axis"]
    side = orientation["side"]
    outward_sign = orientation["outward_sign"]
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in range(dimension) \
            or side not in {"lower", "upper"}:
        raise ValueError("%s is not one rank-%d Cartesian face" % (where, dimension))
    expected_sign = -1 if side == "lower" else 1
    if isinstance(outward_sign, bool) or not isinstance(outward_sign, int) \
            or outward_sign != expected_sign:
        raise ValueError("%s outward sign is inconsistent with its side" % where)
    return 2 * axis + (0 if side == "lower" else 1)


def _periodic_identification_rows(
    data: dict[str, Any], face_types: list[str], *, dimension: int,
) -> list[list[int]]:
    raw = data.get("periodic_identifications", [])
    if not isinstance(raw, list):
        raise TypeError("prepared periodic_identifications must be a list")
    rows = []
    claimed = set()
    mapped = 0
    if dimension not in (1, 2, 3) or len(face_types) != 2 * dimension:
        raise ValueError("prepared face table must contain exactly 2*Dim rows")
    required = {
        "source", "target", "source_face", "target_face", "permutation", "signs",
    }
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) != required:
            raise TypeError("prepared periodic identification rows must have exact v1 keys")
        source_face = row["source_face"]
        target_face = row["target_face"]
        if isinstance(source_face, bool) or not isinstance(source_face, int) \
                or isinstance(target_face, bool) or not isinstance(target_face, int) \
                or source_face not in range(2 * dimension) \
                or target_face not in range(2 * dimension) \
                or source_face == target_face:
            raise ValueError(
                "prepared periodic endpoints must be distinct rank-%d face ordinals" % dimension
            )
        if _boundary_face_ordinal(
                row["source"], dimension=dimension,
                where="periodic[%d].source" % index) != source_face \
                or _boundary_face_ordinal(
                    row["target"], dimension=dimension,
                    where="periodic[%d].target" % index) != target_face:
            raise ValueError("prepared periodic face ordinals changed BoundaryHandle identity")
        permutation = row["permutation"]
        signs = row["signs"]
        if not isinstance(permutation, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in permutation) or sorted(permutation) != list(range(dimension)):
            raise ValueError(
                "prepared periodic permutation must be the exact ranked permutation")
        if not isinstance(signs, list) or len(signs) != dimension \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value not in (-1, 1) for value in signs):
            raise ValueError("prepared periodic signs must contain one -1/+1 per source axis")
        source_axis = source_face // 2
        target_axis = target_face // 2
        if permutation[source_axis] != target_axis:
            raise ValueError("prepared periodic source normal does not map to target normal")
        source_outward = -1 if source_face % 2 == 0 else 1
        target_outward = -1 if target_face % 2 == 0 else 1
        if signs[source_axis] != -source_outward * target_outward:
            raise ValueError(
                "prepared periodic normal sign does not map source interior to target exterior")
        endpoints = {source_face, target_face}
        if claimed & endpoints:
            raise ValueError("one prepared face belongs to multiple periodic identifications")
        claimed.update(endpoints)
        is_mapped = permutation != list(range(dimension)) or signs != [1] * dimension
        if is_mapped:
            mapped += 1
            rows.append([
                source_face, target_face,
                *(int(value) for value in permutation),
                *(int(value) for value in signs),
            ])
    periodic_faces = {
        ordinal for ordinal, face_type in enumerate(face_types) if face_type == "periodic"
    }
    if not claimed.issubset(periodic_faces):
        raise ValueError(
            "prepared periodic face types differ from explicit identification endpoints")
    if mapped and len(rows) != 1:
        raise NotImplementedError(
            "mapped periodic topology currently requires one identification; "
            "mixed periodic corners need a composed native scheduler")
    return rows


def _install_boundary_authorities(engine: Any, install_plan: Any) -> None:
    artifact = install_plan.artifact
    dimension = getattr(artifact, "resolved_dimension", None)
    if isinstance(dimension, bool) or dimension not in (1, 2, 3):
        raise TypeError("runtime boundary installation requires one exact resolved dimension")
    compiled_by_name = {row.name: row for row in artifact.blocks}
    reports = {}
    native = getattr(engine, "_s", None)
    install = getattr(native, "_install_boundary_plan", None)
    install_state_route = getattr(native, "_install_block_state_route", None)
    from pops.runtime._component_execution_context import component_execution_data

    execution_data = component_execution_data(install_plan.execution_context)
    component_installers = {
        "apply_region_batch": getattr(native, "_install_ghost_boundary_component", None),
        "transform_faces": getattr(native, "_install_boundary_flux_component", None),
        "residual": getattr(native, "_install_field_boundary_residual_component", None),
        "jvp": getattr(native, "_install_field_boundary_jvp_component", None),
    }
    prepared = []
    state_routes: dict[str, str] = {}
    required_states: set[str] = set()
    required_fields: set[str] = set()
    for block in install_plan.artifact.plan.blocks:
        state_identities = tuple(getattr(block, "state_identities", ()))
        if len(state_identities) != 1:
            raise TypeError(
                "each installed native block requires one exact qualified state identity")
        state_identity = state_identities[0]
        if not isinstance(state_identity, str) or not state_identity:
            raise TypeError("native block state identity must be a non-empty qualified id")
        previous_state = state_routes.setdefault(state_identity, block.name)
        if previous_state != block.name:
            raise ValueError("one qualified state identity is routed to multiple native blocks")
        if not block.boundaries:
            continue
        if len(block.boundaries) != 1:
            raise ValueError(
                "one block must resolve to exactly one composed GhostProducerPlan; got %d "
                "boundary authorities for %r" % (len(block.boundaries), block.name)
            )
        authority = block.boundaries[0]
        protocol = getattr(authority, "runtime_boundary_data", None)
        if not callable(protocol):
            raise TypeError("resolved boundary authority lacks runtime_boundary_data(params)")
        first, second = protocol(install_plan.params), protocol(install_plan.params)
        if type(first) is not dict or first != second \
                or first.get("schema_version") != 1 \
                or first.get("authority_type") != "prepared_boundary_plan":
            raise TypeError(
                "runtime_boundary_data(params) must return one deterministic prepared v1 plan"
            )
        if not callable(install):
            raise NotImplementedError(
                "the selected native provider cannot install resolved ghost-production plans"
            )
        component = compiled_by_name[block.name].model
        ncomp = getattr(component, "n_vars", None)
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise TypeError("compiled block lacks an authenticated positive n_vars")
        faces = first.get("faces")
        expected_faces = list(range(2 * dimension))
        if not isinstance(faces, list) or len(faces) != 2 * dimension \
                or [row.get("ordinal") for row in faces] != expected_faces:
            raise ValueError(
                "prepared boundary plan must contain canonical axis-major rows for dimension %d"
                % dimension)
        types = [row.get("type") for row in faces]
        if any(value not in {
                "periodic", "foextrap", "dirichlet", "no_flux", "slip_wall", "external",
                "characteristic_no_inflow"}
               for value in types):
            raise NotImplementedError("prepared boundary plan selected an unavailable face producer")
        if "characteristic_no_inflow" in types and not bool(
                getattr(component, "has_characteristic_no_inflow", False)):
            raise NotImplementedError(
                "characteristic no-inflow requires a compiled model prepared with "
                "m.roe_from_jacobian(); no component-wise or Euler-specific fallback exists"
            )
        representations = [row.get("representation", "conservative") for row in faces]
        converter_identities = [row.get("converter") for row in faces]
        for face, (face_type, representation, converter) in enumerate(zip(
                types, representations, converter_identities, strict=True)):
            if representation == "conservative":
                if converter is not None:
                    raise ValueError(
                        "prepared conservative boundary face must not carry a converter")
            elif representation == "primitive":
                if face_type != "dirichlet" or not isinstance(converter, str) or not converter:
                    raise ValueError(
                        "prepared primitive boundary face %d requires an exact fixed-state "
                        "converter identity" % face)
            else:
                raise NotImplementedError(
                    "prepared boundary selected unavailable representation %r" % representation)
            if face_type == "characteristic_no_inflow" and representation != "conservative":
                raise NotImplementedError(
                    "characteristic no-inflow requires a conservative reference state"
                )
        face_identities = [row.get("producer") for row in faces]
        if any(not isinstance(value, str) or not value for value in face_identities):
            raise TypeError(
                "prepared boundary faces require non-empty owner-qualified producer identities")
        component_roles = getattr(component, "cons_roles", None)
        if not isinstance(component_roles, (list, tuple)) \
                or len(component_roles) != ncomp \
                or any(not isinstance(role, str) or not role for role in component_roles):
            raise TypeError(
                "compiled block must expose one authenticated physical role per component")
        values = []
        analytic_opcodes = []
        analytic_literals = []
        analytic_clocks = []
        plan_clocks = set()
        for comp in range(ncomp):
            for row in faces:
                row_values = row.get("values")
                if not isinstance(row_values, list) or len(row_values) != ncomp:
                    raise ValueError("prepared boundary face values must exactly cover every component")
                values.append(float(row_values[comp]))
        for face, row in enumerate(faces):
            programs = row.get("analytic_programs", [])
            clock = row.get("analytic_clock")
            if not isinstance(programs, list) or len(programs) not in (0, ncomp):
                raise ValueError(
                    "prepared boundary analytic programs must be empty or cover every component"
                )
            if programs and (types[face] != "dirichlet" or representations[face] != "conservative"):
                raise NotImplementedError(
                    "prepared analytic boundary programs require conservative fixed-state inflow"
                )
            if clock is not None and (not isinstance(clock, str) or not clock or not programs):
                raise TypeError(
                    "prepared boundary analytic Clock must be non-empty text on an analytic face"
                )
            analytic_clocks.append("" if clock is None else clock)
            if clock is not None:
                plan_clocks.add(clock)
            for component in range(ncomp):
                if not programs:
                    analytic_opcodes.append([])
                    analytic_literals.append([])
                    continue
                program = programs[component]
                if not isinstance(program, dict) or set(program) != {"opcodes", "literals"}:
                    raise TypeError(
                        "prepared boundary analytic program must contain opcodes and literals"
                    )
                opcodes = program["opcodes"]
                literals = program["literals"]
                if (
                    not isinstance(opcodes, list)
                    or not opcodes
                    or any(not isinstance(opcode, str) or not opcode for opcode in opcodes)
                    or not isinstance(literals, list)
                    or len(literals) != len(opcodes)
                ):
                    raise ValueError(
                        "prepared boundary analytic opcode/literal rows must be non-empty and aligned"
                    )
                analytic_opcodes.append(opcodes)
                analytic_literals.append([float(value) for value in literals])
        if len(plan_clocks) > 1:
            raise ValueError("prepared analytic boundary plan cannot mix several logical Clocks")
        boundary_state_identity = _canonical_qualified_id(
            first.get("state"), where="prepared boundary state")
        if boundary_state_identity != state_identity:
            raise ValueError("prepared boundary state differs from its owning block route")
        required_depth = first.get("required_depth")
        if isinstance(required_depth, bool) or not isinstance(required_depth, int):
            raise TypeError("prepared boundary required_depth must be an exact integer")
        periodic_identifications = _periodic_identification_rows(
            first, types, dimension=dimension)
        base_arguments = (
            block.name,
            str(first.get("identity")),
            required_depth,
            types,
            values,
            face_identities,
            list(component_roles),
            list(first.get("omitted_interface_faces", [])),
            state_identity,
            periodic_identifications,
            representations,
            ["" if value is None else value for value in converter_identities],
            analytic_opcodes,
            analytic_literals,
            analytic_clocks,
        )
        component_rows = first.get("component_regions", [])
        if not isinstance(component_rows, list):
            raise TypeError("prepared boundary component_regions must be a list")
        component_jobs = []
        for row in component_rows:
            if not isinstance(row, dict):
                raise TypeError("prepared boundary component region must be a dict")
            component_id = row.get("component_id")
            installed = install_plan.components.get(component_id)
            if installed is None:
                raise ValueError(
                    "boundary Handle %s requires exact component %r; it is not installed"
                    % (row.get("target", {}).get("qualified_id"), component_id)
                )
            if installed.component_manifest.token != row.get(
                    "component_manifest_identity"):
                raise ValueError(
                    "boundary Handle %s changed installed component manifest identity"
                    % row.get("target", {}).get("qualified_id")
                )
            interface = row.get("native_interface")
            if not isinstance(interface, dict) or interface != installed.interface.to_data() \
                    or row.get("interface_version") != installed.interface.version:
                raise ValueError(
                    "boundary Handle %s changed installed interface identity/version"
                    % row.get("target", {}).get("qualified_id")
                )
            if installed.native_handle is None:
                raise ValueError("boundary components must be loaded before native installation")
            region = row.get("region")
            if not isinstance(region, dict):
                raise TypeError("boundary component region descriptor must be a dict")
            parameters = row.get("parameters")
            if not isinstance(parameters, list) or any(
                    not isinstance(value, dict)
                    or set(value) != {"qualified_id", "value"}
                    for value in parameters):
                raise TypeError("boundary component parameter table is not canonical")
            operation = row.get("operation")
            if not isinstance(operation, str):
                raise TypeError("prepared boundary component operation must be text")
            install_component = component_installers.get(operation)
            if not callable(install_component):
                raise NotImplementedError(
                    "the selected native provider cannot install typed boundary operation %r"
                    % operation
                )
            for table_name in ("states", "directions", "fields", "outputs"):
                table = row.get(table_name)
                if not isinstance(table, list) or any(
                        not isinstance(identity, str) or not identity for identity in table):
                    raise TypeError(
                        "boundary component %s table must contain qualified identities"
                        % table_name)
            if row.get("state_identity") != state_identity:
                raise ValueError(
                    "boundary component primary state differs from its owning block route")
            required_states.update(row["states"])
            required_fields.update(row["fields"])
            if any(identity != state_identity for identity in row["directions"]):
                raise NotImplementedError(
                    "native boundary JVP directions must use the owning block state storage")
            if operation in {"residual", "jvp"} and len(row["outputs"]) != 1:
                raise NotImplementedError(
                    "native boundary residual/JVP currently requires one exact mutable output")
            if operation == "transform_faces" and (
                    len(row["outputs"]) != 1 or
                    row["outputs"][0] != row["state_identity"] or row["directions"]):
                raise NotImplementedError(
                    "native post-Riemann boundary flux requires one exact state output and no "
                    "JVP direction table")
            component_jobs.append((
                install_component,
                block.name,
                installed.native_handle,
                row,
                "",
                "",
                execution_data,
            ))
        reports[block.name] = MappingProxyType(dict(first))
        prepared.append((base_arguments, tuple(component_jobs)))

    missing_states = required_states - set(state_routes)
    if missing_states:
        raise ValueError(
            "boundary component state dependencies lack exact native block routes: %s"
            % sorted(missing_states))
    available_fields = {}
    for field_plan in install_plan.artifact.plan.field_plans.values():
        unknown = getattr(getattr(field_plan, "operator", None), "unknown", None)
        identity = getattr(unknown, "qualified_id", None)
        options = getattr(field_plan, "native_options", None)
        slot = options.get("provider_slot") if isinstance(options, Mapping) else None
        if not isinstance(identity, str) or not identity or not isinstance(slot, str) or not slot:
            raise TypeError("resolved field plan lacks exact output identity/provider storage route")
        previous = available_fields.setdefault(identity, slot)
        if previous != slot:
            raise ValueError("one solved field identity has competing native provider routes")
    missing_fields = required_fields - set(available_fields)
    if missing_fields:
        raise ValueError(
            "boundary component field dependencies lack exact solved-field routes: %s"
            % sorted(missing_fields))
    # The exact solved-field -> provider-storage registry is shared by every prepared runtime
    # consumer.  Boundary components use a subset, while AMR tagging may consume another subset;
    # install the complete resolved table once instead of manufacturing a tagging-only route.
    field_routes = tuple(sorted(available_fields.items()))

    # Installation is an all-authorities transaction from Python's point of
    # view.  Validate and authenticate every block/component row before the
    # first native mutation, then roll the pre-build plan registry back if a
    # native constructor/prepare rejects any item.  Retrying therefore never
    # encounters a half-installed duplicate plan.
    discard = getattr(native, "_discard_boundary_plans", None)
    if state_routes and not callable(install_state_route):
        raise NotImplementedError(
            "the selected native provider cannot bind qualified block state storage")
    if state_routes and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back boundary authority installation")
    try:
        for state_identity, block_name in sorted(state_routes.items()):
            cast(Callable[..., Any], install_state_route)(block_name, state_identity)
        install_field_route = getattr(native, "_install_field_storage_route", None)
        if field_routes and not callable(install_field_route):
            raise NotImplementedError(
                "the selected native provider cannot bind qualified boundary field storage")
        for field_identity, provider_slot in field_routes:
            cast(Callable[..., Any], install_field_route)(field_identity, provider_slot)
        for base_arguments, component_jobs in prepared:
            cast(Callable[..., Any], install)(*base_arguments)
            for job in component_jobs:
                installer, *arguments = job
                cast(Callable[..., Any], installer)(*arguments)
    except BaseException:
        cast(Callable[..., Any], discard)()
        raise
    engine._boundary_authorities = MappingProxyType(reports)


def _canonical_qualified_id(value: Any, *, where: str) -> str:
    if not isinstance(value, dict):
        raise TypeError("%s must be one canonical Handle identity" % where)
    identity = value.get("qualified_id")
    if not isinstance(identity, str) or not identity:
        raise TypeError("%s has no owner-qualified identity" % where)
    return identity


def _require_interface_component(install_plan: Any, binding: dict[str, Any]) -> Any:
    if not isinstance(binding, dict) or binding.get("operation") != "evaluate_faces":
        raise TypeError(
            "shared conservative flux requires one typed evaluate_faces component binding")
    component_id = binding.get("component_id")
    installed = install_plan.components.get(component_id)
    if installed is None:
        raise ValueError(
            "shared interface requires exact component %r; it is not installed" % component_id)
    if installed.component_manifest.token != binding.get("component_manifest_identity"):
        raise ValueError("shared interface changed installed component manifest identity")
    interface = binding.get("native_interface")
    # Detaching a compiled plan converts tuple carriers to their JSON-equivalent lists.  The
    # canonical encoder intentionally gives both the same ordered-array identity, so compare that
    # authenticated structure instead of Python container implementation details.
    from pops.identity import canonical_bytes

    if not isinstance(interface, dict) or canonical_bytes(interface) != canonical_bytes(
            installed.interface.to_data()) \
            or binding.get("interface_version") != installed.interface.version:
        raise ValueError("shared interface changed native interface identity/version")
    if installed.native_handle is None:
        raise ValueError("shared NumericalFlux component must be loaded before native installation")
    return installed


def _materialized_shared_interface_levels(native: Any, hierarchy: Any) -> tuple[int, ...]:
    """Return the bootstrap-materialized prefix, never the configured level capacity."""
    provider = getattr(native, "n_levels", None)
    if not callable(provider):
        raise TypeError("native AMR shared-interface provider must expose n_levels()")
    materialized = provider()
    configured = hierarchy.level_count
    if type(materialized) is not int or materialized < 1:
        raise RuntimeError(
            "native AMR shared-interface provider returned an invalid materialized level count")
    if type(configured) is not int or configured < 1 or materialized > configured:
        raise RuntimeError(
            "materialized AMR shared-interface levels exceed the resolved hierarchy capacity")
    return tuple(range(materialized))


def _requires_shared_interface_implicit_jacvec_pair(install_plan: Any) -> bool:
    """Read the authenticated compiled-Program requirement, retaining old explicit artifacts."""
    capabilities = install_plan.artifact.plan.capabilities
    if not isinstance(capabilities, Mapping):
        raise TypeError("compiled shared-interface capabilities must be a mapping")
    evidence = capabilities.get("shared_interfaces")
    if evidence is None:
        # Artifacts predating the implicit pair route could contain only explicit shared rates.
        return False
    if not isinstance(evidence, Mapping) or set(evidence) != {"implicit_jacvec_pair"}:
        raise TypeError("compiled shared-interface capability evidence is not canonical")
    required = evidence["implicit_jacvec_pair"]
    if type(required) is not bool:
        raise TypeError("compiled shared-interface implicit-JVP requirement must be an exact bool")
    return required


def _validate_shared_interface_implicit_execution_envelope(
    execution_data: dict[str, Any], rank_count: int
) -> None:
    """Authenticate the narrow native pair envelope without mutating runtime state."""
    if type(rank_count) is not int or rank_count < 1:
        raise RuntimeError("native shared-interface rank count must be a positive integer")
    device = execution_data.get("device_identity")
    memory_space = execution_data.get("memory_space")
    if device not in ("host", "cpu") or memory_space != 1:
        raise NotImplementedError(
            "shared NumericalFlux implicit JVP is currently host-memory-only; device or "
            "managed-memory execution is refused until its paired packing and residual "
            "evaluation have a native portability proof")
    communicator = execution_data.get("communicator_identity")
    if communicator != "serial" or rank_count != 1:
        raise NotImplementedError(
            "shared NumericalFlux implicit JVP is currently serial-only; MPI execution is "
            "refused until its pair admission and local packing have a collective deadlock proof")


def _validate_shared_interface_implicit_execution_before_install(
    install_plan: Any,
) -> None:
    """Refuse an unsupported compiled pair before Program or interface installation mutates AMR."""
    if not _requires_shared_interface_implicit_jacvec_pair(install_plan):
        return
    from pops.runtime._component_execution_context import component_execution_data
    from pops._native_selector import selected_native_module

    _pops = selected_native_module(required=True)
    _validate_shared_interface_implicit_execution_envelope(
        component_execution_data(install_plan.execution_context),
        _pops.n_ranks(),
    )


def _validate_refined_shared_interface_execution(
    levels: tuple[int, ...],
    execution_data: dict[str, Any],
    rank_count: int,
    *,
    dynamic_regrid: bool = False,
    implicit_jacvec_pair: bool = False,
    complete_bind: bool = False,
) -> None:
    """Require one contiguous materialized prefix on the selected communicator.

    Frozen and depth-preserving dynamic hierarchies share this exact execution contract.  Native
    rematerialization prepares a detached collective registry and publishes it only after every
    ``MPI_COMM_WORLD`` rank agrees on the replacement layout identity.
    """
    if not levels or levels != tuple(range(len(levels))):
        raise ValueError("shared-interface materialized levels must be a contiguous L0 prefix")
    if type(rank_count) is not int or rank_count < 1:
        raise RuntimeError("native shared-interface rank count must be a positive integer")
    if type(dynamic_regrid) is not bool:
        raise TypeError("shared-interface dynamic_regrid must be an exact bool")
    if type(implicit_jacvec_pair) is not bool or type(complete_bind) is not bool:
        raise TypeError(
            "shared-interface implicit-JVP and complete-bind contracts must be exact bools")
    if implicit_jacvec_pair:
        _validate_shared_interface_implicit_execution_envelope(
            execution_data, rank_count
        )
    if implicit_jacvec_pair and complete_bind and levels != (0, 1):
        raise NotImplementedError(
            "shared NumericalFlux implicit JVP requires exactly materialized levels (L0, L1) "
            "at bind")
    communicator = execution_data.get("communicator_identity")
    if communicator == "serial":
        if rank_count != 1:
            raise RuntimeError(
                "serial shared-interface execution cannot run in a multi-rank native world")
        return
    if communicator != "MPI_COMM_WORLD":
        raise TypeError("shared-interface execution requires serial or exact MPI_COMM_WORLD")


def finalize_runtime_authorities(
    engine: Any, install_plan: Any, *, complete: bool = False
) -> None:
    """Install authorities for the currently materialized native level prefix.

    Physical ghost plans are installed before block construction so generated closures capture them.
    A shared NumericalFlux is different: both exact endpoint MultiFabs must exist before the scheduler
    can prove their BoxArray, DistributionMapping and face geometry. AMR calls this finalizer before
    bootstrap to authenticate level-zero ownership, after each successful level creation so the next
    proper-nesting proof sees an exact parent-level route, and once with ``complete=True`` before
    bind freezes. Repeated calls must extend the exact prefix and can never reinstall or silently
    replace an existing route.
    """
    from pops.runtime._component_execution_context import component_execution_data

    reports = getattr(engine, "_boundary_authorities", None)
    if reports is None:
        raise RuntimeError("post-block authority finalization lost pre-build boundary reports")
    native = getattr(engine, "_s", None)
    install = getattr(native, "_install_interface_flux_component", None)
    previous_reports = getattr(engine, "_interface_authorities", None)
    if previous_reports is None:
        previous_reports = {}
    if not isinstance(previous_reports, Mapping):
        raise TypeError("installed shared-interface authority reports must be a mapping")
    rows: dict[str, dict[str, Any]] = {}
    owners: dict[str, set[str]] = {}
    endpoint_owners: dict[str, dict[str, set[str]]] = {}
    for block_name, report in reports.items():
        bindings = report.get("interface_component_bindings", [])
        if not isinstance(bindings, list):
            raise TypeError("prepared interface_component_bindings must be a list")
        for row in bindings:
            if not isinstance(row, dict) or set(row) != {"interface", "component"}:
                raise TypeError("prepared interface component binding is not canonical")
            interface = row["interface"]
            identity = _canonical_qualified_id(
                interface.get("handle") if isinstance(interface, dict) else None,
                where="shared interface")
            previous = rows.setdefault(identity, row)
            if previous != row:
                raise ValueError(
                    "shared interface %s has competing runtime declarations" % identity)
            owners.setdefault(identity, set()).add(block_name)
        endpoints = report.get("interface_endpoints", [])
        if not isinstance(endpoints, list):
            raise TypeError("prepared interface_endpoints must be a list")
        for endpoint in endpoints:
            if not isinstance(endpoint, dict) or set(endpoint) != {"interface", "owned_sides"}:
                raise TypeError("prepared shared-interface endpoint row is not canonical")
            identity = endpoint["interface"]
            if not isinstance(identity, str) or not identity:
                raise TypeError("prepared shared-interface endpoint identity is invalid")
            sides = endpoint["owned_sides"]
            if not isinstance(sides, list) or any(side not in {"left", "right"} for side in sides):
                raise TypeError("prepared shared-interface endpoint sides are invalid")
            table = endpoint_owners.setdefault(
                identity, {"left": set(), "right": set()})
            for side in sides:
                table[side].add(block_name)
    if not rows:
        if previous_reports:
            raise RuntimeError(
                "shared-interface declarations disappeared between authority finalizations")
        engine._interface_authorities = MappingProxyType({})
        return
    if not callable(install):
        raise NotImplementedError(
            "the selected native provider cannot install shared NumericalFlux components")

    block_layouts: dict[str, str] = {}
    for assignment in install_plan.artifact.layout_plan.assignments:
        if assignment.subject_kind != "block":
            continue
        name = assignment.subject.local_id
        if name in block_layouts:
            raise ValueError("native block has multiple LayoutPlan assignments")
        block_layouts[name] = assignment.layout.qualified_id
    block_names_provider = getattr(native, "block_names", None)
    if not callable(block_names_provider):
        raise TypeError("native shared-interface provider must expose block_names()")
    block_names = tuple(cast(Any, block_names_provider()))
    if len(block_names) != len(set(block_names)):
        raise ValueError("native block registry contains duplicate names")
    block_indices = {name: index for index, name in enumerate(block_names)}
    execution_data = component_execution_data(install_plan.execution_context)
    implicit_jacvec_pair = _requires_shared_interface_implicit_jacvec_pair(install_plan)
    adaptive = {row.adaptive for row in install_plan.artifact.layout_plan.layouts}
    levels = (0,)
    if adaptive == {True}:
        from pops.mesh._amr import FrozenHierarchy, RegridSchedule

        hierarchy = install_plan.resolved_hierarchy.plan
        frozen = type(hierarchy.regrid) is FrozenHierarchy
        dynamic_refined = (
            type(hierarchy.regrid) is RegridSchedule and hierarchy.level_count >= 2
        )
        if not frozen and not dynamic_refined:
            raise NotImplementedError(
                "shared interface runtime finalization supports any frozen materialized L0 "
                "prefix; dynamic regrid requires at least two configured levels and the complete "
                "prefix active at bind")
        levels = _materialized_shared_interface_levels(native, hierarchy)
        from pops._native_selector import selected_native_module

        _pops = selected_native_module(required=True)
        _validate_refined_shared_interface_execution(
            levels,
            execution_data,
            _pops.n_ranks(),
            dynamic_regrid=dynamic_refined,
            implicit_jacvec_pair=implicit_jacvec_pair,
            complete_bind=complete,
        )
        if complete and dynamic_refined and levels != tuple(range(hierarchy.level_count)):
            raise NotImplementedError(
                "dynamic shared interfaces require the complete configured prefix materialized "
                "at bind; post-bind active-depth creation/removal is not executable"
            )
    elif adaptive != {False}:
        raise ValueError("shared interface finalization requires one coherent layout capability")

    installed_reports = {}
    jobs = []
    if set(previous_reports) - set(rows):
        raise RuntimeError(
            "installed shared-interface authority has no current resolved declaration")
    import hashlib
    import json

    for identity, row in sorted(rows.items()):
        interface = row["interface"]
        declaration_identity = hashlib.sha256(
            json.dumps(
                interface,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        endpoints = []
        for side_name in ("left", "right"):
            side = interface.get(side_name)
            if not isinstance(side, dict):
                raise TypeError("shared interface %s endpoint is not canonical" % side_name)
            layout_id = _canonical_qualified_id(
                side.get("layout"), where="shared interface %s layout" % side_name)
            matches = endpoint_owners.get(
                identity, {"left": set(), "right": set()})[side_name]
            if len(matches) != 1:
                raise ValueError(
                    "shared interface %s BoundaryHandle must identify exactly one native block"
                    % side_name)
            endpoint = next(iter(matches))
            if block_layouts.get(endpoint) != layout_id:
                raise ValueError(
                    "shared interface %s endpoint layout differs from native block assignment"
                    % side_name)
            endpoints.append(endpoint)
        left, right = endpoints
        if owners[identity] != {left, right}:
            raise ValueError(
                "shared interface %s runtime ownership differs from its endpoint plans" % identity)
        if block_layouts[left] != block_layouts[right]:
            raise NotImplementedError(
                "native shared NumericalFlux requires co-located endpoint blocks in one layout")
        try:
            left_index, right_index = block_indices[left], block_indices[right]
        except KeyError as error:
            raise ValueError(
                "shared interface endpoint block %r was not materialized" % error.args[0]) from None
        installed = _require_interface_component(install_plan, row["component"])
        component_id = row["component"]["component_id"]
        previous = previous_reports.get(identity)
        previous_levels: tuple[int, ...] = ()
        if previous is not None:
            if not isinstance(previous, Mapping) or set(previous) != {
                    "left_block", "right_block", "levels", "component_id",
                    "declaration_identity"}:
                raise TypeError(
                    "installed shared-interface authority report is not canonical")
            if previous["left_block"] != left or previous["right_block"] != right \
                    or previous["component_id"] != component_id \
                    or previous["declaration_identity"] != declaration_identity:
                raise RuntimeError(
                    "shared-interface authority changed after level-zero installation")
            raw_levels = previous["levels"]
            if type(raw_levels) is not tuple or any(
                    type(level) is not int for level in raw_levels):
                raise TypeError(
                    "installed shared-interface levels must be one exact tuple of integers")
            previous_levels = raw_levels
            if previous_levels != tuple(range(len(previous_levels))) \
                    or any(level not in levels for level in previous_levels):
                raise RuntimeError(
                    "installed shared-interface levels are not a prefix of materialized levels")
        # Empty overrides are deliberate: LoadedComponent owns the authenticated
        # parameters/target JSON captured from the installed component manifest.
        # Boundary binding scalars travel independently in the typed invocation
        # request and must never replace that component preparation contract.
        parameters_json = ""
        target_json = ""
        for level in levels:
            if level in previous_levels:
                continue
            jobs.append((
                left_index, right_index, level, installed.native_handle,
                interface, row["component"], parameters_json, target_json,
                execution_data,
            ))
        installed_reports[identity] = MappingProxyType({
            "left_block": left,
            "right_block": right,
            "levels": levels,
            "component_id": component_id,
            "declaration_identity": declaration_identity,
        })
    discard = getattr(native, "_discard_interface_flux_components", None)
    checkpoint_provider = getattr(native, "_interface_flux_installation_checkpoint", None)
    rollback_installations = getattr(
        native, "_rollback_interface_flux_installations", None
    )
    transactional_prefix = callable(checkpoint_provider) and callable(
        rollback_installations
    )
    if jobs and not transactional_prefix and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back shared interface installation")
    accepted_size = None
    if jobs and transactional_prefix:
        accepted_size = cast(Callable[[], Any], checkpoint_provider)()
        if type(accepted_size) is not int or accepted_size < 0:
            raise RuntimeError(
                "native shared-interface installation checkpoint is invalid"
            )
    try:
        for job in jobs:
            cast(Callable[..., Any], install)(*job)
    except BaseException:
        try:
            if accepted_size is None:
                cast(Callable[..., Any], discard)()
            else:
                cast(Callable[[int], Any], rollback_installations)(accepted_size)
        finally:
            engine._interface_authorities = MappingProxyType(dict(previous_reports))
        raise
    engine._interface_authorities = MappingProxyType(installed_reports)


def _install_amr_provider_authorities(engine: Any, install_plan: Any) -> None:
    """Install every AMR provider through its authority-carried runtime protocol."""

    providers = install_plan.amr_providers
    if not isinstance(providers, Mapping) \
            or tuple(providers) != ("clustering", "tagger", "reflux"):
        raise ValueError(
            "adaptive runtime requires exact clustering, tagger and reflux providers")
    native = getattr(engine, "_s", None)
    from pops.amr.providers import prepare_amr_provider_installation
    from pops.runtime._component_execution_context import component_execution_data

    layout_identity = install_plan.artifact.layout_plan.qualified_id
    execution_data = component_execution_data(install_plan.execution_context)
    jobs = []
    reports = {}
    resolved_tagging = getattr(
        getattr(install_plan, "bootstrap_plan", None), "tagging", None)
    resolved_tagging_identity = getattr(resolved_tagging, "qualified_id", None)
    for role, frozen in providers.items():
        prepared = prepare_amr_provider_installation(
            role=role,
            frozen_binding=frozen,
            layout_identity=layout_identity,
            components=install_plan.components,
            native=native,
            resolved_tagging_identity=resolved_tagging_identity,
        )
        if prepared.installer is not None:
            jobs.append((
                prepared.installer,
                prepared.native_handle,
                prepared.binding,
                execution_data,
            ))
        reports[prepared.role] = MappingProxyType(prepared.binding)

    discard = getattr(native, "_discard_amr_provider_components", None)
    if jobs and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back AMR provider installation")
    try:
        for installer, handle, binding, execution in jobs:
            cast(Callable[..., Any], installer)(handle, binding, execution)
    except BaseException:
        cast(Callable[..., Any], discard)()
        raise
    engine._amr_provider_authorities = MappingProxyType(reports)


def install_runtime_authorities(engine: Any, install_plan: Any) -> None:
    """Install every pre-build authority carried by one normalized install plan."""
    _install_boundary_authorities(engine, install_plan)
    adaptive = {row.adaptive for row in install_plan.artifact.layout_plan.layouts}
    if adaptive == {False}:
        return
    if adaptive != {True}:
        raise ValueError("runtime authorities require one coherent layout capability")

    _install_amr_provider_authorities(engine, install_plan)

    execution = install_plan.amr_execution
    protocol = getattr(execution, "runtime_execution_data", None)
    if not callable(protocol):
        raise TypeError("adaptive execution authority must implement runtime_execution_data()")
    first, second = protocol(), protocol()
    if type(first) is not dict or first != second \
            or set(first) != {"schema_version", "authority_type", "mode", "relations"} \
            or first.get("schema_version") != 2 \
            or first.get("authority_type") != "amr_execution":
        raise TypeError("AMR runtime_execution_data() must return one deterministic v2 dict")
    relations = first["relations"]
    if not isinstance(relations, list):
        raise TypeError("AMR execution relations must be a list")
    if first.get("mode") == "synchronous":
        nlevels = len(install_plan.resolved_hierarchy.plan.transitions) + 1
        relations = [
            {
                "parent_level": parent, "child_level": parent + 1,
                "temporal_ratio": {"numerator": 1, "denominator": 1},
                "remainder_policy": "integral_only",
            }
            for parent in range(nlevels - 1)
        ]
    elif first.get("mode") != "subcycled":
        raise ValueError("AMR execution mode must be subcycled or synchronous")
    expected = len(install_plan.resolved_hierarchy.plan.transitions)
    if len(relations) != expected:
        raise ValueError("AMR execution requires one temporal relation per hierarchy transition")
    for index, row in enumerate(relations):
        if (not isinstance(row, dict) or set(row) != {
                "parent_level", "child_level", "temporal_ratio", "remainder_policy"}):
            raise ValueError("AMR execution temporal relation has incomplete keys")
        ratio = row["temporal_ratio"]
        if (row["parent_level"] != index or row["child_level"] != index + 1
                or not isinstance(ratio, dict)
                or set(ratio) != {"numerator", "denominator"}
                or isinstance(ratio["numerator"], bool)
                or not isinstance(ratio["numerator"], int)
                or isinstance(ratio["denominator"], bool)
                or not isinstance(ratio["denominator"], int)
                or ratio["denominator"] <= 0 or ratio["numerator"] < ratio["denominator"]
                or row["remainder_policy"] not in {
                    "integral_only", "explicit_final_substep"}):
            raise ValueError("AMR execution temporal relation is not canonical")
        if (ratio["numerator"] % ratio["denominator"] != 0
                and row["remainder_policy"] == "integral_only"):
            raise ValueError("non-integral AMR temporal relation requires an explicit remainder")
    engine.set_temporal_relations(
        [int(row["temporal_ratio"]["numerator"]) for row in relations],
        [int(row["temporal_ratio"]["denominator"]) for row in relations],
        [str(row["remainder_policy"]) for row in relations],
    )
    installed_execution = dict(first)
    installed_execution["relations"] = [
        {
            **row,
            "temporal_ratio": dict(row["temporal_ratio"]),
        }
        for row in relations
    ]
    engine._amr_execution_authority = MappingProxyType(installed_execution)

    if install_plan.bootstrap_plan is not None:
        from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

        flow_bootstrap_tagging(
            engine, install_plan.bootstrap_plan, install_plan.params,
            clock_identity=install_plan.amr_providers["tagger"]["clock_identity"],
            field_plans=install_plan.artifact.plan.field_plans,
        )


__all__ = ["finalize_runtime_authorities", "install_runtime_authorities"]
