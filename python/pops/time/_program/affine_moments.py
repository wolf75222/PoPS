"""Typed common affine push-forward of a complete Cartesian velocity-moment state."""
from __future__ import annotations

from typing import Any

from pops.time._authoring import atomic_authoring
from pops.time._program.value_validation import (
    require_compatible_spaces, require_owned, require_top_level,
)
from pops.time.operator_resolution import resolve_operator_handle
from pops.time.values import ProgramValue


def affine_moment_update(
    program: Any, state: Any, mean: Any, *, linear_operator: Any,
    theta_dt: Any, order: int, name: Any, rotation: str = "cayley",
) -> Any:
    """Author the kernel; model/frame/AMR validation remains explicit and fail-closed."""
    from pops.moments.model_builder import moment_names

    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 4:
        raise ValueError("affine_moment_update order must be an integer from one to four")
    if not isinstance(rotation, str) or rotation not in ("cayley", "exponential"):
        raise ValueError("affine_moment_update rotation must be 'cayley' or 'exponential'")
    for label, value in (("state", state), ("mean", mean)):
        if not isinstance(value, ProgramValue) or value.vtype != "state":
            raise TypeError("affine_moment_update %s must be a typed State value" % label)
        require_owned(program, value, "affine_moment_update " + label)
        require_top_level(program, value, "affine_moment_update " + label)
    if state.block != mean.block:
        raise ValueError("affine_moment_update inputs must belong to the same block")
    require_compatible_spaces(state.space, mean.space, "affine_moment_update", typed_pair=True)
    if tuple(state.space.components) != tuple(moment_names(order)):
        raise ValueError("affine_moment_update requires the complete canonical 2V raw-moment basis")
    if program._recording:
        raise ValueError("affine_moment_update must be authored at module-scope Program level")
    operator = resolve_operator_handle(
        program, linear_operator, where="affine_moment_update linear_operator",
        expected_kinds="local_linear_operator", values=(state, mean))
    coefficient = program._coeff_dict(theta_dt, "theta_dt", "affine_moment_update")
    attrs = {"linear_operator": operator.name, "order": order, "theta_dt": coefficient}
    # Keep existing Cayley IR/identity bytes unchanged, including an explicit default.
    if rotation != "cayley":
        attrs["rotation"] = rotation
    return program._new(
        "state", "affine_moment_update", (state, mean),
        attrs,
        name, state.block, space=state.space, point=mean.point,
        field_context=mean.field_context, state_ref=state.state_ref,
    )


class _ProgramAffineMoments:
    @atomic_authoring
    def affine_moment_update(
        self, state: Any, mean: Any, *, linear_operator: Any,
        theta_dt: Any, order: int = 4, rotation: str = "cayley", name: Any = None,
    ) -> Any:
        """Push one common affine velocity map through all 2V raw moments.

        ``mean`` is the actual first-moment endpoint from the electric/magnetic
        solve, in the same complete state space as ``state``. Its density must be
        unchanged. The first-moment submatrix of ``linear_operator`` must be the
        skew rotation ``Omega*[[0,1],[-1,0]]``. ``theta_dt`` is half the source
        interval and may depend on ``Program.dt``. Both first moments are copied
        exactly; higher moments follow the same affine velocity map.

        ``rotation="cayley"`` preserves the existing CN rotation and program
        identity. ``rotation="exponential"`` uses the exact centered gyro phase
        ``2*Omega*theta_dt`` for the represented inputs, with a compensated
        product. The full interval and phase product must be finite; an
        unsupported phase is refused without falling back to Cayley. This
        integrates centered moments exactly for a homogeneous source cell with
        constant Omega. The supplied coupled mean remains approximate when
        obtained by CN, so this is not a full exact or AP source integrator.

        Refined AMR supports only a source-first prefix followed by conservative
        transport. Any transport ancestry is refused, since a source applied
        before reflux cannot stand in for a source applied after synchronization.
        This operation has no temperature floor or realizability projection.
        """
        return affine_moment_update(
            self, state, mean, linear_operator=linear_operator,
            theta_dt=theta_dt, order=order, name=name, rotation=rotation)


def validate_affine_moment_prefix(program: Any) -> None:
    """Authenticate the narrow source-first dependency envelope, never an op-name waiver."""
    from pops.time._program.value_validation import TOP_LEVEL_REGION

    forbidden = {
        "rhs", "diffusive_rhs", "history", "layout_map_import", "solve_spatial_nonlinear",
        "subcycle", "synchronize", "post_synchronization",
    }

    def walk(value: Any, seen: set[int]) -> None:
        value = program._canonical_value(value)
        if id(value) in seen:
            return
        seen.add(id(value))
        if value.op in forbidden:
            raise ValueError(
                "AMR affine_moment_update requires a synchronized source-first prefix; "
                "transport/history ancestry must be synchronized before a later source update")
        if value.attrs.get("schedule") is not None:
            raise ValueError("AMR affine_moment_update does not admit scheduled source ancestry")
        for index, child in enumerate(value.inputs):
            child = program._canonical_value(child)
            # A scalar history at the typed linear solve's initial-guess edge is
            # algorithmic state. It does not enter the converged source equation.
            # Do not mark it visited: the same history must still be refused if
            # another edge uses it as an RHS, coefficient or physical state.
            if (value.op == "solve_linear" and value.vtype == "scalar_field"
                    and value.attrs.get("problem_kind") in {
                        "matrix_free_linear", "scalar_tensor_elliptic_hierarchy"}
                    and value.attrs.get("has_guess") is True and index == 2
                    and len(value.inputs) == 3 and child.op == "history"
                    and child.vtype == "scalar_field" and child.attrs.get("ncomp") == 1
                    and child.attrs.get("schedule") is None):
                continue
            walk(child, seen)
        # Recorded matrix-free applies, branches and loops can capture a transported
        # state through attrs rather than inputs. Use the same complete traversal as
        # Program liveness; a pure elliptic apply region is legal, hidden transport is not.
        for child in program._subblock_value_refs(value):
            walk(child, seen)

    for value in program._values:
        if value.op == "affine_moment_update":
            if value.region != TOP_LEVEL_REGION:
                raise ValueError("AMR affine_moment_update requires a top-level source-first prefix")
            walk(value, set())
