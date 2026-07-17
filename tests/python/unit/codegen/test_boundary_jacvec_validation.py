"""Resolve-time guards for external PreparedBoundaryPlan residual/JVP pairs."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from pops.codegen._interface_validation import validate_prepared_boundary_jacvec


STATE = "case::block::state"
FIELD_A = "case::field::potential"
FIELD_B = "case::field::temperature"


def _component_row(
        operation: str, *, directions: tuple[str, ...] | None = None,
        outputs: tuple[str, ...] | None = None, fields: tuple[str, ...] = (),
        states: tuple[str, ...] = (STATE,),
        component_id: str = "pops://test/field-boundary@1") -> dict[str, object]:
    is_jvp = operation == "jvp"
    return {
        "target": {"qualified_id": "case::boundary::%s" % operation},
        "component_id": component_id,
        "component_manifest_identity": "component-manifest:test-field-boundary",
        "native_interface": {
            "name": "field_boundary_closure", "version": 1,
        },
        "interface_version": 1,
        "operation": operation,
        "producer_identity": "case::boundary::producer",
        "state_identity": STATE,
        "ghost_identity": "case::boundary::left-face",
        "region": {
            "kind": "face", "region_identity": "case::boundary::left-face",
        },
        "states": list(states),
        "directions": list(
            ((STATE,) if is_jvp else ()) if directions is None else directions),
        "fields": list(fields),
        "parameters": [],
        "outputs": list(
            ("case::boundary::%s-output" % operation,) if outputs is None else outputs),
        "rate": "",
        "nonlinear_iterate": STATE,
    }


class _Boundary:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def compile_boundary_data(self) -> dict[str, object]:
        return {"component_region_templates": deepcopy(self._rows)}


def _program(*, field_coupled: bool, nested: bool = False) -> object:
    iterate = SimpleNamespace(block=SimpleNamespace(local_id="block"))
    jacvec = SimpleNamespace(
        op="rhs_jacvec", name="implicit_boundary_jvp",
        inputs=(object(), object(), iterate, object()),
        attrs={"field_coupled": field_coupled},
    )
    matrix = SimpleNamespace(
        op="matrix_free_operator", name="newton_operator",
        attrs={"apply_block": (jacvec,)},
    )
    values = (matrix,)
    if nested:
        values = (SimpleNamespace(
            op="branch", name="nested_newton",
            attrs={"true_block": (matrix,)},
        ),)
    return SimpleNamespace(_values=values)


def _validate(rows: list[dict[str, object]], *, field_coupled: bool,
              nested: bool = False) -> None:
    block = SimpleNamespace(
        name="block",
        numerics=SimpleNamespace(boundaries=(_Boundary(rows),)),
    )
    validate_prepared_boundary_jacvec(
        (block,), _program(field_coupled=field_coupled, nested=nested))


@pytest.mark.parametrize("field_coupled", (False, True))
def test_boundary_jacvec_accepts_one_exact_state_only_pair(field_coupled: bool) -> None:
    _validate(
        [_component_row("residual"), _component_row("jvp")],
        field_coupled=field_coupled,
    )


def test_boundary_jacvec_accepts_multiple_frozen_primal_fields() -> None:
    fields = (FIELD_A, FIELD_B)
    _validate(
        [_component_row("residual", fields=fields),
         _component_row("jvp", fields=fields)],
        field_coupled=False,
    )


def test_boundary_jacvec_rejects_field_coupling_without_field_tangents() -> None:
    with pytest.raises(
            NotImplementedError,
            match=r"no field-tangent materializer for field_coupled=True"):
        _validate(
            [_component_row("residual", fields=(FIELD_A,)),
             _component_row("jvp", fields=(FIELD_A,))],
            field_coupled=True,
        )


def test_boundary_jacvec_rejects_cross_block_state_dependency() -> None:
    other = "case::other-block::state"
    with pytest.raises(
            NotImplementedError,
            match=r"local single-block linearization.*no coupled iterate/direction"):
        _validate(
            [_component_row("residual", states=(STATE, other)),
             _component_row("jvp", states=(STATE, other))],
            field_coupled=False,
        )


@pytest.mark.parametrize("directions", ((), ("case::other::state",), (STATE, STATE)))
def test_boundary_jacvec_rejects_non_unique_or_foreign_direction(
        directions: tuple[str, ...]) -> None:
    with pytest.raises(
            NotImplementedError,
            match=r"exactly one external boundary JVP direction equal to the owning state"):
        _validate(
            [_component_row("residual"),
             _component_row("jvp", directions=directions)],
            field_coupled=False,
        )


@pytest.mark.parametrize("operation,outputs", (
    ("residual", ()),
    ("jvp", ("case::output::first", "case::output::second")),
))
def test_boundary_jacvec_rejects_non_unique_output(
        operation: str, outputs: tuple[str, ...]) -> None:
    rows = [_component_row("residual"), _component_row("jvp")]
    rows[0 if operation == "residual" else 1] = _component_row(
        operation, outputs=outputs)
    with pytest.raises(
            NotImplementedError,
            match=r"exactly one mutable external boundary output"):
        _validate(rows, field_coupled=False)


@pytest.mark.parametrize("rows", (
    [_component_row("residual")],
    [_component_row("residual"),
     _component_row("jvp", component_id="pops://test/other-boundary@1")],
))
def test_boundary_jacvec_rejects_missing_or_changed_exact_pair(
        rows: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match=r"exact external FieldBoundaryClosure residual/JVP pair|"
                                               r"exact component and dependency contract"):
        _validate(rows, field_coupled=False)


def test_boundary_jacvec_nested_error_reports_the_exact_control_path() -> None:
    with pytest.raises(
            NotImplementedError,
            match=(r"branch\.true_block -> matrix_free_operator\.apply_block.*"
                   r"exactly one external boundary JVP direction")):
        _validate(
            [_component_row("residual"), _component_row("jvp", directions=())],
            field_coupled=False,
            nested=True,
        )
