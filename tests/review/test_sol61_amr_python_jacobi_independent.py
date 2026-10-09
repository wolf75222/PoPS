"""Independent contract attacks on actual Python IR/resolve/emission; no native run."""
from copy import copy, deepcopy
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
from pops.solvers import Newton
from pops.identity import make_identity
from pops.time import SolveRequestError
from pops.time._program.serialization import _json_ready
from pops.time._program.spatial_solve import SPATIAL_BASIS_JACOBI
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
from pops.codegen.scratch_plan import build_scratch_plan
from pops.time.values import ProgramValue
import test_sol61_amr_public_original as physical


def clone(token, *, attrs, op=None):
    return ProgramValue(token.prog, token.id, token.vtype, op or token.op, token.inputs,
        attrs, token.name, token.block, space=token.space,
        source_location=token.source_location, field_context=token.field_context,
        region=token.region, state_ref=token.state_ref, point=token.point,
        provenance=token.provenance)


@pytest.fixture(scope="module")
def pair():
    legacy = physical.authored(3, (2, 0, 1), seed=True)
    with patch.object(physical, "Newton", partial(Newton, right_preconditioner="SpatialBasisJacobi@1")):
        selected = physical.authored(3, (2, 0, 1), seed=True)
    return legacy, selected


def test_actual_resolve_cpp_policy_and_physical_suffix(pair):
    (case0, layout0, program0, token0, _), (case1, layout1, program1, token1, _) = pair
    cpp0, resolved0 = physical.emit(case0, layout0)
    cpp1, resolved1 = physical.emit(case1, layout1)
    enum = ", pops::runtime::program::AmrFieldRightPreconditioner::kSpatialBasisJacobi"
    assert enum not in cpp0 and cpp1.count(enum) == 1
    # Real generated callback and consumer remain byte-identical, downstream of preparation.
    marker = "auto amr_original_field_%d_body" % token1.id
    assert marker in cpp0 and marker in cpp1
    assert cpp0.split(marker, 1)[1] == cpp1.split(marker, 1)[1]
    assert token0.attrs["solve_request"]["equation_identity"] == token1.attrs["solve_request"]["equation_identity"]
    assert token0.attrs["newton_controls"] == token1.attrs["newton_controls"]
    assert token0.attrs["solver_identity"] != token1.attrs["solver_identity"]
    assert program0._ir_hash() != program1._ir_hash()
    assert resolved0.time._serialize()["version"] == 8
    assert resolved1.time._serialize()["version"] == 9
    old = build_scratch_plan(resolved0.time).to_dict()
    new = build_scratch_plan(resolved1.time).to_dict()
    old_buffers = [row for row in old["persistent"] if row["kind"] == "prepared_spatial_residual"]
    new_buffers = [row for row in new["persistent"] if row["kind"] == "prepared_spatial_residual"]
    assert len(old_buffers) == len(new_buffers) == 1
    assert new_buffers[0]["buffers"] == old_buffers[0]["buffers"] + 1
    assert "1 + stored DOFs" in new_buffers[0]["note"]


@pytest.mark.parametrize("mutation", ["schema", "request_policy", "solver_digest", "null_policy", "unknown_resigned", "erased_attrs"])
def test_request_mutations_fail_actual_validation(pair, mutation):
    program, token = pair[1][2:4]
    attrs = deepcopy(_json_ready(token.attrs))
    if mutation == "schema":
        attrs["solve_request"]["schema_version"] = 1
    elif mutation == "request_policy":
        attrs["solve_request"]["realization"]["right_preconditioner"] = "Identity@1"
    elif mutation == "solver_digest":
        attrs["solver_identity"] = make_identity("prepared-spatial-newton", attrs["newton_controls"]).token
        attrs["solve_request"]["solver_identity"] = attrs["solver_identity"]
    elif mutation == "null_policy":
        attrs["right_preconditioner"] = None
        attrs["solve_request"]["realization"]["right_preconditioner"] = None
    elif mutation == "unknown_resigned":
        unknown = "pops.amr.original-spatial-jacobi.basis-response@2"
        attrs["right_preconditioner"] = unknown
        attrs["solve_request"]["realization"]["right_preconditioner"] = unknown
        attrs["solver_identity"] = make_identity("prepared-spatial-newton-v2", {
            "controls": attrs["newton_controls"], "right_preconditioner": unknown}).token
        attrs["solve_request"]["solver_identity"] = attrs["solver_identity"]
    else:
        del attrs["right_preconditioner"]
    before = program._ir_hash()
    forged = clone(token, attrs=attrs)
    with pytest.raises(SolveRequestError):
        validate_nonlinear_field_request(program, forged)
    assert program._ir_hash() == before


def test_prepared_and_descriptor_tamper_refused():
    solver = Newton(right_preconditioner="SpatialBasisJacobi@1")
    prepared = solver.prepare_program_solve()
    expected = make_identity("prepared-spatial-newton-v2", {
        "controls": prepared.controls.to_data(), "right_preconditioner": SPATIAL_BASIS_JACOBI})
    assert prepared.identity == expected
    for policy in (None, "SpatialBasisJacobi@1", "pops.amr.original-spatial-jacobi.basis-response@2"):
        with pytest.raises(SolveRequestError):
            replace(prepared, right_preconditioner=policy)
    for policy in (True, 1, "Identity@1", "SpatialBasisJacobi@2"):
        solver._right_preconditioner = policy
        with pytest.raises(SolveRequestError):
            solver.prepare_program_solve()
    solver._right_preconditioner = None
    assert prepared.identity == expected
    assert solver.prepare_program_solve().identity != expected


def test_uniform_refuses_resolution_without_emission_or_authoring_mutation():
    with patch.object(physical, "Newton", partial(Newton, right_preconditioner="SpatialBasisJacobi@1")):
        case, layout, program, _, _ = physical.authored(2, (1, 0), uniform=True)
    before = program._ir_hash()
    with patch.object(physical, "emit_cpp_program", side_effect=AssertionError("emitter must not run")):
        with pytest.raises(ValueError, match="Uniform is unsupported"):
            physical.emit(case, layout)
    assert program._ir_hash() == before


@pytest.mark.parametrize("selected", [False, True])
def test_actual_serialization_walk_preserves_recursive_policy_version(pair, selected):
    # Routing-only malformed-region probe, not admission of a lazy spatial solve:
    # the public original-field builder explicitly requires top-level authoring.
    program, token = pair[int(selected)][2:4]
    probe = copy(program)
    nested = clone(token, op="post_synchronization", attrs={"body_block": (token,)})
    object.__setattr__(probe, "_values", [nested])
    assert probe._serialize(include_provenance=False)["version"] == (9 if selected else 8)
    # Use the actual issued token in its actual owner for this negative query,
    # so refusal is for the mutating solve op, not an ownership mismatch.
    before = program._ir_hash()
    previous = program._dt_bound
    try:
        object.__setattr__(program, "_dt_bound", ((token,), token))
        with pytest.raises(ValueError, match="set_dt_bound: body and captures may only read"):
            program._serialize(include_provenance=False)
    finally:
        object.__setattr__(program, "_dt_bound", previous)
    assert program._ir_hash() == before
