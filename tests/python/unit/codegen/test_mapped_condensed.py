"""Mapped FV source authoring and exact tensor boundary options."""
from __future__ import annotations

import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.fields.bcs import Dirichlet, Neumann, Periodic
from pops.linalg import LinearProblem
from pops.model import OperatorHandle
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.time import FailRun, Program
from typed_program_support import state_refs
from test_hierarchy_scoped_solve_emit import _coupled_model


def _mapped_program(*, charge=True, mismatched_map=False, damping=1.0):
    model = _coupled_model("mapped_condensed")
    radius = model.aux("radius")
    cosine = model.aux("cosine")
    sine = model.aux("sine")
    model.linear_source("coordinate_gradient", matrix=[
        [0, 0, 0], [0, cosine, -sine / radius], [0, sine, cosine / radius],
    ])
    model.linear_source("coordinate_metric", matrix=[
        [0, 0, 0], [0, radius, 0], [0, 0, 1 / radius],
    ])
    emitted, _ = lower_and_validate(model, facade=model)
    registry = emitted.operator_registry()
    handles = {
        op.name: OperatorHandle(op.name, kind=op.kind, owner=registry.owner_path, signature=op.signature)
        for op in registry.operators_of_kind("local_linear_operator")
    }
    program = Program("mapped_source")._bind_operators(emitted)
    block, state = state_refs(program, "disk", model=emitted)
    temporal = program.state(block[state])
    current = temporal.n
    magnetic = handles["electrostatic_lorentz_J"]
    maps = {"gradient_map": handles["coordinate_gradient"], "base_tensor": handles["coordinate_metric"]}
    coeffs = program.condensed_coeffs(state=current, linear_operator=magnetic, subset=(1, 2),
                                     c=program.dt * program.dt * 4, th_dt=program.dt / 2, **maps)
    rhs_maps = dict(maps)
    if mismatched_map:
        rhs_maps["base_tensor"] = handles["coordinate_gradient"]
    rhs_storage = program.scalar_field("rho_minus_current")
    previous = None if charge else program.history("disk.potential", lag=1, ncomp=1, block=block)
    rhs = program.condensed_rhs(rhs_storage, previous, current, linear_operator=magnetic,
                                subset=(1, 2), th_dt=program.dt / 2, g=program.dt / 2,
                                charge_component=0 if charge else None, **rhs_maps)
    operator = program.matrix_free_operator("mapped_elliptic", scope=Hierarchy())

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("laplacian")
        return -1 * builder.apply_laplacian_coeff(laplacian, value, coeffs)

    program.set_apply(operator, apply)
    solver = CompositeTensorFAC(boundary_conditions=(Neumann(0), Dirichlet(0), Periodic(), Periodic()),
                                diagonal_average="arithmetic", correction_damping=damping)
    potential = program.solve(LinearProblem(operator, rhs, scope=Hierarchy(), nullspace=None), solver=solver).consume(action=FailRun())
    reconstructed = program.condensed_reconstruct(state=current, phi=potential, linear_operator=magnetic,
                                                  subset=(1, 2), th_dt=program.dt / 2,
                                                  gradient_map=maps["gradient_map"], gradient_scale=4)
    endpoint = program.value("next", 1 * reconstructed, at=temporal.next.point)
    program.commit(temporal.next, endpoint)
    return program, emitted


@pytest.mark.parametrize("charge", (True, False))
def test_mapped_source_lowers_metric_gradient_flux_and_authored_faces(charge):
    program, model = _mapped_program(charge=charge)
    source = emit_cpp_program(program, model=model, target="amr_system")
    assert '"operator.arithmetic_diagonal", true' in source
    assert '"boundary.face.0", std::int64_t{2}' in source
    assert '"boundary.face.1", std::int64_t{1}' in source
    assert "ctx.fill_condensed_potential(" in source
    assert "cond_coordinate_gradient_" in source
    assert ".zero_flux_faces" in source
    if charge:
        assert "chargeA(index, 0)" in source
        assert "ctx.prepare_condensed_prior(" not in source
    else:
        assert "ctx.condensed_tensor_laplacian(" in source
        assert "ctx.prepare_condensed_prior(" in source


def test_condensed_field_rejects_different_metric_operators():
    with pytest.raises(ValueError, match="disagree on base_tensor"):
        _mapped_program(mismatched_map=True)


@pytest.mark.parametrize("faces", ((Neumann(1), Dirichlet()), (Dirichlet(2), Neumann()), (Periodic(),)))
def test_tensor_boundary_rejects_unimplemented_or_inexact_laws(faces):
    with pytest.raises((TypeError, ValueError)):
        CompositeTensorFAC(boundary_conditions=faces)


def test_tensor_boundary_snapshot_cannot_be_changed_after_authoring():
    outer = Dirichlet(0)
    solver = CompositeTensorFAC(boundary_conditions=(Neumann(0), outer))
    identity = solver.identity
    outer.value = 100
    assert solver.identity == identity
    assert solver.canonical_options()["boundary_conditions"] == {"0": "neumann", "1": "dirichlet"}


def test_mapped_condensed_refuses_a_uniform_runtime_without_its_boundary_provider():
    program, model = _mapped_program()
    with pytest.raises((ValueError, NotImplementedError), match="AMR|amr_system|unsupported"):
        emit_cpp_program(program, model=model, target="system")


def test_tensor_damping_is_optional_authenticated_and_emitted():
    assert "correction_damping" not in CompositeTensorFAC().canonical_options()
    program, model = _mapped_program(damping=0.5)
    source = emit_cpp_program(program, model=model, target="amr_system")
    assert '"fac.correction_damping", static_cast<double>(0.5)' in source
    assert CompositeTensorFAC(correction_damping=0.5).identity != CompositeTensorFAC().identity


@pytest.mark.parametrize("damping", (0.0, -0.1, 1.01, float("inf"), float("nan"), True))
def test_tensor_damping_rejects_invalid_or_boolean_values(damping):
    with pytest.raises((TypeError, ValueError)):
        CompositeTensorFAC(correction_damping=damping)
