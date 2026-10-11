"""A local condensed source preserves incoming transport quadrature, never invents it."""
import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_lowerability import _check_amr_flux_weights
from pops.model import OperatorHandle
from pops.numerics.terms import Flux
from pops.time import Program
from test_hierarchy_scoped_solve_emit import _coupled_model
from typed_program_support import state_refs


def source_program(*, integrated=True, opaque=False):
    model = _coupled_model("condensed_source_flux_budget")
    emitted, _ = lower_and_validate(model, facade=model)
    registry = emitted.operator_registry()
    declared = next(op for op in registry.operators_of_kind("local_linear_operator")
                    if op.name == "electrostatic_lorentz_J")
    rotation = OperatorHandle(declared.name, kind=declared.kind, owner=registry.owner_path,
                              signature=declared.signature)
    program = Program("source_after_transport")._bind_operators(emitted)
    block, state = state_refs(program, "plasma", model=emitted)
    q = program.state(block[state])
    rate = program.rhs(state=q.n, terms=[Flux()])
    factor = program.dt / 2 if integrated else 1
    h1 = program.value("transport half", q.n + factor * rate, at=q.n.point)
    work = program.value("independent momentum", 1 * h1, at=h1.point)
    if opaque:
        theta = program._new("state", "opaque_source", (work,), {}, "unknown source", work.block,
                             space=work.space)
    else:
        phi = program.history("scalar_phi", lag=1, ncomp=1, block=block)
        theta = program.condensed_reconstruct(state=work, phi=phi,
            linear_operator=rotation, subset=(1, 2), th_dt=program.dt / 2, c_rho=0)
    endpoint = program.value("source endpoint", 2 * theta - h1, at=q.next.point)
    program.commit(q.next, endpoint)
    return program


def test_condensed_source_preserves_integrated_transport_ledger():
    _check_amr_flux_weights(source_program())


def test_condensed_source_does_not_hide_unintegrated_transport():
    with pytest.raises(ValueError, match="dt powers"):
        _check_amr_flux_weights(source_program(integrated=False))


def test_unknown_state_transform_is_still_refused():
    with pytest.raises(ValueError, match="without exact flux-weight propagation"):
        _check_amr_flux_weights(source_program(opaque=True))
