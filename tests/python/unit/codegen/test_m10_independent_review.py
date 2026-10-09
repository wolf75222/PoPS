"""Independent finite-volume M10 checks; no codegen oracle and no native claim."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from tests.python.integration.runtime import test_m10_self_consistent_sg as authored


def poisson(charge):
    n = len(charge)
    identity = np.eye(n)
    operator = n*n*(2*identity-np.roll(identity,1,axis=1)-np.roll(identity,-1,axis=1))
    # Independent gauge-fixed matrix solve, rather than the author's Fourier eigenvalue.
    return np.linalg.lstsq(np.vstack((operator,np.ones(n))), np.r_[charge,0.], rcond=None)[0]


def fitted_faces(density, potential):
    # Integrating-factor edge problem: G=-j=D/h*(exp(delta)*nR-nL)/int_0^1 exp(t*delta)dt.
    # Gauss integration is independent of the author's Bernoulli branch and expm1 recipe.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    delta = np.roll(potential,-1)-potential
    integrals = .5*np.sum(weights[:,None]*np.exp(.5*(nodes[:,None]+1)*delta[None]),axis=0)
    return authored.D*len(density)*(np.exp(delta)*np.roll(density,-1)-density)/integrals


def step(density, faces):
    return density+authored.DT*len(density)*(faces-np.roll(faces,1))


def test_independent_poisson_and_integrating_factor_agree_and_discriminate_bad_routes():
    equilibrium, background, density, potential, stale = authored._data()
    solved = poisson(density-background)
    np.testing.assert_allclose(solved,potential,rtol=0,atol=2e-14)
    faces = fitted_faces(density,solved)
    np.testing.assert_allclose(faces,authored._face_flux(density,potential),rtol=0,atol=2e-13)
    expected = step(density,faces)
    np.testing.assert_allclose(expected,authored._one_step(density,potential),rtol=0,atol=3e-15)
    assert np.max(np.abs(fitted_faces(equilibrium,poisson(equilibrium-background)))) < 2e-13
    assert abs(expected.sum()-density.sum()) < 1e-13
    stale_gap = np.max(np.abs(expected-step(density,fitted_faces(density,stale))))
    assert stale_gap > authored.STALE_MIN_GAP
    assert stale_gap/authored.STATE_ATOL > 500
    assert np.max(np.abs(expected-step(density,fitted_faces(density,-solved)))) > 1e-5
    diffusion_again = authored.D*authored.N*(np.roll(density,-1)-density)
    assert np.max(np.abs(expected-step(density,faces+diffusion_again))) > 1e-5


def test_discrete_equilibrium_is_not_claimed_as_exact_average_of_the_continuum_profile():
    nodes, weights = np.polynomial.legendre.leggauss(24)
    centers = (np.arange(authored.N)+.5)/authored.N
    x = centers[None]+nodes[:,None]/(2*authored.N)
    exact_average = .5*np.sum(weights[:,None]*np.exp(-authored.AMPLITUDE*np.cos(2*np.pi*x)),axis=0)
    equilibrium, _, _, _, potential = authored._data()
    assert np.max(np.abs(exact_average-equilibrium)) > 1e-4
    assert np.max(np.abs(fitted_faces(exact_average,potential))) > 1e-5


def test_public_poisson_load_and_stored_field_use_exact_current_state_occurrences():
    from pops.time._evaluation_point import evaluation_stage_fraction
    _,_,_,_,program,_ = authored._case()
    current = {node.block.local_id:node for node in program._values if node.op == "state"}
    load = next(node for node in program._values if node.op == "field_problem_load")
    assert {node.id for node in load.inputs} == {current["density"].id,current["background"].id}
    assert all(node is current[node.block.local_id] for node in load.inputs)
    assert evaluation_stage_fraction(load) == 0
    stores = [node for node in program._values if node.op == "store_history"]
    assert len(stores) == 1 and stores[0].attrs["history"] == "stage-potential"
    assert stores[0].inputs[0].op == "field_component"
    assert evaluation_stage_fraction(stores[0].inputs[0]) == 0


def synthetic_records():
    _,_,density,potential,_ = authored._data()
    faces = fitted_faces(density,potential)
    records = []
    for cell in range(authored.N):
        for side in (0,1):
            flux = float(faces[(cell+side-1)%authored.N])
            orientation = -1 if side == 0 else 1
            records.append(dict(operation_identity="one-fitted-operator",
                occurrence_identity="joint-occurrences:0,1", evaluation_context="stage:0",
                quadrature_identity="cell:%d/axis:0/side:%d" % (cell,side),
                orientation=orientation, multiplicity=1,face_measure=1.,
                temporal_weight=authored.DT,numerical_flux=flux,
                integrated_amount=orientation*authored.DT*flux))
    return records, step(density,faces), density, potential


@pytest.mark.parametrize("empty_rank", (False,True))
def test_partitioned_or_empty_rank_ledger_is_gathered_without_deduplication(monkeypatch, empty_rank):
    records, actual, density, potential = synthetic_records()
    pieces = [records,[]] if empty_rank else [records[::2],records[1::2]]
    context = SimpleNamespace(communicator=SimpleNamespace(identity="MPI_COMM_WORLD",handle=object()))
    runtime = SimpleNamespace(_executor=SimpleNamespace(_program_exchange_records=lambda:pieces[1]))
    def gather(world, local):
        assert world is context.communicator.handle and local is pieces[1]
        return pieces
    monkeypatch.setattr(authored,"allgather_value",gather)
    global_records = authored._global_exchange_records(runtime,context)
    assert len(pieces[1]) != 2*authored.N  # original rank-local assertion was wrong
    authored._assert_exchange_records(global_records,actual,density,potential)


@pytest.mark.parametrize("defect", ("duplicate", "orientation", "double", "time"))
def test_ledger_rejects_missing_or_duplicated_incidence_and_wrong_sign_or_weight(defect):
    records, actual, density, potential = synthetic_records()
    records = copy.deepcopy(records)
    if defect == "duplicate":
        records[1] = records[0].copy()
    elif defect == "orientation":
        records[0]["orientation"] *= -1
    elif defect == "double":
        records[0]["multiplicity"] = 2
    else:
        records[0]["temporal_weight"] *= 2
    with pytest.raises(AssertionError):
        authored._assert_exchange_records(records,actual,density,potential)


def test_rank_local_receipt_error_is_converged(monkeypatch):
    context = SimpleNamespace(communicator=SimpleNamespace(identity="MPI_COMM_WORLD",handle=object()))
    monkeypatch.setattr(authored,"allgather_value",lambda world,value:[value,""])
    def failed_write():
        raise OSError("unwritable receipt")
    with pytest.raises(AssertionError,match="rank 0: OSError: unwritable receipt"):
        authored._collective_check(context,failed_write)
