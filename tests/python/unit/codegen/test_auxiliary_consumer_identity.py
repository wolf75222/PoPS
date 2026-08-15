"""Exact auxiliary consumer identities stay unique across shared-model blocks."""

from __future__ import annotations

from pops.codegen.program_emit_kernels import program_provider_consumer_qid
from pops.physics._facade import Model
from pops.problem import Case


def _transport():
    model = Model("shared_aux_transport")
    (density,) = model.conservative_vars("density")
    model.flux(x=[density])
    model.eigenvalues(x=[density])
    model.primitive_vars(density=density)
    model.conservative_from([density])
    return model


def test_shared_model_block_instance_owners_are_officially_distinct() -> None:
    model = _transport()
    case = Case("aux-consumer-identity")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)

    ion_owner = str(ions.instance_owner_path.canonical())
    electron_owner = str(electrons.instance_owner_path.canonical())

    assert ion_owner != electron_owner
    assert "block:ions" in ion_owner
    assert "block:electrons" in electron_owner
    assert str(model.owner_path.canonical()) in ion_owner
    assert str(model.owner_path.canonical()) in electron_owner


def test_native_registrar_uses_block_instance_consumer_qid() -> None:
    model = _transport()
    case = Case("aux-consumer-identity-emit")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_owner = str(ions.instance_owner_path.canonical())
    electron_owner = str(electrons.instance_owner_path.canonical())
    model_owner = str(model.owner_path.canonical())

    ion_src = model.__pops_native_loader_source__(
        name="ions", target="system", consumer_owner_qid=ion_owner
    )
    electron_src = model.__pops_native_loader_source__(
        name="electrons", target="system", consumer_owner_qid=electron_owner
    )

    ion_plan = ion_owner + "/physical_flux"
    electron_plan = electron_owner + "/physical_flux"
    model_plan = model_owner + "/physical_flux"

    assert 'ConsumerPlan{"%s"' % ion_plan in ion_src
    assert 'ConsumerPlan{"%s"' % electron_plan in electron_src
    assert ion_plan != electron_plan
    assert 'ConsumerPlan{"%s"' % model_plan not in ion_src
    assert 'ConsumerPlan{"%s"' % model_plan not in electron_src
    assert 'ConsumerPlan{"%s"' % ion_plan not in electron_src
    assert 'ConsumerPlan{"%s"' % electron_plan not in ion_src


def test_program_consumer_qid_uses_block_instance_owner() -> None:
    model = _transport()
    case = Case("aux-consumer-identity-program")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)

    ion_qid = program_provider_consumer_qid(model, 4, ions)
    electron_qid = program_provider_consumer_qid(model, 4, electrons)
    model_qid = program_provider_consumer_qid(model, 4)

    assert ion_qid != electron_qid
    assert ion_qid == str(ions.instance_owner_path.canonical()) + "/program/4"
    assert electron_qid == str(electrons.instance_owner_path.canonical()) + "/program/4"
    assert model_qid == str(model.owner_path.canonical()) + "/program/4"
    assert model_qid != ion_qid
