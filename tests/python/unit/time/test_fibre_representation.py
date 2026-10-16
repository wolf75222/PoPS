"""Finite-fibre algebra and metadata identity; these are not native receptions."""
import json
import pytest

import pops
from pops.identity.encoding import canonical_bytes
from pops.identity.semantic import semantic_value
from pops._ir.quantity import PhysicalDimension
from pops.fibres import FiberAxis, FiberRepresentation
from pops.representations import Representation


def _fibre(weights=(1., 1.)):
    return FiberRepresentation((FiberAxis("v", (-.5,.5), weights, PhysicalDimension(), "zero_flux"),))


def _declared(fibre):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    frame = CartesianDomain("physical",(0.,),(1.,)).frame(Cartesian1D())
    model = pops.Model("same_model_name",frame=frame)
    state = model.state("f",components=fibre.components,representation=fibre.representation,
                        sampling="cell_average")
    case = pops.Case("fibre")
    block = case.block("kinetic",model)
    program = pops.Program("fibre")
    temporal = program.state(block[state])
    program.commit(temporal.next,program.value("next",temporal.n,at=temporal.next.point))
    case.program(program)
    return model,state,program,case


def test_tensor_encoding_neighbours_and_fibre_contractions():
    axes = (FiberAxis("vx",(-.5,.5),(1.,1.),PhysicalDimension(),"periodic",(-1.,0.,1.)),
            FiberAxis("vy",(-1.,0.,1.),(1.,1.,1.),PhysicalDimension(),"zero_flux",(-1.5,-.5,.5,1.5)))
    fibre = FiberRepresentation(axes)
    assert fibre.shape == (2,3) and fibre.size == 6
    assert all(fibre.flat_index(fibre.multi_index(j)) == j for j in range(6))
    assert fibre.neighbour((1,1),axis=0,offset=1) == (0,1)
    assert fibre.neighbour((0,2),axis=1,offset=1) is None
    assert fibre.moment((1.,)*6,(0,0)) == 6.
    assert fibre.moment((1.,)*6,(1,0)) == 0.
    values = (1.,2.,3.,4.,5.,6.)
    divergence = fibre.flux_divergence(values,axis=1,flux=lambda l,r,x:(l+r)/2)
    assert sum(divergence[:3]) == 0. and sum(divergence[3:]) == 0.
    assert fibre.pullback(3.2) == (3.2,)*6
    with pytest.raises(ValueError,match="control-volume"):
        _fibre().flux_divergence((1.,2.),axis=0,flux=lambda l,r,x:l)


def test_representation_descriptor_is_inert_deeply_immutable_and_changes_hash():
    source = {"weights":[1.,2.],"axes":[{"name":"v"}]}
    descriptor = Representation("conservative",source)
    source["weights"][0] = 900.
    assert descriptor.data["weights"] == (1.,2.)
    with pytest.raises(TypeError):
        descriptor.data["weights"][0] = 2.
    with pytest.raises(TypeError,match="inert"):
        Representation("conservative",{"callback":lambda x:x})
    first,first_state,_,_ = _declared(_fibre((1.,1.)))
    second,second_state,_,_ = _declared(_fibre((1.,2.)))
    assert first_state.space != second_state.space
    assert hash(first_state.space) != hash(second_state.space)
    assert first.module.module_hash() != second.module.module_hash()


def test_fibre_descriptor_survives_module_freeze_program_rebuild_and_snapshot():
    fibre = _fibre()
    model,state,program,case = _declared(fibre)
    before = state.space.to_data()
    module = model.module
    assert module.state_spaces()["f"].to_data() == before
    module.freeze()
    graph = program.to_graph()
    rebuilt = program._rebuild(lambda value:True)
    assert [value.space.to_data() for value in rebuilt._values] == [value.space.to_data() for value in program._values]
    frozen_before = rebuilt._serialize()
    rebuilt.freeze()
    assert rebuilt._serialize() == frozen_before
    assert graph is not None
    assert before["representation_data"] == semantic_value(fibre.to_data(), where="fibre")
    fibre.validate(next(value for value in rebuilt._values if value.op == "state"))
    with pytest.raises(ValueError,match="different finite fibre"):
        _fibre((1.,2.)).validate(state)
    # Validation detaches/freeze-copies the complete model authority.
    validated = pops.validate(case)
    assert validated is not None
    assert state.space.to_data() == before


def test_representation_scalar_types_have_one_exact_equality_hash_and_wire_key():
    from pops.model.spaces import StateSpace
    values = (True, 1, 1., 0., -0.)
    representations = [Representation("same", {"value": value}) for value in values]
    spaces = [StateSpace("same", components=("q",), representation_data={"value": value})
              for value in values]
    assert len(set(representations)) == len(values)
    assert len(set(spaces)) == len(values)
    for index, (representation, space) in enumerate(zip(representations, spaces, strict=True)):
        wire = representation.to_dict()
        assert canonical_bytes(wire)
        assert canonical_bytes(space.to_data())
        copied = Representation(representation.name, wire["data"])
        assert copied == representation and hash(copied) == hash(representation)
        for other in representations[index+1:]:
            assert representation != other


def test_fibre_history_budgeted_capture_and_time_step_are_strict_native_identity_data():
    model, state, program, _case = _declared(_fibre())
    current = next(value for value in program._values if value.op == "state")
    result = program.capture(lambda builder: builder.fold(current, 2,
        lambda inner, carried, index: inner.value("twice_carried", 2*carried)))
    accepted = program.require_success(result)
    program.store_history("fibre_budget_result", accepted, depth=2)
    program.step_strategy(pops.time.FixedDt(.125))
    assert program.validate()
    temporal = program.temporal_manifest()
    # This is the owning shape rejected by the native M21 Bind, not a permissive
    # JSON dump. canonical_bytes must see no bare float in any history Space.
    encoded = canonical_bytes({"payload": {"temporal": temporal}})
    assert b"binary64" in encoded
    assert temporal["histories"][0]["space"]["representation_data"] == state.space.to_data()["representation_data"]
    rebuilt = program._rebuild(lambda value: True)
    assert canonical_bytes({"payload": {"temporal": rebuilt.temporal_manifest()}}) == encoded
    model.module.freeze()
    rebuilt.freeze()
    assert canonical_bytes({"payload": {"temporal": rebuilt.temporal_manifest()}}) == encoded


def test_m19_real_spatial_transport_poisson_and_velocity_stencil_emitted():
    from examples.api_v02.models_M15_M28 import build_m19
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    spec = build_m19()
    resolved = pops.resolve(pops.validate(spec.case),layout=spec.layout)
    cpp = emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert next(iter(spec.initial_values.values())).shape == (8,1,16)
    assert "ctx.spatial_interaction(" in cpp and "Kokkos::sin" in cpp
    assert "ctx.rhs_scratch(" in cpp
    assert "ctx.commit_many(" in cpp
    initial = next(iter(spec.initial_values.values()))
    assert not spec.invariants({"distribution":initial})["passed"]
