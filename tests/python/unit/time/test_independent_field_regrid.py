"""Source contracts for independent provider topology; native proof is separate."""
import pytest
from pops.codegen.program_emit_field_representation import emit_field_representation
from pops.time._program.distributed_arrays import Distributed
from pops.time._program.field_representation import validate_field_representation_program
from pops.time._program.detach import detach_compiled_program
from pops.time.value_signatures import value_signature, same_signature
from tests.python.unit.fields.test_independent_field_authority_v02 import source


def fixture():
    program, observations, field, problem = source(joint=True)
    first, second = [observations[field[unknown]] for unknown in problem.unknowns]
    Distributed(program, storage_bytes=1024*1024, storage_versions=128)
    program.__dict__["_continuation_regrid_authoring"] = {"blocks": (), "dimension": 2}
    topology = program.regrid_topology(first, level=program.int64(0), side="source", dimension=2)
    return program, first, second, topology


def test_independent_field_exports_with_exact_provider_without_state_owner():
    p, field, _other, topology = fixture()
    bundle = p.field_bundle(field, topology=topology, cell_capacity=512, patch_capacity=16)
    target = p.regrid_topology(field, level=p.int64(1), side="target", dimension=2)
    destination = p.scalar_field("target", like=field, topology=target, initial=p.float64(0))
    assert same_signature(value_signature(destination), value_signature(field))
    restored = p.bundle_field(bundle, template=field, topology=target)
    assert restored.block is None
    assert same_signature(value_signature(restored), value_signature(field))
    var = {field.id: "old_field", topology.id: "old_topology", target.id: "new_topology"}
    lines = []
    emit_field_representation(p, bundle, var, None, lines, [], {}, {}, "amr_system")
    code = "\n".join(lines)
    assert "old_topology.program_block() != -1" in code
    assert "old_topology.field_provider()" in code and "field_problem" in code
    assert "ctx.export_field_bundle(old_field" in code
    detached = detach_compiled_program(p)
    validate_field_representation_program(detached)
    assert detached._ir_hash() == p._ir_hash()


def test_other_unknown_cannot_borrow_same_width_provider_topology():
    p, _field, other, topology = fixture()
    before = len(p._values)
    with pytest.raises(TypeError, match="exact topology"):
        p.field_bundle(other, topology=topology, cell_capacity=512, patch_capacity=16)
    assert len(p._values) == before


def test_provider_cannot_choose_unbound_dimension_or_storage_width():
    p, field, _other, topology = fixture()
    with pytest.raises(ValueError, match="dimension"):
        p.regrid_topology(field, level=p.int64(0), side="target", dimension=3)
    target = p.regrid_topology(field, level=p.int64(0), side="target", dimension=2)
    with pytest.raises(ValueError, match="component"):
        p.scalar_field("wrong_width", 2, like=field, topology=target, initial=p.float64(0))


@pytest.mark.parametrize("axes", [None, [1, 0]])
def test_nonlinear_observation_storage_axes_are_part_of_the_exact_solve_contract(axes):
    from examples.api_v02.continuation_independent_regrid import build_independent_continuation_regrid
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.time import SolveRequestError
    from pops.time._program.serialization import _json_ready

    spec = build_independent_continuation_regrid()
    program = spec.case._time
    token = next(value for value in program._values if value.op == "solve_spatial_field")
    validate_nonlinear_field_request(program, token)
    assert tuple(token.attrs["source_contract"]["observation_axes"]) == (0, 1)
    request = _json_ready(token.attrs["solve_request"])
    if axes is None:
        del request["physical_problem"]["observation_axes"]
    else:
        request["physical_problem"]["observation_axes"] = axes
    forged = program._replace_value(token, attrs={**token.attrs, "solve_request": request})
    with pytest.raises(SolveRequestError, match="equation_identity_drift"):
        validate_nonlinear_field_request(program, forged)


@pytest.mark.parametrize("reject", [False, True])
def test_independent_regrid_unknown_space_is_not_a_runtime_input(reject):
    import pops
    from pops.model.provider_pack import ProviderPack
    from examples.api_v02.continuation_independent_regrid import build_independent_continuation_regrid

    spec = build_independent_continuation_regrid(reject_regrid_parent=reject)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan.blocks
    for block in plan.blocks:
        pack = ProviderPack.from_data(block.resolved_operations.to_data()["provider_evidence"]["auxiliary"])
        required_inputs = [key for key in pack if pack.declared_entry(key).producer == "runtime_input"
                           and pack.declared_entry(key).availability
                           and pack.declared_entry(key).slot is not None]
        assert required_inputs == []
    observations = [value for value in spec.case._time._values if value.op == "field_component"]
    assert observations
    assert all(value.attrs["field_authority"]["kind"] == "field_problem" for value in observations)
