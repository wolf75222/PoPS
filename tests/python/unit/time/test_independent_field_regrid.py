"""The independent provider-owned Field moves through regrid_read; native proof is separate.

N10 lot 3b: the port deleted the topology, the bundles and ``scalar_field(topology=)``, and the
refusals that guarded their arguments (another Field's topology, a ``dimension=`` mismatch, a
target width chosen by the case) went with them: the read allocates from the source Field.
"""
import pytest


def test_independent_field_is_read_with_its_exact_provider_and_no_state_owner():
    from examples.api_v02.continuation_independent_regrid import build_independent_continuation_regrid
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time._graph.value_traversal import walk_program_nodes
    from pops.time._program.continuation_regrid import TARGET_INSTANCE, hierarchy_instance
    from pops.time.value_signatures import same_signature, value_signature

    program = build_independent_continuation_regrid().case._time
    reads = [value for value in walk_program_nodes(program._values)
             if value.op == "continuation_regrid_read"]
    assert [value.attrs["path"] for value in reads] == [("state",), ("previous",)]
    previous = reads[1]
    assert previous.block is None
    assert value_signature(previous)["field_authority"]["kind"] == "field_problem"
    assert same_signature(value_signature(previous), value_signature(previous.inputs[0]))
    assert all(hierarchy_instance(value) == TARGET_INSTANCE for value in reads)
    source = emit_cpp_program(program, target="amr_system")
    assert source.count("ctx.regrid_read(") == 2


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
