"""The complete Uniform carrier retains provisional results of spatial trials."""
import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time.variant_types import Bool, Product, type_of
from tests.python.integration.runtime import test_implicit_diffusion_lifecycle as diffusion
from tests.python.integration.runtime import test_native_retry_transactions as source


def publish_trial(program, temporal, trial):
    memory = program.product(previous=program.float64(1.0), next_dt=program.float64(0.125))
    initial = program.product(state=temporal.n, time=program.float64(0.0),
                              step=program.int64(0), controller=memory)
    schema = ContinuationSchema(
        initial, time=("time",), step=("step",), fields=((temporal.next, ("state",)),),
        roles=((("controller", "previous"), "retry_control"),
               (("controller", "next_dt"), "retry_control")),
    )
    base = program.continuation_open(initial, schema)
    old = program.continuation_read(base)
    duration = program.requested_dt()
    proposal = program.product(h=duration)
    continuation = program.product(state=trial, time=old["time"] + duration,
                                   step=old["step"] + program.int64(1),
                                   controller=old["controller"])
    envelope = program.product(continuation=continuation, increment=program.product(),
                               observations=program.product(), checks=program.product())
    candidate = program.continuation_freeze(base, proposal, envelope)

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal),
                          candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product()))
    def attest(builder, args):
        expected = args["base"]["time"] + args["proposal"]["h"]
        return builder.product(accepted=args["candidate"]["continuation"]["time"] == expected,
                               checks=builder.product())

    program.continuation_publish(program.continuation_accept(
        candidate, program.continuation_attest(candidate, attest)))


@pytest.mark.parametrize("op", ["source", "diffusive_rhs"])
def test_uniform_spatial_trial_uses_one_complete_candidate_publication(op):
    if op == "source":
        case, layout = source.make_case(direct=True, publish=publish_trial)
    else:
        case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                           publish=publish_trial)
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert sum(value.op == op for value in plan.time._values) == 1
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert cpp.count("ctx.continuation_publish(") == 1
    assert "ctx.continuation_initialize(" in cpp
