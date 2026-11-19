"""Publish the exact proposed trial through the Uniform continuation carrier."""

from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time.variant_types import Bool, Product, type_of


def publish_trial(program, temporal, trial, *, attest_sibling=False):
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

    assessed = program.continuation_freeze(base, proposal, envelope) if attest_sibling else candidate
    program.continuation_publish(program.continuation_accept(
        candidate, program.continuation_attest(assessed, attest)))
