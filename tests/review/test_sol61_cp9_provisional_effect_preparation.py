"""Public Source authoring only; Native test is collected separately."""
import pops
from tests.python.integration.runtime.test_uniform_checkpoint_provisional_effect import build


def test_public_checkpoint_effect_resolves_on_same_authored_clock():
    case,layout,initial,dt=build()
    resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    assert set(initial)=={'early_transport','late_transport'} and dt>0
    assert resolved.consumer_graph.nodes
    row,=resolved.consumer_graph.nodes
    assert row.kind.value=='checkpoint' and row.target_uri=='transactional-cp9'
