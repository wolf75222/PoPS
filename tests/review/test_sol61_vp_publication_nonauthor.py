"""Source counter-cases use claims from the actual public VP resolved Program."""
from copy import deepcopy
import pytest
from tests.python.support.m19_vlasov_poisson_case import build
from pops.codegen.program_field_publication import _merge_publication_claim
from pops.time._program.serialization import _json_ready


@pytest.fixture(scope='module')
def claims(tmp_path_factory):
    resolved=build(tmp_path_factory.mktemp('vp-occurrence-claims'))
    block=next(row for row in resolved.blocks if row.name=='z_phase')
    rows=block.resolved_operations.provider_evidence['program_field_publications']
    assert len(rows)==1
    claim=_json_ready(rows[0])
    assert claim['occurrence_contract']=='mapped-publication-occurrences@1'
    assert len(claim['mapped_occurrences'])==2
    physical={key:value for key,value in claim.items() if key not in ('mapped_occurrences','occurrence_contract')}
    return tuple({**physical,'mapped_output':row} for row in claim['mapped_occurrences'])


@pytest.mark.parametrize('change',('producer','unknown','physical_map','source_port','target_port'))
def test_actual_vp_foreign_authority_stays_refused(claims,change):
    a,b=deepcopy(claims)
    if change in b['mapped_output']:
        b['mapped_output'][change]={'foreign':True}
    else:
        b[change]={'foreign':True}
    with pytest.raises(ValueError,match='competing'):
        _merge_publication_claim(a,b)


def test_actual_vp_same_point_distinct_invocation_stays_refused(claims):
    a,b=deepcopy(claims)
    assert a['mapped_output']['invocation']!=b['mapped_output']['invocation']
    b['mapped_output']['target_point']=a['mapped_output']['target_point']
    with pytest.raises(ValueError,match='same point'):
        _merge_publication_claim(a,b)


def test_actual_vp_all_model_bricks_emit_with_declared_storage(tmp_path):
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.module_emit_brick import emit_cpp_brick
    resolved=build(tmp_path)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    bodies={row.name:emit_cpp_brick(graph.model_for_block(row.name)._m,name='VP_'+row.name)
            for row in resolved.blocks}
    assert set(bodies)=={'a_spectator','m_number','z_phase'}
    assert all(body.strip() for body in bodies.values())
