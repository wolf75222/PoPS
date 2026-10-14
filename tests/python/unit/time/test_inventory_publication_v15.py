"""Authoring gate for complete C12 cases; native proof is a separate receiver."""
import json
import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_source_map import source_map
from examples.api_v02.inventory_publication_v15 import build_inventory_publication_v15


@pytest.mark.parametrize('mode', (
    'dynamic_replacement', 'nonlinear_exact', 'nonlinear_tolerated',
    'bad_inverse', 'negative_branch'))
def test_inventory_publication_closed_graph_validates_resolves_and_emits(mode, tmp_path):
    spec = build_inventory_publication_v15(mode=mode)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    _retain(source, tmp_path)


def test_inventory_current_dependency_validates_resolves_and_emits(tmp_path):
    from examples.api_v02.current_inventory_v15 import build_current_inventory_v15
    spec = build_current_inventory_v15()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    _retain(source, tmp_path)


def _retain(source, directory):
    # Correspondence is checked on the emitted bytes, not inferred from a symbol.
    # This is source evidence only: no compiler or native System is invoked.
    (directory/'program.cpp').write_text(source)
    correspondence = source_map(source)
    (directory/'source_map.json').write_text(json.dumps(correspondence, sort_keys=True, indent=2)+'\n')
    assert correspondence['complete'] and not correspondence['unlocated']
    assert correspondence['operations']


def test_inventory_derivative_primal_ledger_validates_resolves_and_emits(tmp_path):
    from examples.api_v02.derivative_inventory_v15 import build_derivative_inventory_v15
    spec = build_derivative_inventory_v15()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    _retain(source, tmp_path)
