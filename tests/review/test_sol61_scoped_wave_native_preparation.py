import numpy as np
import pops
from tests.python.support import scoped_wave_transport as s
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_codegen import emit_cpp_program


def test_public_two_fluxes_validate_resolve_emit():
    case,layout,subjects=s.build();resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={"model_source_policy":"require"})
    assert tuple(b.name for b in resolved.blocks)==s.PARTITION
    assert tuple(len(b.state_spaces) for b in resolved.blocks)==(1,1)
    cpp=emit_cpp_program(resolved.time,model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert 'pops_program_block_count() { return 2; }' in cpp


def test_reference_discriminates_wrong_state_and_global_bound():
    q=s.initial()[s.PARTITION[1]];correct=s.step(q,1)
    assert np.max(np.abs(correct-s.step(q,1,wrong_state=True)))>1e-5
    assert np.max(np.abs(correct-s.step(q,1,wrong_global=True)))>1e-5
    np.testing.assert_allclose(correct.sum(axis=(1,2)),q.sum(axis=(1,2)),rtol=0,atol=2e-13)
    assert np.min(correct[2])>0


def test_constant_preservation_and_independent_fourier_symbol():
    for k,count in enumerate((2,3)):
        constant=np.full((count,s.N,s.N),2.)
        np.testing.assert_array_equal(s.step(constant,k),constant)
    coordinate=np.arange(s.N)*2*np.pi/s.N
    mode=np.broadcast_to(np.cos(coordinate),(s.N,s.N))
    q=np.stack((mode,3*mode))
    v=s.VELOCITIES[0][0];angle=2*np.pi/s.N
    expected=q+s.DT*s.N*(-v*np.sin(angle)*np.stack((-np.sin(coordinate),-3*np.sin(coordinate)))[:,None,:]+2*(np.cos(angle)-1)*q)
    np.testing.assert_allclose(s.step(q,0),expected,rtol=0,atol=2e-15)


def test_actual_selected_emission_reads_density_last():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.module_codegen import _emit_bricks
    case,layout,_=s.build();resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={"model_source_policy":"require"})
    selected=resolved.blocks[1]
    model,_=lower_and_validate(selected.model,state_space=selected.state_spaces[0],resolved_operations=selected.resolved_operations,numerics=selected.numerics)
    text=_emit_bricks(model._m)[1]
    assert 'late_density = U[2]' in text
    assert 'early_left' not in text


def test_all_model_provenance_renderer_and_missing_authority_refusal(tmp_path,monkeypatch):
    # Source-only renderer probes: no compiler, loader, or Native result is represented.
    import json
    from types import SimpleNamespace as NS
    from tests.python.support import m16_explicit_native_capture as c
    binary=tmp_path/'source-standin.so';binary.write_bytes(b'SOURCE-only-no-DSO')
    calls=[]
    class Model:
        so_path=str(binary)
        def __init__(self,name):self.name=name
        def source_provenance(self,*,require_complete):
            assert require_complete is True;calls.append(self.name)
            return {'schema':'SOURCE-renderer-standin','complete':True,'name':self.name}
        def dump_cpp(self,path):Path(path).write_text('SOURCE '+self.name)
    from pathlib import Path
    models=[Model(name) for name in s.PARTITION]
    blocks=[NS(name=name,state_spaces=(name,),model=model) for name,model in zip(s.PARTITION,models)]
    program=NS(so_path=str(binary),abi_key='SOURCE',problem_hash='SOURCE',cache_key='SOURCE')
    row=NS(program=program,layout_id='SOURCE',target='amr_system',block_names=s.PARTITION,identity=NS(token='SOURCE'))
    monkeypatch.setattr(c,'select_partition_program',lambda *args:row)
    def dumped(program,directory):
        for name in ('program.cpp','program.ir.json'): (directory/name).write_text('SOURCE')
    monkeypatch.setattr(c,'dump_retained_program',dumped)
    artifact=NS(blocks=blocks,artifact_identity=NS(token='SOURCE'),manifest=lambda:NS(to_dict=lambda:{}))
    c.retain_provenance(artifact,None,NS(__file__=str(binary)),tmp_path,blocks=s.PARTITION,require_model_sources=True)
    proof=json.loads((tmp_path/'provenance.json').read_text())
    assert proof['schema']=='pops.m16-explicit-retained-provenance@2'
    assert calls==list(s.PARTITION)
    assert {b['block'] for b in proof['model_binaries']}==set(s.PARTITION)
    assert all(b['actual_source']['complete'] is True for b in proof['model_binaries'])
    assert all(b['retained_source']['file'] in proof['files'] for b in proof['model_binaries'])
    artifact.blocks=blocks[:1]
    import pytest
    with pytest.raises(ValueError,match='full partition'):
        c.retain_provenance(artifact,None,NS(__file__=str(binary)),tmp_path,blocks=s.PARTITION,require_model_sources=True)
    artifact.blocks=blocks
    def missing(**kwargs):raise ValueError('actual source evidence missing')
    monkeypatch.setattr(models[1],'source_provenance',missing)
    with pytest.raises(ValueError,match='actual source evidence missing'):
        c.retain_provenance(artifact,None,NS(__file__=str(binary)),tmp_path,blocks=s.PARTITION,require_model_sources=True)
