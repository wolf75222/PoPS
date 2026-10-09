"""Installed public pipeline and retained outputs; no physics/core implementation."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import numpy as np
import pops
from examples.migration.scientific.api040_stage_fields_library import author_case,literal_inputs,DT,DECAY
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evidence_json import evidence_dumps,ENCODING

ROOT=Path(__file__).resolve().parents[3]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Capture:
    arrays: dict
    directory: Path
    variant: str
    decay: Fraction


def identity():
    from pops._native_selector import select_native_dimension
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature,pops_include
    from scripts.check_packaging_manifest import read_manifest,PYTHON_SOURCE_SUFFIXES
    from scripts.verify_installed_native import verify_installed_native
    package=Path(pops.__file__).resolve().parent
    assert any(package==Path(sysconfig.get_path(k)).resolve()/'pops' for k in ('purelib','platlib'))
    assert package.is_relative_to(Path(sys.prefix).resolve()) and not package.is_relative_to(ROOT)
    native=select_native_dimension(2)
    origin=verify_installed_native(expect_dimension=2,expect_mpi=bool(native.__has_mpi__),
                                   expect_parallel_hdf5=bool(native.__has_parallel_hdf5__))
    assert module_header_signature()==pops_header_signature(package/'include')
    assert Path(pops_include()).resolve()==(package/'include').resolve()
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--','python/pops/'],text=True).splitlines()
    names=[name for name in names if Path(name).suffix in PYTHON_SOURCE_SUFFIXES]
    names+=['include/'+str(name) for name in read_manifest(ROOT).installed_headers]+['include/pops_headers.manifest']
    for name in names:
        installed=package/(name.removeprefix('python/pops/') if name.startswith('python/pops/') else name)
        assert sha(installed)==sha(ROOT/name),name
    world=native.mpi_world() if native.__has_mpi__ else None
    assert world is None or (int(world.rank)==0 and int(world.size)==1), 'First scientific witness is world1 only'
    return {'Source':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'Native':sha(native.__file__),'SDK':module_header_signature(),
            'native_path':str(native.__file__),'native_origin':str(origin),
            'package':str(package),'installed_payload_files':len(names)}


def checkpoint_receipt(path):
    path=Path(path);members={}
    with np.load(path,allow_pickle=False) as z:
        for key in z.files:
            a=np.asarray(z[key]);members[key]={'dtype':a.dtype.str,'shape':list(a.shape),
                'bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
    return {'path':str(path),'sha256':sha(path),'members':members}


def capture(variant,directory):
    """Execute only when an actual pytest/runtime phase is authorized."""
    from pops.model.provider_pack import ProviderPack
    assert variant in ('representative','rename','decay')
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    before=identity()
    labels={} if variant!='rename' else {role:'renamed_'+role for role in (
        'domain','donor_model','donor_state','receiver_model','receiver_state','phi_input','psi_input',
        'case','donor_block','receiver_block','field_owner','phi','psi','field_problem','method')}
    decay=Fraction(11,10) if variant=='decay' else DECAY
    authored=author_case(labels=labels,decay=decay);initial=literal_inputs()
    np.savez(directory/'actual-bind-inputs.npz',**initial)
    plan=pops.resolve(pops.validate(authored.case),layout=authored.layout)
    assert plan.resolved_dimension==2
    artifact=pops.compile(plan)
    for i,block in enumerate(artifact.blocks):block.model.dump_cpp(directory/('model%d.cpp'%i))
    for i,row in enumerate(artifact.layout_programs):row.program.dump_cpp(directory/('program%d.cpp'%i))
    receiver=labels.get('receiver_block','receiver');donor=labels.get('donor_block','reservoir')
    runtime=pops.bind(artifact,initial_state={receiver:initial['receiver'],donor:initial['donor']},
        resources={'execution_context':artifact_execution_context(artifact)})
    report=pops.run(runtime,t_end=float(DT),max_steps=1,console=False)
    shape=initial['receiver'].shape
    arrays={'qfinal':np.asarray(runtime.state_global(receiver)).reshape(shape).copy(),
            'd_final':np.asarray(runtime.state_global(donor)).reshape(shape).copy(),
            'd_initial':initial['donor'].copy()}
    expected_histories={'receiver-stage-%d'%stage for stage in range(3)}|{
        'observed-stage-%d-%s'%(stage,role) for stage in range(3) for role in ('phi','psi')}
    assert set(runtime.history_names())==expected_histories
    for stage in range(3):
        arrays['q%d'%stage]=np.asarray(runtime.history_global('receiver-stage-%d'%stage,1)).reshape(shape).copy()
        for role in ('phi','psi'):
            arrays['%s%d'%(role,stage)]=np.asarray(runtime.history_global('observed-stage-%d-%s'%(stage,role),1)).reshape(shape).copy()
    block=next(row for row in artifact.plan.blocks if row.name==receiver)
    operations=block.resolved_operations.to_data()
    pack=ProviderPack.from_data(operations['provider_evidence']['auxiliary'])
    for role,default in [('phi','screened_first'),('psi','screened_second')]:
        keys=[key for key in pack if key.component==labels.get(role+'_input',default)]
        assert len(keys)==1
        arrays['resident_'+role]=np.asarray(runtime._executor.auxiliary_component(keys[0])).reshape(shape).copy()
    assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in arrays.values())
    np.savez(directory/'actual-arrays.npz',**arrays)
    (directory/'actual-report.json').write_text(evidence_dumps({'report':report.to_data(),'encoding':ENCODING}))
    checkpoint=runtime.checkpoint(directory/'accepted-checkpoint')
    after=identity();assert after==before
    receipt={'variant':variant,'decay':str(decay),'labels':labels,'identity':after,
        'accepted_steps':report.accepted_steps,'time':runtime.time(),'macro_step':runtime.macro_step(),
        'checkpoint':checkpoint_receipt(checkpoint),'provider_operations':operations,
        'program':plan.time._serialize(),'field_plans':{name:value.to_data() for name,value in plan.program_field_plans.items()},
        'histories':{name:{'depth':runtime.history_depth(name),'slot':1,'dt':runtime.history_slot_dt(name,1)} for name in sorted(expected_histories)},
        'binaries':{str(path):sha(path) for path in [artifact.so_path,*artifact.layout_program_paths.values(),*(b.model.so_path for b in artifact.blocks)]},
        'scope':'Actual public world1 capture; author test comparison is not separate independent reception',
        'gaps':['No successful CG report/iteration getter is claimed; independent saved-field residuals are checked',
                'History sample window/dt are not a persistent readout of Field publication stage c'],
        'independent_oracle_required':True}
    (directory/'capture-receipt.json').write_text(evidence_dumps(receipt))
    assert report.accepted_steps==1 and runtime.macro_step()==1 and runtime.time()==float(DT)
    return Capture(arrays,directory,variant,decay)
