"""Pure Source/Host TU and scalar naming control; no installation/native extension."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(1,str(ROOT))

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    prefix=Path(sys.prefix)
    common=['/usr/bin/clang++','-Xpreprocessor','-fopenmp','-std=c++20','-DPOPS_NATIVE_DIM=2',
        '-DPOPS_HAS_KOKKOS=1','-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI=1','-I'+str(ROOT/'include'),'-I'+str(prefix/'include')]
    def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    commands=[]
    def run(label,argv,expected=0):
        result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True)
        stdout=out/(label+'.stdout');stderr=out/(label+'.stderr')
        stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
        commands.append({'label':label,'argv':argv,'returncode':result.returncode,'expected':expected,
                         'stdout':pin(stdout),'stderr':pin(stderr)})
        if result.returncode!=expected:raise RuntimeError(label+' failed, raw retained')
    from tests.python.support.public_cpp_symbol_case import build,NAMES
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen._compiler_lowering import require_compiler_lowering
    hooke=Path(sys.argv[2]).resolve()
    assert hooke.is_file()
    # Exact immutable non-author case authority is recorded in the receipt.
    cases=(('actual-red',runpy.run_path(str(hooke))['build']()),
           ('collisions-unicode',build()),('permuted',build(tuple(reversed(NAMES)))),
           ('component-collisions-recovery',build(state_names=('a b','a_b'),recovery=True)),
           ('derived-primitive',build(primitive=True)),
           *((label,build(state_names=names,primitive=True)) for label,names in (
               ('quoted-components',('a"b','a_b')),('backslash-components',(r'a\b','a_b')),
               ('newline-components',('a\nb','a_b')),('reserved-type',('State','a_b')),
               ('reserved-template',('Axis','Prim')),('astral-unicode',('κ','🚀')))))
    cpp=[];metadata_bodies=[];metadata_expected=[]
    for label,resolved in cases:
        graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks);directory=out/label;directory.mkdir()
        path=directory/'program.cpp';path.write_text(emit_cpp_program(resolved.time,model_graph=graph));cpp.append(path)
        for i,block in enumerate(resolved.blocks):
            path=directory/('model-%d.cpp'%i)
            path.write_text(require_compiler_lowering(graph.model_for_block(block.name)).native_loader_source(
                name='SymbolProbe%d'%i,consumer_owner_qid=block.instance_owner_qid,
                declare_auxiliary_providers=block.declares_auxiliary_providers));cpp.append(path)
            methods=re.findall(r'^  static pops::VariableSet (?:conservative|primitive)_vars[^\n]+',path.read_text(),re.M)
            assert len(methods)==2
            metadata_bodies.append('struct Metadata%d {\n%s\n};'%(len(metadata_bodies),'\n'.join(methods)))
            impl=getattr(graph.model_for_block(block.name),'_m',graph.model_for_block(block.name))
            metadata_expected.extend(tuple(name.encode('utf8').hex() for name in names)
                                     for names in (impl.cons_names,impl.prim_state))
    run('actual-program-and-model-host-tu',common+['-fsyntax-only']+[str(p) for p in cpp])
    metadata=out/'metadata-roundtrip.cpp'
    rows=['#include <pops/core/state/variables.hpp>','#include <iostream>','#include <iomanip>',
          *metadata_bodies,'int main(){']
    for i in range(len(metadata_bodies)):
        for method in ('conservative_vars','primitive_vars'):
            rows.append('for(const auto& name: Metadata%d::%s().names){'% (i,method))
            rows.extend(['for(unsigned char c:name) std::cout<<std::hex<<std::setw(2)<<std::setfill(\'0\')<<unsigned(c);',
                         'std::cout<<" ";}std::cout<<"\\n";'])
    rows.append('}');metadata.write_text('\n'.join(rows)+'\n')
    metadata_binary=out/'metadata-roundtrip'
    run('metadata-host-compile',['/usr/bin/clang++','-std=c++20','-I'+str(ROOT/'include'),str(metadata),'-o',str(metadata_binary)])
    run('metadata-byte-roundtrip',[str(metadata_binary)])
    actual=[tuple(line.split()) for line in (out/'metadata-byte-roundtrip.stdout').read_text().splitlines()]
    assert actual==metadata_expected
    from pops.codegen.cpp_symbols import variable_scope,variable_identifier
    from pops.codegen.cpp_writer import _cpp_expand,_cse_emit
    from pops._ir.expr import Var,Minimum,Const
    from pops.codegen.cpp_symbols import variable_bindings
    with variable_scope(('aux',name) for name in NAMES):
        declarations=['const double %s=%d;'%(variable_identifier(name,'aux'),i+1) for i,name in enumerate(NAMES)]
        expression=sum((i+1)*Var(name,'aux') for i,name in enumerate(NAMES))
        rendered=_cpp_expand(expression,{})
        hidden=Minimum(Var(NAMES[0],'aux'),Const(1))
        observed_lines,hidden_values,observed=_cse_emit([hidden],'double','',materialize_all=True,
            return_names=True,scalar_bindings=variable_bindings((hidden,)))
        hidden_name=variable_identifier(NAMES[0],'aux')
        guard=' || '.join('!std::isfinite(%s)'%name for name in observed)
    finite=out/'finite-leaf.cpp';finite.write_text('\n'.join([
        '#include <Kokkos_MathematicalFunctions.hpp>','#include <pops/core/foundation/types.hpp>',
        '#include <cmath>','#include <limits>','int main(){',
        'const double '+hidden_name+'=std::numeric_limits<double>::quiet_NaN();',
        *observed_lines,'const bool rejected='+guard+';',
        'return rejected && std::isfinite('+hidden_values[0]+')?0:1;','}'])+'\n')
    finite_binary=out/'finite-leaf'
    run('finite-leaf-host-compile',common+[str(finite),'-o',str(finite_binary)])
    run('hidden-nonfinite-leaf-refusal',[str(finite_binary)])
    scalar=out/'scalar.cpp';scalar.write_text('\n'.join(['#include <iostream>','#include <pops/core/foundation/types.hpp>','int main(){',*declarations,
        'const double result='+rendered+';', 'std::cout<<result<<"\\n";',
        'return result==91.0?0:1;','}'])+'\n')
    binary=out/'scalar';run('scalar-host-compile',['/usr/bin/clang++','-std=c++20','-I'+str(ROOT/'include'),str(scalar),'-o',str(binary)])
    run('distinct-symbol-scalar-control',[str(binary)])
    replay=str(ROOT/'tests/review/sol61_provider_instance_source_replay.py')
    for mode in ('baseline-legacy','candidate-legacy'):
        run(mode,[sys.executable,replay,mode,str(out/mode)])
    equality=[]
    for path in sorted((out/'baseline-legacy').glob('*.cpp')):
        candidate=out/'candidate-legacy'/path.name
        assert path.read_bytes()==candidate.read_bytes()
        equality.append({'original':pin(path),'candidate':pin(candidate)})
    assert not any(name.startswith('pops.') and Path(getattr(module,'__file__','') or '').suffix in ('.so','.dylib','.pyd')
                   for name,module in tuple(sys.modules.items()))
    report={'schema':'sol61.cpp-local-symbol-source-host@1','contract':'cpp-local-symbols@2','scope':'Source/Host, no PoPS Native/MPI/GPU',
            'metadata_byte_roundtrip':{'source':pin(metadata),'records':len(metadata_expected)},'original_case':pin(hooke),'actual_translation_units':len(cpp),'cpp':[pin(p) for p in cpp],
            'scalar_control':pin(scalar),'hidden_nonfinite_leaf_control':pin(finite),'legacy_full_cpp_equal':equality,'commands':commands}
    (out/'receipt.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(len(cpp),'actual TU, distinct symbols scalar91, 3legacyCPP equal PASS')

if __name__=='__main__':main()
