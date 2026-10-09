"""Finite physical parameter fault: Source/actual-header only, not Native."""
from pathlib import Path
import subprocess
import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.moment_path_kernel import emit_moment_path_kernel
from tests.python.support.atomic_cubature_path_case import declarations,make_case
from tests.python.support.atomic_cubature_fv_oracle import DT,initial_averages


def test_public_parameter_is_in_law_covector_and_resolved_graph():
    case,layout,parameter=make_case(nonconservative=True,amr=True,fixed_dt=DT,scale_parameter=True)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    block=resolved.blocks[0]
    lowered,_=lower_and_validate(block.model,state_space=block.state_spaces[0],
                                resolved_operations=block.resolved_operations,numerics=block.numerics)
    brick=_emit_bricks(lowered._m)[1]
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_emit_params import program_param_entries
    entries=program_param_entries(resolved.time,ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert any(row[1]=='advection_scale' and row[3]==1. for row in entries)
    assert 'param' in brick
    assert parameter.local_id=='advection_scale'
    assert initial_averages()[1].min()>4.


def test_actual_header_initial_spd_admitted_but_huge_finite_direction_refused(tmp_path):
    *_,path,basis,parameter=declarations(nonconservative=True,scale_parameter=True)
    kernel='\n'.join(emit_moment_path_kernel(path.native_kernel()['plan'],'FaultKernel'))
    q=initial_averages()[:,0,0]
    cpp=tmp_path/'fault.cpp';binary=tmp_path/'fault'
    source='#include <pops/numerics/moments/normalized_moment_path.hpp>\n#include <array>\n'+kernel
    source+='\nint main(){std::array<double,6> q={'+','.join(float(v).hex() for v in q)+'};\n'
    source+='''
if(FaultKernel::admissibility(q)!=pops::PathStatus::Success)return 1;
auto small=FaultKernel{}.path_integral(q,q,{1.,0.});if(!small.succeeded())return 2;
const double alpha=1.e308;if(!std::isfinite(alpha))return 3;
auto bad=FaultKernel{}.path_integral(q,q,{alpha,0.});
if(bad.succeeded()||bad.status!=pops::PathStatus::NonFiniteResult)return 4;
return 0;}
'''
    cpp.write_text(source)
    root=Path(__file__).resolve().parents[2]
    subprocess.run(['c++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(root/'include'),str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
