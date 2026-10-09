import copy,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import pytest
from pops.model import Module,Rate,FluxWaveLaw,ModuleManifest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.params import RuntimeParam
from pops._ir.expr import Const,Var,Minimum,Maximum
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen._compile_emit import emit_cpp_native_loader
from pops.numerics.riemann.waves import provider_of
ROOT=Path(__file__).resolve().parents[2]
def build():
 frame=Rectangle("independent slab",(0,0),(1,2)).frame(Cartesian2D());m=Module("renamed signed trio",frame=frame)
 s=m.state_space("late material",("marker","energy","density"),sampling="cell_average");u=m.state_symbols(s)
 gain=m.param(RuntimeParam("transport_gain",default=0.5));a=m.value(gain);coord=Var("location_read","aux")
 m.aux_field("location_read","cell_scalar",frame=frame.canonical_id,unit="1")
 typed_x,typed_y=frame.axes;x,y=typed_x.name,typed_y.name;vx=(a+coord,-u[2],Const(-0.0));vy=(u[1]/2,-a-coord,Const(0.0))
 values={y:vy,x:vx};pairs={axis:(Minimum(Minimum(v[0],v[1]),v[2]),Maximum(Maximum(v[0],v[1]),v[2])) for axis,v in values.items()}
 f=m.operator("signed_trio_flux",signature=s>>Rate(s),kind="grid_operator",requirements={"aux":("location_read",)},expr={y:tuple(v*q for v,q in zip(vy,u)),x:tuple(v*q for v,q in zip(vx,u))},lowering={"flux_wave_law":FluxWaveLaw(s,values,signed_bounds=pairs)})
 m.rate_operator("trio_balance",state_space=m.state_handle(s),flux=True,fluxes=(f,),default_flux=f,sources=[])
 return m,s,f
def test_param_coordinate_density_late_actual_emission_and_codec(tmp_path):
 m,s,f=build();carrier,_=lower_and_validate(m);assert provider_of(carrier).kind=="explicit_pair"
 assert ModuleManifest.from_dict(m.manifest().to_dict()).to_dict()==m.manifest().to_dict()
 cpp=emit_cpp_native_loader(carrier._m);assert "wave_speeds" in cpp and "location_read" in cpp and "transport_gain" in cpp
 path=tmp_path/"trio.cpp";path.write_text(cpp)
 headers=Path(os.environ["POPS_TEST_HOST_SDK_INCLUDE"]);command=["clang++","-std=c++20","-fsyntax-only","-Xpreprocessor","-fopenmp","-DPOPS_HAS_KOKKOS","-DPOPS_NATIVE_DIM=2","-I",str(ROOT/"include"),"-I",str(headers),str(path)]
 p=subprocess.run(command,capture_output=True,text=True);assert p.returncode==0,p.stderr
@pytest.mark.parametrize("mutation",["output","component","axis","pairbool"] )
def test_detached_owner_shape_axes_refused(mutation):
 m,_,_=build();d=m.manifest().to_dict();law=next(r for r in d["operators"] if r["kind"]=="grid_operator")["lowering_route"]["flux_wave_law"]
 if mutation=="output":law["output_state"]["name"]="foreign"
 elif mutation=="component":law["values"]["x"]["roots"].pop()
 elif mutation=="axis":law["signed_bounds"]["z"]=law["signed_bounds"].pop("x")
 else:law["signed_bounds"]["x"]["roots"][0]=True
 with pytest.raises((ValueError,TypeError)):ModuleManifest.from_dict(d)

def test_signed_graph_unknown_node_refused():
 m,_,_=build();d=m.manifest().to_dict();law=next(r for r in d["operators"] if r["kind"]=="grid_operator")["lowering_route"]["flux_wave_law"]
 law["signed_bounds"]["x"]["nodes"][0]={"op":"invented_opcode","args":[]}
 with pytest.raises((ValueError,TypeError)):ModuleManifest.from_dict(d)

def test_independent_numeric_param_coordinate_late_component():
 m,_,_=build();carrier,_=lower_and_validate(m);aux={"transport_gain":.5,"location_read":.25}
 assert tuple(carrier.eval_wave_speeds(np.array([999.,10.,2.]),aux,0))==(-2.,.75)
 assert tuple(carrier.eval_wave_speeds(np.array([999.,10.,2.]),aux,1))==(-.75,5.)

def test_explicit_signed_zero_pair_keeps_left_right_bits():
 frame=Rectangle("zero pair",(0,0),(1,1)).frame(Cartesian2D());m=Module("zero pair",frame=frame);s=m.state_space("zeroState",("c0","c1"),sampling="cell_average");u=m.state_symbols(s)
 zminus,zplus=Const(-0.0),Const(0.0);v={"x":(zminus,zplus),"y":(zminus,zplus)}
 f=m.operator("zero_flux",signature=s>>Rate(s),kind="grid_operator",expr={axis:tuple(zplus*q for q in u) for axis in v},lowering={"flux_wave_law":FluxWaveLaw(s,v,signed_bounds={axis:(zminus,zplus) for axis in v})})
 m.rate_operator("zero_balance",state_space=m.state_handle(s),flux=True,fluxes=(f,),default_flux=f,sources=[])
 carrier,_=lower_and_validate(m)
 for axis in (0,1):
  lo,hi=carrier.eval_wave_speeds(np.array([2.,3.]),{},axis);assert bool(np.signbit(lo)) and not bool(np.signbit(hi))

@pytest.mark.parametrize("mutation",["selfref","boolchild","orphan","duplicate","arity","bodytype","nanmeta"])
def test_common_dag_structural_refusals(mutation):
 from pops._ir.visitors import validate_dag_key_data
 g={"protocol":"pops.expr.dag.v1","nodes":[["var","aux","alpha"],["var","aux","beta"],["minimum",[0,1]]],"roots":[2,1]}
 validate_dag_key_data(g,root_count=2)
 if mutation=="selfref":g["nodes"][2][1][0]=2
 elif mutation=="boolchild":g["nodes"][2][1][0]=True
 elif mutation=="orphan":g["nodes"].append(["var","aux","unused"])
 elif mutation=="duplicate":g["nodes"][1]=g["nodes"][0]
 elif mutation=="arity":g["nodes"][2][1].append(0)
 elif mutation=="bodytype":g["nodes"][2][1]={"left":0,"right":1}
 else:g["nodes"][0]=["gradient_magnitude",float("nan"),0]
 with pytest.raises(ValueError):validate_dag_key_data(g,root_count=2)
