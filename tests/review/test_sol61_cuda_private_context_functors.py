"""Host checks of real named device callable bodies; no CUDA qualification."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PREFIX = Path("/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17")

def test_named_functors_real_field_views(tmp_path):
    source = (ROOT / "src/runtime/system/system_install.cpp").read_text()
    body = re.search(r"template <int Dim>\nstruct CoupledFrequencyMaximumKernel \{.*?\n\};", source, re.S).group()
    cpp = r"""
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/coupling/source/coupled_source_program.hpp>
#include <limits>
#include <cassert>
using namespace pops;
BODY
template<int D> void check() {
  Real data[24]{};
  FieldView<const Real,D> view{};
  view.data=data; view.ncomp=3; view.component_stride=8;
  Index<D> cell{};
  for(int a=0;a<D;++a) {view.origin[a]=-2;view.extents[a]=2;view.strides[a]=1<<a;cell[a]=-2;}
  data[16]=Real(3.25);
  runtime::program::hierarchy_tensor_detail::OriginalCandidateFiniteKernel<D,decltype(view)> finite{view,3};
  assert(finite(cell)==Real(0));
  data[16]=std::numeric_limits<Real>::quiet_NaN(); assert(finite(cell)==Real(1));
  data[16]=std::numeric_limits<Real>::infinity(); assert(finite(cell)==Real(1));
  data[16]=Real(3.25);
  CoupledFreqKernel<D> inner{};
  inner.n_in=1; inner.in[0]=view;inner.in_comp[0]=2;
  inner.n_const=1;inner.consts[0]=Real(0.5);
  inner.prog.len=3;inner.prog.op[0]=int(CsOp::PushReg);inner.prog.arg[0]=0;
  inner.prog.op[1]=int(CsOp::PushReg);inner.prog.arg[1]=1;inner.prog.op[2]=int(CsOp::Mul);
  CoupledFrequencyMaximumKernel<D> maximum{inner};
  assert(maximum(cell)==Real(1.625));
  cell[0]=-1;data[17]=Real(-4);assert(maximum(cell)==Real(-2));
  data[17]=std::numeric_limits<Real>::quiet_NaN();
  assert(maximum(cell)==std::numeric_limits<Real>::lowest());
  // Component zero is not substituted for selected component two.
  data[1]=Real(999);data[17]=Real(6);assert(maximum(cell)==Real(3));
}
template<int D> void range_check() {
  Real data[24]{};
  FieldView<const Real,D> view{};
  view.data=data;view.ncomp=3;view.component_stride=8;
  Index<D> lo{},hi{};
  for(int a=0;a<D;++a) {lo[a]=-2;hi[a]=-1;view.origin[a]=-2;view.extents[a]=2;view.strides[a]=1<<a;}
  const int last=(1<<D)-1;
  for(int i=0;i<=last;++i) data[16+i]=Real(i+1);
  runtime::program::hierarchy_tensor_detail::OriginalCandidateFiniteKernel<D,decltype(view)> finite{view,3};
  data[16+last]=std::numeric_limits<Real>::quiet_NaN();
  assert(for_each_cell_reduce_max(Box<D>{lo,lo},finite)==Real(0));
  assert(for_each_cell_reduce_max(Box<D>{lo,hi},finite)==Real(1));
  data[16+last]=Real(last+1);
  CoupledFreqKernel<D> inner{};
  inner.n_in=1;inner.in[0]=view;inner.in_comp[0]=2;
  inner.prog.len=1;inner.prog.op[0]=int(CsOp::PushReg);inner.prog.arg[0]=0;
  CoupledFrequencyMaximumKernel<D> maximum{inner};
  assert(for_each_cell_reduce_max(Box<D>{lo,lo},maximum)==Real(1));
  assert(for_each_cell_reduce_max(Box<D>{lo,hi},maximum)==Real(last+1));
}
int main(int argc,char** argv){
  Kokkos::initialize(argc,argv);
  check<1>();check<2>();check<3>();range_check<1>();range_check<2>();range_check<3>();
  Kokkos::finalize();
}
""".replace("BODY",body)
    path=tmp_path/"probe.cpp";path.write_text(cpp)
    binary=tmp_path/"probe"
    command=["/opt/homebrew/opt/llvm/bin/clang++","-std=c++20","-fopenmp","-ffp-contract=off","-fno-fast-math","-DPOPS_HAS_KOKKOS",f"-I{ROOT}/include",f"-I{PREFIX}/include",str(path),f"-L{PREFIX}/lib","-lkokkoscore",f"-Wl,-rpath,{PREFIX}/lib","-o",str(binary)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True)

def test_private_routes_use_named_callables():
    header=(ROOT/"include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp").read_text()
    source=(ROOT/"src/runtime/system/system_install.cpp").read_text()
    section=header.split("void validate_finite_original_candidate_() const {",1)[1].split("std::vector<field_type> accepted_publication_",1)[0]
    assert "[=] POPS_HD" not in section
    assert "grown_box()" in section and "OriginalCandidateFiniteKernel" in section
    section=source.split("std::function<Real()> maximum_frequency;",1)[1].split("install_prepared_coupling_operator(label",1)[0]
    assert "[=] POPS_HD" not in section
    assert "reference.box(local)" in section and "CoupledFrequencyMaximumKernel<Dim>{kernel}" in section
