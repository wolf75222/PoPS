"""Actual FAC mismatch arithmetic against independent physical face conservation.

Host storage/index substitutions are explicit; no native tower, MPI or solve is claimed.
"""

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def faces(tmp_path_factory):
    folder = tmp_path_factory.mktemp("original-fac-flux-host")
    text = (ROOT / "include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp").read_text()
    start = text.index("template <int Dim>\nstruct FluxMismatchTransfer")
    stop = text.index("template <int Dim>\nvoid execute_flux_mismatches", start)
    kernel = text[start:stop]
    scaffold = r"""
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
using Real=double;
#define POPS_HD
template<int D>using Index=std::array<int,D>;
template<int D>struct Box{};
namespace pops::amr {template<int D>using RefinementRatio=std::array<int,D>;}
template<class T,int D>struct FieldView {
 T* data=nullptr;
 T& operator()(const Index<D>& p,int)const{
  int i=p[0];if constexpr(D==2)i=64*p[0]+p[1];return data[i];
 }
};
KERNEL
template<int D>void run(int side,bool cover){
 std::vector<double>parent(4096),fine(4096),residual(4096),covered(4096),pk(4096),fk(4096);
 auto at=[](const Index<D>&p){int i=p[0];if constexpr(D==2)i=64*p[0]+p[1];return i;};
 Index<D>coarse{},neighbor{};coarse[0]=side<0?3:12;if constexpr(D==2)coarse[1]=4;
 neighbor=coarse;neighbor[0]-=side;
 parent[at(coarse)]=1.1;parent[at(neighbor)]=2.7;
 pk[at(coarse)]=-.2;pk[at(neighbor)]=1.8;covered[at(coarse)]=cover?1:0;
 const double H=1./16.,transverse=D==1?1.:1./8.,coarse_measure=H*transverse;
 const int tangential=D==1?1:3;
 double fine_faces=0,fine_budget=0;
 for(int j=0;j<tangential;++j){
  Index<D>inner=coarse;inner[0]=2*coarse[0]+(side<0?2:-1);
  if constexpr(D==2)inner[1]=3*coarse[1]+j;
  auto ghost=inner;ghost[0]+=side;
  fine[at(inner)]=3.4+.2*j;fine[at(ghost)]=.9-.1*j;
  fk[at(inner)]=.7+.1*j;fk[at(ghost)]=-1.3+.15*j;
  const double face=.5*(fk[at(inner)]+fk[at(ghost)]);
  fine_faces+=face*(fine[at(ghost)]-fine[at(inner)]);
  fine_budget+=(coarse_measure/(2*tangential))*face*
    (fine[at(inner)]-fine[at(ghost)])/(H*H/4);
 }
 const double coarse_face=.5*(pk[at(coarse)]+pk[at(neighbor)])*
    (parent[at(coarse)]-parent[at(neighbor)]);
 const double before=coarse_face/(H*H);residual[at(coarse)]=before;
 FluxMismatchTransfer<D> op;
 op.parent={parent.data()};op.fine={fine.data()};op.residual={residual.data()};
 op.covered={covered.data()};op.parent_coefficient={pk.data()};op.fine_coefficient={fk.data()};
 op.ratio[0]=2;if constexpr(D==2)op.ratio[1]=3;
 op.child_side=side;op.inverse_spacing_squared=1/(H*H);
 op.fine_face_weight=2./tangential;op.sign=-1;op.arithmetic_average=true;
 op(coarse);
 std::cout<<"["<<D<<","<<side<<","<<(cover?1:0)<<","<<before<<","<<residual[at(coarse)]
  <<","<<fine_faces<<","<<coarse_measure<<","<<fine_budget<<"]";
}
int main(){std::cout<<std::setprecision(17)<<"[";bool comma=false;
 for(int side:{-1,1})for(bool covered:{false,true}){
  if(comma)std::cout<<",";comma=true;run<1>(side,covered);std::cout<<",";run<2>(side,covered);
 }std::cout<<"]\n";}
""".replace("KERNEL", kernel)
    cpp, binary = folder / "flux.cpp", folder / "flux"
    cpp.write_text(scaffold)
    subprocess.run(
        ["/usr/bin/clang++", "-std=c++20", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(subprocess.check_output([str(binary)], text=True))


@pytest.mark.parametrize("dimension,side", ((1, -1), (1, 1), (2, -1), (2, 1)))
def test_signed_arithmetic_crossflux_matches_physical_interface(faces, dimension, side):
    row = next(row for row in faces if row[:3] == [dimension, side, 0])
    _, _, _, _, after, fine_faces, measure, fine_budget = row
    # A=-div(D grad) replaces the coarse interface flux by summed fine flux.
    transverse_count = 1 if dimension == 1 else 3
    expected = 2 * fine_faces / (transverse_count * (1 / 16) ** 2)
    assert after == pytest.approx(expected, abs=2e-12)
    assert measure * after + fine_budget == pytest.approx(0, abs=2e-12)


@pytest.mark.parametrize("dimension,side", ((1, -1), (1, 1), (2, -1), (2, 1)))
def test_covered_parent_is_not_a_second_interface_dof(faces, dimension, side):
    row = next(row for row in faces if row[:3] == [dimension, side, 1])
    assert row[4] == row[3]
