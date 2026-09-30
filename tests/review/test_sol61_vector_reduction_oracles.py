"""Independent numerical witnesses; legacy methods are extracted unchanged for host checks.

No candidate dot_all implementation, native field storage, or MPI is exercised here.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def pairing(a, b, weights=None):
    weights = (1.0,) * len(a) if weights is None else weights
    return math.fsum(x * y * w for x, y, w in zip(a, b, weights, strict=True))


def rotate(q, i, j, theta):
    out = list(q)
    c, s = math.cos(theta), math.sin(theta)
    out[i], out[j] = c * q[i] - s * q[j], s * q[i] + c * q[j]
    return tuple(out)


@pytest.mark.parametrize("width", (1, 2, 3, 5))
def test_all_coordinate_pairing_permutation_oracle(width):
    a = tuple(float(i + 2) for i in range(width))
    b = tuple((-1.0) ** i * (2 * i + 1) for i in range(width))
    permutation = tuple(reversed(range(width)))
    assert pairing(a, b) == pairing(
        tuple(a[i] for i in permutation), tuple(b[i] for i in permutation)
    )
    assert pairing(a, a) > 0
    if width == 1:
        assert pairing(a, b) == a[0] * b[0]


def test_component_zero_can_change_under_rotation_or_hide_wrong_tail():
    q = (2.0, 3.0, 5.0, 7.0, 11.0)
    rotated = (-3.0, 2.0, 5.0, 7.0, 11.0)
    assert pairing(q, q) == pairing(rotated, rotated) == 208
    assert q[0] ** 2 == 4 and rotated[0] ** 2 == 9
    corrupt = (q[0], *(10 * x for x in q[1:]))
    assert corrupt[0] ** 2 == q[0] ** 2
    assert pairing(corrupt, corrupt) == 20404
    zero_first = (0.0, 3.0, 4.0)
    assert zero_first[0] ** 2 == 0 and pairing(zero_first, zero_first) == 25


@pytest.mark.parametrize("time", (0.0, 0.1, 0.3, 1.7))
def test_five_component_skew_rotation_and_relabeling(time):
    q = (2.0, 3.0, 5.0, 7.0, 11.0)
    # exp(t J), J[0,2]=-0.7; J[1,4]=-1.1, opposite entries positive.
    result = rotate(rotate(q, 0, 2, 0.7 * time), 1, 4, 1.1 * time)
    assert pairing(result, result) == pytest.approx(208, rel=2e-15)
    p = (4, 2, 0, 3, 1)
    relabeled = tuple(result[i] for i in p)
    assert pairing(relabeled, relabeled) == pairing(result, result)
    assert result[3] == q[3]
    if time:
        assert not math.isclose(result[0] ** 2, q[0] ** 2, rel_tol=1e-3)


def test_declared_gram_pairing_differs_from_raw_euclidean_coordinates():
    q, weights = (2.0, 3.0, 5.0), (1.0, 2.0, 7.0)
    z = tuple(math.sqrt(w) * x for x, w in zip(q, weights, strict=True))
    rotated = rotate(z, 0, 2, math.pi / 2)
    result = tuple(x / math.sqrt(w) for x, w in zip(rotated, weights, strict=True))
    assert pairing(result, result, weights) == pytest.approx(pairing(q, q, weights), rel=2e-15)
    assert not math.isclose(pairing(result, result), pairing(q, q), rel_tol=1e-12)


def test_owner_mask_and_measure_oracles_are_distinct():
    cells = ((2., 3.), (100., 200.), (1., 2.), (3., 4.), (math.nan, math.nan), (5., 6.))
    active, volumes = (True, False, True, True, False, True), (.5, .5, .25, .25, .25, .25)
    raw = math.fsum(pairing(q, q) for q, owned in zip(cells, active, strict=True) if owned)
    comp0 = math.fsum(q[0] ** 2 for q, owned in zip(cells, active, strict=True) if owned)
    physical = math.fsum(pairing(q, q) * v for q, owned, v in zip(cells, active, volumes, strict=True) if owned)
    assert (raw, comp0, physical) == (104, 39, 29.25)
    assert math.fsum(()) == 0  # Empty local rank remains a collective participant.


def test_legacy_public_ir_stays_component_zero_and_rejects_cross_block():
    from pops import Case
    from pops.frames import Cartesian2D
    from pops.physics import Model
    from pops.time import Program

    model = Model("vector_review", frame=Cartesian2D())
    q = model.state("U", components=("a", "b", "c"))
    case = Case("vector_review_case")
    left, right = (case.block(name, model, states=(q,)) for name in ("left", "right"))
    p = Program("vector_review_program")
    u, v = p.state(left[q]), p.state(right[q])
    assert p.dot(u.n, u.n).attrs == {"kind": "dot"}
    assert p.norm2(u.n).attrs == {"kind": "norm2"}
    assert p.norm_inf(u.n).attrs == {"kind": "norm_inf"}
    with pytest.raises(ValueError, match="same block"):
        p.dot(u.n, v.n)
    with pytest.raises(ValueError, match="State/RHS"):
        p.dot(u.n, 1.0)


def extract_methods(path):
    source = path.read_text()
    methods = []
    for name in ("norm2", "norm_inf", "dot"):
        start = source.index(f"Real {name}(int program_block,")
        opening = source.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        methods.append(source[start:end])
    return "\n".join(methods)


def test_actual_uniform_and_amr_legacy_reduction_methods_host(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "bounded host compiler is required for this independent probe"
    uniform = extract_methods(ROOT / "include/pops/runtime/program/program_context.hpp")
    amr = extract_methods(ROOT / "include/pops/runtime/program/amr_program_context_spatial_operations.inc")
    scaffold = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using Real=double; using ExecutionLane=int;
struct Field { std::vector<std::vector<double>> cells; };
std::string events; bool fail_local=false; int component=-1;
namespace pops {
double dot_active_local(const Field&a,const Field&b,int c,const Field*mask) {
  component=c; if(fail_local) throw std::runtime_error("injected local kernel fault");
  double sum=0; for(size_t i=0;i<a.cells.size();++i)
    if(!mask || mask->cells[i][0]) sum+=a.cells[i][c]*b.cells[i][c]; return sum;
}
double reduce_active_norm_inf_local(const Field&a,int c,const Field*mask) {
  component=c; if(fail_local) throw std::runtime_error("injected local kernel fault");
  double v=0; for(size_t i=0;i<a.cells.size();++i)
    if(!mask || mask->cells[i][0]) v=std::max(v,std::abs(a.cells[i][c])); return v;
}}
double all_reduce_sum(double x,const ExecutionLane&){events+='S';return x;}
double all_reduce_max(double x,const ExecutionLane&){events+='M';return x;}
struct Base {
 using field_type=Field; int lane=17; Field mask;
 std::vector<Field> levels,masks; bool missing_right=false;
 const ExecutionLane& prepared_execution_lane()const{return lane;}
 const Field* pointwise_active_mask(int,const Field&)const{return &mask;}
 void converge_owner_reduction_(std::exception_ptr e,const ExecutionLane&,const char*)const {
   events+='V'; if(e)std::rethrow_exception(e);
 }
 void require_same_field_contract_(const Field&a,const Field&b,const char*)const {
   if(a.cells.size()!=b.cells.size())throw std::invalid_argument("layout");
   for(size_t i=0;i<a.cells.size();++i)
     if(a.cells[i].size()!=b.cells[i].size())throw std::invalid_argument("width");
 }
 template<class F>void for_each_owner_active_level_(int,const Field&,const Field*right,F f)const {
   for(size_t i=0;i<levels.size();++i)
     f(levels[i],right&&!missing_right?&levels[i]:nullptr,&masks[i]);
 }
};
struct Uniform:Base { @UNIFORM@ };
struct AMR:Base { @AMR@ };
template<class C>void check(C&c,const Field&f) {
 events.clear();assert(c.dot(0,f,f)==39);assert(component==0 && events=="VS");
 events.clear();assert(c.norm2(0,f)==std::sqrt(39.));assert(component==0 && events=="VS");
 events.clear();assert(c.norm_inf(0,f)==5);assert(component==0 && events=="VM");
 fail_local=true;events.clear();bool refused=false;
 try{c.dot(0,f,f);}catch(const std::runtime_error&){refused=true;}
 assert(refused && events=="V");fail_local=false;
}
int main(){
 double nan=std::numeric_limits<double>::quiet_NaN();
 Field all{{{2,3},{100,200},{1,2},{3,4},{nan,nan},{5,6}}};
 Uniform u;u.mask={{{1},{0},{1},{1},{0},{1}}};check(u,all);
 AMR a;a.levels={{{{2,3},{100,200}}},{{{1,2},{3,4},{nan,nan},{5,6}}}};
 a.masks={{{{1},{0}}},{{{1},{1},{0},{1}}}};check(a,all);
 a.missing_right=true;events.clear();bool refused=false;
 try{a.dot(0,all,all);}catch(const std::logic_error&){refused=true;}assert(refused && events=="V");
 AMR empty;events.clear();assert(empty.dot(0,all,all)==0 && events=="VS");
 Field tail_nan{{{2,nan}}};Uniform tail;tail.mask={{{1}}};
 assert(tail.dot(0,tail_nan,tail_nan)==4);
 std::cout<<"11 host checks PASS; actual legacy methods; mocked storage/collectives\n";
}
'''.replace("@UNIFORM@", uniform).replace("@AMR@", amr)
    source, binary = tmp_path / "legacy.cpp", tmp_path / "legacy"
    source.write_text(scaffold)
    compiled = subprocess.run((compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)), capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run((str(binary),), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "11 host checks PASS" in result.stdout
