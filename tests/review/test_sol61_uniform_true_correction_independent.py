"""Host execution of exact Uniform GMRES bodies; storage/Kokkos/MPI are substitutes."""

from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def method(text, signature):
    start = text.index(signature)
    begin = text.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def test_exact_uniform_gmres_true_correction_policy(tmp_path):
    source = (ROOT / "include/pops/numerics/elliptic/interface/field_newton_krylov.hpp").read_text()
    methods = "\n".join(
        method(source, signature)
        for signature in (
            "template <class JvpProvider>\n  LinearResult solve_linear_",
            "bool update_correction_(",
            "Real& h_(",
        )
    )
    code = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using Real=double; using field_type=std::vector<Real>;
struct ExecutionLane{}; namespace Kokkos { void fence(){} }
double all_reduce_sum(double x,const ExecutionLane&){return x;}
double dot_all_local(const field_type&a,const field_type&b){double s=0;for(int i=0;i<2;++i)s+=a[i]*b[i];return s;}
void scale(field_type&a,double x){for(auto&v:a)v*=x;}
void saxpy(field_type&a,double x,const field_type&b){for(int i=0;i<2;++i)a[i]+=x*b[i];}
void lincomb(field_type&a,double x,const field_type&b,double y,const field_type&c){for(int i=0;i<2;++i)a[i]=x*b[i]+y*c[i];}
struct Options{int restart=4,linear_max_iterations=4;};
struct Engine{
 Options options_; bool verify_true_correction_=false;
 field_type linear_residual_=field_type(2),correction_=field_type(2),image_=field_type(2),work_=field_type(2);
 std::vector<field_type>basis_=std::vector<field_type>(5,field_type(2));
 std::vector<double>hessenberg_=std::vector<double>(20),cosine_=std::vector<double>(4),sine_=std::vector<double>(4),rotated_rhs_=std::vector<double>(5),coefficients_=std::vector<double>(4);
 struct LinearResult{bool converged=false;int evaluations=0;};
 void copy_(const field_type&a,field_type&b,const ExecutionLane&){b=a;}
 template<class Fn>void local_phase_(const ExecutionLane&,Fn&&fn){fn();}
 double norm_(const field_type&a,const ExecutionLane&){return std::sqrt(dot_all_local(a,a));}
 static bool finite_(double a){return std::isfinite(a);}
METHODS
};
int main(){
 ExecutionLane lane; field_type q{0,0},rhs{3,4};
 // Actual normalized central FD of F_i(q)=q_i+q_i^3 at q=0.
 // It is homogeneous in direction but not additive, a permitted approximate JVP.
 auto jvp=[](const auto&,const field_type&v,field_type&out,int){
   double nv=std::sqrt(dot_all_local(v,v)),h=nv>0 ? 1/nv : 1;
   for(int i=0;i<2;++i){double p=h*v[i],n=-h*v[i];out[i]=.5/h*(p+p*p*p)-.5/h*(n+n*n*n);}
 };
 Engine legacy;auto old=legacy.solve_linear_(q,rhs,1e-10,jvp,0,lane);
 field_type image(2),error(2);jvp(q,legacy.correction_,image,0);lincomb(error,1,rhs,-1,image);
 double actual=legacy.norm_(error,lane);
 assert(old.converged && actual>1e-3);
 std::cout<<"legacy_projected_accept actual="<<actual<<" evaluations="<<old.evaluations<<"\n";
NEW_POLICY
}
"""
    assert "verify_true_correction_" in methods
    policy = r"""
 Engine guarded;guarded.verify_true_correction_=true;
 auto now=guarded.solve_linear_(q,rhs,1e-10,jvp,0,lane);
 jvp(q,guarded.correction_,image,0);lincomb(error,1,rhs,-1,image);
 double real=guarded.norm_(error,lane);
 assert(!now.converged || real<=1e-10);
 assert(now.evaluations>old.evaluations);
 std::cout<<"true_policy refused_or_true_residual actual="<<real<<" evaluations="<<now.evaluations<<"\n";
"""
    code = code.replace("METHODS", methods).replace("NEW_POLICY", policy)
    src = tmp_path / "uniform_actual_gmres.cpp"
    exe = tmp_path / "uniform_actual_gmres"
    src.write_text(code)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(src), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "true_policy refused_or_true_residual" in result.stdout
    assert "legacy_projected_accept" in result.stdout
    print(result.stdout)
