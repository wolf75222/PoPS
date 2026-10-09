"""Real Native codec and System method bodies; host storage/lane/fence substitutes."""

from pathlib import Path
import shutil
import subprocess
import sysconfig

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_pybind_closures_with_real_facade_headers(tmp_path):
    """Syntax only: real class interfaces/pybind bodies, no linking or runtime acceptance."""
    import pybind11

    compiler = shutil.which("clang++")
    assert compiler
    prefix = (
        "#include <pops/runtime/system.hpp>\n"
        "#include <pops/runtime/amr_system.hpp>\n"
        "#include <pybind11/pybind11.h>\nnamespace py=pybind11;\n"
        "using System=pops::System<pops::kNativeDimension>;\n"
        "using AmrSystem=pops::AmrSystem<pops::kNativeDimension>;\n"
    )
    for family, klass in (("system", "System"), ("amr", "AmrSystem")):
        source = (ROOT / ("python/bindings/core/init/init_" + family + ".cpp")).read_text()
        start = source.rfind(".def(", 0, source.index('"_checkpoint_program_diagnostics"'))
        end = source.index('.def("_accepted_balance_terms"', start)
        body = source[start:end]
        assert "&payload" in body and "std::function" not in body
        path = tmp_path / (family + "_bindings.cpp")
        path.write_text(prefix + "void bind(py::class_<" + klass + ">& cls){cls" + body + ";}")
        subprocess.run(
            [
                compiler,
                "-std=c++23",
                "-DPOPS_NATIVE_DIM=2",
                "-DPOPS_HAS_KOKKOS=1",
                "-Xpreprocessor",
                "-fopenmp",
                "-I" + str(ROOT / "include"),
                "-I/Users/romaindespoulain/miniforge3/envs/pops-api040/include",
                "-I" + sysconfig.get_paths()["include"],
                "-I" + pybind11.get_include(),
                "-fsyntax-only",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def _method(source, signature):
    start = source.index(signature)
    left = source.index("{", start)
    depth = 1
    right = left + 1
    while depth:
        depth += (source[right] == "{") - (source[right] == "}")
        right += 1
    return "template <int Dim>\n" + source[start:right]


@pytest.mark.parametrize("family", ("system", "amr"))
def test_native_codec_and_actual_system_atomic_restore(tmp_path, family):
    source = (ROOT / "src/runtime/system/system_program.cpp").read_text()
    methods = "\n".join(
        _method(source, signature)
        for signature in (
            "std::vector<std::uint8_t> System<Dim>::checkpoint_program_diagnostics() const",
            "void System<Dim>::validate_checkpoint_program_diagnostics(",
            "void System<Dim>::restore_checkpoint_program_diagnostics(",
        )
    )
    code = r"""
#include <pops/runtime/program/program_diagnostics_checkpoint.hpp>
#include <atomic>
#include <cassert>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
using pops::Real; using pops::RealBits;
using Map=std::map<std::string,Real>;
bool remote_failure=false,fence_failure=false;int votes=0,fences=0;
namespace Kokkos {void fence(){++fences;if(fence_failure)throw std::runtime_error("fence");}}
namespace pops {
struct ExecutionLane{int rank()const{return 0;}int size()const{return 1;}};
void collectively_rethrow_exception(std::exception_ptr e,const ExecutionLane&,const char*){
 ++votes;if(e)std::rethrow_exception(e);if(remote_failure)throw std::runtime_error("remote");
}
struct Impl{struct Program{Map diagnostics_;}program_;bool external_restart_transaction_=false,external_step_transaction_committed_=false;int depth=0;};
struct Authority{std::atomic<int>pending{0};};
template<int Dim>struct System{
 std::unique_ptr<Impl>p_=std::make_unique<Impl>();
 std::unique_ptr<Authority>solve_outcome_authority_=std::make_unique<Authority>();
 ExecutionLane lane;
 int step_transaction_depth()const{return p_->depth;}
 const ExecutionLane&prepared_boundary_execution_lane()const{return lane;}
 std::vector<std::uint8_t> checkpoint_program_diagnostics()const;
 void validate_checkpoint_program_diagnostics(std::span<const std::uint8_t>)const;
 void restore_checkpoint_program_diagnostics(std::span<const std::uint8_t> (*)(const void*),const void*);
};
METHODS
}
using pops::runtime::program::checkpoint_program_diagnostics;
using pops::runtime::program::read_program_diagnostics_checkpoint;
bool same(const Map&a,const Map&b){
 if(a.size()!=b.size())return false;auto x=a.begin(),y=b.begin();
 for(;x!=a.end();++x,++y)if(x->first!=y->first||std::bit_cast<RealBits>(x->second)!=std::bit_cast<RealBits>(y->second))return false;
 return true;
}
template<class F>void refuses(F&&fn){bool bad=false;try{fn();}catch(...){bad=true;}assert(bad);}
void set_word(std::vector<std::uint8_t>&x,std::size_t p,std::uint64_t v){for(unsigned i=0;i<8;++i)x[p+i]=static_cast<std::uint8_t>(v>>(8*i));}
int main(){
 Map initial{{"",std::bit_cast<Real>(RealBits{0})},{std::string("x\0y",3),std::bit_cast<Real>(RealBits{1})},{std::string("\xff",1),std::bit_cast<Real>(RealBits{1}<<(sizeof(RealBits)*8-1))}};
 if constexpr(sizeof(RealBits)==8){initial.emplace("nan",std::bit_cast<Real>(static_cast<RealBits>(0x7ff8000000001234ULL)));initial.emplace("infinity",std::bit_cast<Real>(static_cast<RealBits>(0x7ff0000000000000ULL)));}
 else{initial.emplace("nan",std::bit_cast<Real>(static_cast<RealBits>(0x7fc01234ULL)));}
 const auto encoded=checkpoint_program_diagnostics(initial);
 assert(same(initial,read_program_diagnostics_checkpoint(encoded)));
 assert(checkpoint_program_diagnostics(read_program_diagnostics_checkpoint(encoded))==encoded);
 for(std::size_t n=0;n<encoded.size();++n)refuses([&]{read_program_diagnostics_checkpoint(std::span(encoded).first(n));});
 auto corrupt=encoded;corrupt.push_back(0);refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;corrupt[0]='x';refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;set_word(corrupt,8,sizeof(RealBits)==8?32:64);refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;set_word(corrupt,32,~std::uint64_t{0});refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;set_word(corrupt,40,~std::uint64_t{0});refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;set_word(corrupt,16,1);refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 corrupt=encoded;set_word(corrupt,24,2);refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 assert(same(initial,read_program_diagnostics_checkpoint(checkpoint_program_diagnostics(initial,1,2),1,2)));
 refuses([&]{read_program_diagnostics_checkpoint(checkpoint_program_diagnostics(initial,1,2));});
 Map bad{{"pops.balance-term.bad",1}};refuses([&]{checkpoint_program_diagnostics(bad);});
 // Two encoded empty names provide a canonical duplicate attack without undefined parsing.
 corrupt=checkpoint_program_diagnostics(Map{{"",1}});set_word(corrupt,32,2);corrupt.insert(corrupt.end(),16,0);
 refuses([&]{read_program_diagnostics_checkpoint(corrupt);});
 auto unordered=checkpoint_program_diagnostics(Map{{"a",1},{"b",2}});unordered[48]='z';refuses([&]{read_program_diagnostics_checkpoint(unordered);});
 pops::System<1>system;system.p_->program_.diagnostics_=initial;
 assert(system.checkpoint_program_diagnostics()==encoded);
 system.p_->depth=1;refuses([&]{system.checkpoint_program_diagnostics();});system.p_->depth=0;
 system.solve_outcome_authority_->pending=1;refuses([&]{system.checkpoint_program_diagnostics();});system.solve_outcome_authority_->pending=0;
 Map replacement{{"different",std::bit_cast<Real>(RealBits{1})}};const auto good=checkpoint_program_diagnostics(replacement);
 auto apply=[&](const auto&image){system.restore_checkpoint_program_diagnostics(+[](const void* p)->std::span<const std::uint8_t>{return *static_cast<const std::vector<std::uint8_t>*>(p);},&image);};
 refuses([&]{apply(good);});assert(same(initial,system.p_->program_.diagnostics_));
 system.p_->external_restart_transaction_=true;
 refuses([&]{system.restore_checkpoint_program_diagnostics(nullptr,nullptr);});assert(same(initial,system.p_->program_.diagnostics_));
 refuses([&]{apply(corrupt);});assert(same(initial,system.p_->program_.diagnostics_));
 remote_failure=true;refuses([&]{apply(good);});remote_failure=false;assert(same(initial,system.p_->program_.diagnostics_));
 fence_failure=true;refuses([&]{apply(good);});fence_failure=false;assert(same(initial,system.p_->program_.diagnostics_));
 refuses([&]{system.restore_checkpoint_program_diagnostics(+[](const void*)->std::span<const std::uint8_t>{throw std::runtime_error("producer");},nullptr);});assert(same(initial,system.p_->program_.diagnostics_));
 system.solve_outcome_authority_->pending=1;refuses([&]{apply(good);});system.solve_outcome_authority_->pending=0;
 system.p_->external_step_transaction_committed_=true;refuses([&]{apply(good);});system.p_->external_step_transaction_committed_=false;
 apply(good);assert(same(replacement,system.p_->program_.diagnostics_));
 apply(std::vector<std::uint8_t>{});assert(system.p_->program_.diagnostics_.empty());
 assert(votes>=8&&fences>=8);
 std::cout<<"codec_native_width="<<sizeof(RealBits)*8<<" votes="<<votes<<" fences="<<fences<<"\n";
}
""".replace("METHODS", methods)
    if family == "amr":
        amr_source = (ROOT / "src/runtime/amr/amr_system.cpp").read_text()
        amr_methods = "\n".join(
            _method(amr_source, signature.replace("System<Dim>", "AmrSystem<Dim>"))
            for signature in (
                "std::vector<std::uint8_t> System<Dim>::checkpoint_program_diagnostics() const",
                "void System<Dim>::validate_checkpoint_program_diagnostics(",
                "void System<Dim>::restore_checkpoint_program_diagnostics(",
            )
        )
        code = (
            code.replace(methods, amr_methods)
            .replace("struct System{", "struct AmrSystem{")
            .replace("pops::System<1>", "pops::AmrSystem<1>")
        )
        code = (
            code.replace("program_;", "program;")
            .replace("program_.", "program.")
            .replace("external_restart_transaction_", "restart_transaction")
            .replace("external_step_transaction_committed_", "restart_transaction_committed")
        )
        code = code.replace(
            "int depth=0;};",
            "int depth=0; ExecutionLane lane; const ExecutionLane& require_package_assembly_lane()const{return lane;}};",
        )
        code = code.replace(
            " system.solve_outcome_authority_->pending=1;refuses([&]{system.checkpoint_program_diagnostics();});system.solve_outcome_authority_->pending=0;",
            "",
        )
        code = code.replace(
            " system.solve_outcome_authority_->pending=1;refuses([&]{apply(good);});system.solve_outcome_authority_->pending=0;",
            "",
        )
        code = code.replace("assert(votes>=8&&fences>=8)", "assert(votes>=7&&fences>=7)")
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler
    src = tmp_path / "diagnostic_actual.cpp"
    src.write_text(code)
    for real in ("double", "float"):
        exe = tmp_path / ("diagnostic_" + real)
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O0",
                "-DPOPS_REAL_TYPE=" + real,
                "-I" + str(ROOT / "include"),
                "-I/Users/romaindespoulain/miniforge3/envs/pops-api040/include",
                str(src),
                "-o",
                str(exe),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        print(result.stdout)
