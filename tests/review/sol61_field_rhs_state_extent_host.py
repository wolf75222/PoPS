"""Run the actual Native read-effect admission method in a small stdlib harness.

The closure table is supplied by the harness; this is not an MPI/solver/DAG test.
No PoPS extension is imported and no native runtime TU is compiled.
"""
from pathlib import Path
import hashlib
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
text = (root / "src/runtime/amr/amr_system.cpp").read_text()
start = text.index("  void validate_accepted_halo_field_state_reads() const {")
end = text.index("\n  void prepare_accepted_halo_field_dependencies", start)
method = text[start:end]
source = r'''
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
struct Rhs { unsigned read_contract_version=0; std::array<unsigned,2> state_read_cells{};
 std::string read_authority; std::string declared_input_identity; };
struct Binding { std::string identity; };
struct Plan { bool use_prepared_level_rhs=false; std::vector<std::vector<Rhs>> rhs_by_block;
 std::vector<Binding> providers; };
struct Hierarchy { std::vector<std::vector<std::vector<std::string>>> accepted_halo_boundary_fields; };
struct Fixture {
 Hierarchy hierarchy; Hierarchy* prepared_hierarchy=&hierarchy;
 std::map<std::string,Plan> field_plans;
 std::set<std::string> accepted_halo_field_order(const std::vector<std::string>& rows) const {
   return {rows.begin(), rows.end()}; }
METHOD
};
int main() {
 Fixture f; f.hierarchy.accepted_halo_boundary_fields={{{"needed"}}};
 f.field_plans["needed"]={false, {{{1,{0,0},"compiler.cell-state-ast@1","binding"}}}, {{"binding"}}};
 f.field_plans["unused"].use_prepared_level_rhs=true;
 f.validate_accepted_halo_field_state_reads();
 const auto good=f.field_plans.at("needed");
 int refusals=0;
 auto reject=[&] { try { f.validate_accepted_halo_field_state_reads(); }
   catch(const std::invalid_argument&) { ++refusals; return; }
   throw std::runtime_error("uncertified producer admitted"); };
 f.field_plans["needed"].rhs_by_block[0][0].read_contract_version=0; reject();
 f.field_plans["needed"]=good;
 f.field_plans["needed"].rhs_by_block[0][0].state_read_cells[1]=1; reject();
 f.field_plans["needed"]=good;
 f.field_plans["needed"].rhs_by_block[0][0].read_authority.clear(); reject();
 f.field_plans["needed"]=good;
 f.field_plans["needed"].providers.push_back({"opaque-InputAux"}); reject();
 f.field_plans["needed"]=good;
 f.field_plans["needed"].use_prepared_level_rhs=true; reject();
 if(refusals!=5) return 1;
 std::cout<<"PASS actual read-effect guard: direct valid-cell route, unused opaque Field, 5 refusals\n";
}
'''.replace("METHOD", method)
with tempfile.TemporaryDirectory(prefix="pops-rhs-reach-host-") as directory:
    cpp = Path(directory) / "guard.cpp"
    binary = Path(directory) / "guard"
    cpp.write_text(source)
    subprocess.run(["clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print("actual_guard_sha256=" + hashlib.sha256(method.encode()).hexdigest())
