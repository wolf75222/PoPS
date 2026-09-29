"""Integral identifiers retain exact Unicode bytes across the generated C++ boundary."""
import shutil
import subprocess

import pops
import pytest

from pops.codegen.program_integral_transfers import emit_integral_declarations


def test_embedded_nul_is_refused_before_it_can_truncate_the_native_identity():
    program = pops.Program("names")
    before = program._serialize()
    with pytest.raises(ValueError, match="without NUL"):
        program.integral_state("q\0tail", initial=.7)
    assert program._serialize() == before


def test_valid_unicode_names_reach_native_declaration_with_exact_utf8_bytes(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++20 compiler is needed for the declaration boundary test")
    program = pops.Program("unicode")
    names = ("charge", "électricité", "電荷", "λ")
    handles = {name: program.integral_state(name, initial=.7) for name in names}
    declarations = "\n".join(emit_integral_declarations(program))
    source = tmp_path / "names.cpp"
    source.write_text(
        "#include <string>\n#include <vector>\n#include <cstdio>\n"
        "namespace pops { using Real=double; }\n"
        "struct C { std::vector<std::string> names; "
        "void declare_integral_state(const std::string& name,double){names.push_back(name);} };\n"
        "int main(){ C ctx;\n" + declarations + "\n"
        "for(const auto& name:ctx.names){for(unsigned char c:name) std::printf(\"%02x\",c);"
        "std::puts(\"\");} }\n")
    executable = tmp_path / "names"
    subprocess.run([compiler, "-std=c++20", "-O2", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [handles[name].identity.encode().hex() for name in sorted(names)]
