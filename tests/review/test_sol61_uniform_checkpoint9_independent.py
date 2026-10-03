"""Independent Source API compatibility probe; no Native qualification."""
from pathlib import Path
import re,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]

def test_actual_public_observation_preserves_noarg_method_type(tmp_path):
    header=(ROOT/'include/pops/runtime/system.hpp').read_text()
    declaration=re.search(r'observe_accepted_state_storage\([^;]*;',header).group(0)
    # Exact public declaration from the real header; tiny explicit host seam only.
    code='#include <vector>\n#include <cstdint>\n#include <type_traits>\nstruct System { using R=std::vector<std::vector<std::uint8_t>>; R '+declaration+' };\nusing Method=System::R(System::*)()const;\nstatic_assert(std::is_same_v<decltype(&System::observe_accepted_state_storage),Method>);\n'
    source=tmp_path/'signature.cpp';source.write_text(code)
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler
    result=subprocess.run([compiler,'-std=c++20','-fsyntax-only',str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
