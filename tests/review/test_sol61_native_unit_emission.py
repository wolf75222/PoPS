"""Source C++ emission uses canonical units, not Python dictionary literals."""
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import shutil
import pytest
from pops.model import PhysicalDimension
from pops.codegen._native_units import optional_unit_cpp

@pytest.mark.parametrize("unit", [None, 'kg" / s\n', PhysicalDimension(), PhysicalDimension((("length",Fraction(1,2)),("time",Fraction(-2))))])
def test_optional_unit_roundtrip_and_actual_cpp(unit,tmp_path):
    cpp=optional_unit_cpp(unit)
    if unit is None: assert cpp=="std::nullopt"
    else:
        literal=cpp[len("std::optional<std::string>{"):-1]
        text=json.loads(literal)
        assert text == unit if type(unit) is str else json.loads(text)==unit.to_data()
        if isinstance(unit,PhysicalDimension): assert optional_unit_cpp(unit.to_data())==cpp
    source=tmp_path/"units.cpp"
    source.write_text('#include <optional>\n#include <string>\nint main(){ auto unit = '+cpp+'; (void)unit; }\n')
    result=subprocess.run([shutil.which("clang++") or "c++","-std=c++20","-fsyntax-only",str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_true_resolved_model_provider_routes_syntax(tmp_path):
    from tests.python.support.m19_thermal_consumed_case import build
    from pops.codegen._compiler_lowering import require_compiler_lowering
    resolved=build(tmp_path)
    root=Path(__file__).resolve().parents[2]
    deps=Path('/Users/romaindespoulain/miniforge3/envs/pops-api040-initial14/include')
    assert (deps/'Kokkos_Core.hpp').is_file()
    for i,block in enumerate(resolved.blocks):
        emitter=require_compiler_lowering(block.model).emit_model
        source=emitter.__pops_native_loader_source__(name='ActualUnitModel'+str(i),target='system',consumer_owner_qid=block.instance_owner_qid)
        path=tmp_path/('model-'+str(i)+'.cpp');path.write_text(source)
        command=[shutil.which('clang++') or 'c++','-std=c++20','-fsyntax-only','-DPOPS_NATIVE_DIM=2','-DPOPS_HAS_KOKKOS','-DPOPS_HAS_MPI','-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI','-Xpreprocessor','-fopenmp','-I'+str(root/'include'),'-I'+str(deps),str(path)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (tmp_path/('model-'+str(i)+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,result.stderr[-5000:]


def test_noncanonical_or_opaque_units_refused():
    with pytest.raises(ValueError):
        optional_unit_cpp({"kind":"physical_dimension","powers":[["length",2,4]]})
    for unit in (False, 7, object(), ""):
        with pytest.raises(TypeError): optional_unit_cpp(unit)


@pytest.mark.parametrize("text", ["🌍", "μm", 'quote"and\\slash', "zero\x00tail", "control\x01"+"7", "tab\tline\n"])
def test_host_optional_string_preserves_all_utf8_bytes(text,tmp_path):
    source=tmp_path/"utf8.cpp"; binary=tmp_path/"utf8"
    source.write_text('#include <optional>\n#include <string>\n#include <cstdio>\nint main(){ auto unit = '+optional_unit_cpp(text)+'; for(unsigned char byte:*unit) std::printf("%02x",unsigned(byte)); }\n')
    subprocess.run([shutil.which("clang++") or "c++","-std=c++20",str(source),"-o",str(binary)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    assert result.stdout == text.encode("utf-8").hex()

def test_ascii_legacy_literals_remain_byte_identical():
    for text in ("kg/m^2", 'quote"and\\slash', "line\n"):
        assert optional_unit_cpp(text)=="std::optional<std::string>{%s}" % json.dumps(text)
