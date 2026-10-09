"""Real C++ string transports and public model emission; no PoPS Native run."""
from fractions import Fraction
from pathlib import Path
import subprocess
import os
import sys
import pytest
from pops.model import PhysicalDimension
from pops.codegen._native_units import canonical_unit_text, optional_unit_cpp, unit_c_string_cpp


def test_native_string_bytes_and_length_boundary(tmp_path):
    units = [PhysicalDimension(), PhysicalDimension((("温度\0é", Fraction(-2, 3)),)), "named température"]
    rows = [(unit_c_string_cpp(u), canonical_unit_text(u).encode()) for u in units]
    nul = "named\0unit"
    with pytest.raises(ValueError, match="length-aware"):
        unit_c_string_cpp(nul)
    source = tmp_path / "bytes.cpp"
    body = ['#include <string>', '#include <optional>', '#include <iostream>', 'int main(){']
    for expr, expected in rows:
        body += [f'{{std::string value={expr};', 'for(unsigned char c:value) std::cout << unsigned(c) << ","; std::cout << "\\n";}']
    body += [f'auto value={optional_unit_cpp(nul)};', 'for(unsigned char c:*value) std::cout << unsigned(c) << ",";', '}']
    source.write_text('\n'.join(body))
    exe=tmp_path/'bytes'
    subprocess.run(['clang++','-std=c++20',str(source),'-o',str(exe)],check=True,capture_output=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    expected=[','.join(map(str, b))+',' for _,b in rows]+[','.join(map(str,nul.encode()))+',']
    assert result.stdout.splitlines() == expected


def test_public_vp_and_thermal_full_model_tus(tmp_path):
    from tests.python.support.m19_vlasov_poisson_case import build as vp
    from tests.python.support.m19_thermal_consumed_case import build as thermal
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen._compile_emit import emit_cpp_native_loader
    root=Path(__file__).resolve().parents[2]
    # Explicit read-only host dependency: does not load or build PoPS Native.
    headers=Path(os.environ.get('POPS_TEST_HOST_SDK_INCLUDE', str(Path(sys.prefix)/'include')))
    assert (headers/'Kokkos_Core.hpp').is_file()
    for label, builder in [('vp',vp),('thermal',thermal)]:
        resolved=builder(tmp_path/label)
        for block in resolved.blocks:
            carrier,_=lower_and_validate(block.model,resolved_operations=block.resolved_operations)
            source=tmp_path/(label+'-'+block.name+'.cpp')
            source.write_text(emit_cpp_native_loader(carrier._m))
            command=['clang++','-std=c++20','-fsyntax-only','-Xpreprocessor','-fopenmp',
                     '-DPOPS_HAS_KOKKOS','-DPOPS_NATIVE_DIM=2','-I',str(root/'include'),'-I',str(headers),str(source)]
            result=subprocess.run(command,capture_output=True,text=True)
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("named_nul", [False, True])
def test_nonkinetic_public_field_flux_units(tmp_path, named_nul):
    from pops.model import Module, Rate, PhysicalSupport
    from pops.model.flux_waves import FluxWaveLaw
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen._compile_emit import emit_cpp_native_loader
    frame=Rectangle('fabric', (0,0), (1,1)).frame(Cartesian2D())
    module=Module('independent heat transport',frame=frame)
    support=PhysicalSupport((('y','height'),('x','width')))
    unit=PhysicalDimension(((("温度\0é" if named_nul else "温度é"),Fraction(2,3)),))
    state=module.state_space('enthalpy',('marker','heat'),support=support,units=(unit,unit),sampling='cell_average')
    field=module.field_space('conducted input',('temperature',),support=support,units=(unit,),sampling='cell')
    u=module.state_symbols(state); t=module.field_symbols(field)[0]
    flux=module.operator('thermal_flux',signature=(state,field)>>Rate(state),kind='grid_operator',
       expr={'x':(t*u[0],t*u[1]),'y':(2*t*u[0],2*t*u[1])},
       lowering={'flux_wave_law':FluxWaveLaw(state,{'x':(t,t),'y':(2*t,2*t)})})
    module.rate_operator('heat',state_space=module.state_handle(state),flux=True,fluxes=(flux,),default_flux=flux,sources=[])
    carrier,_=lower_and_validate(module)
    text=emit_cpp_native_loader(carrier._m)
    # Compile the actual requirement table separately from loader/Kokkos dependencies.
    start=text.index('  inline static constexpr std::array<pops::QualifiedProviderRequirement,')
    end=text.index('  }};',start)+len('  }};')
    table=text[start:end]
    source=tmp_path/'requirements.cpp'
    root=Path(__file__).resolve().parents[2]
    source.write_text('#include <pops/numerics/fv/flux_interfaces.hpp>\n#include <iostream>\nstruct Probe {\n'+table+'\n};\nint main(){std::cout << Probe::flux_provider_requirements[0].unit;}')
    exe=tmp_path/'requirements'
    result=subprocess.run(['clang++','-std=c++20','-Xpreprocessor','-fopenmp','-DPOPS_HAS_KOKKOS','-DPOPS_NATIVE_DIM=2','-I',str(root/'include'),'-I',str(Path(sys.prefix)/'include'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(exe)],check=True,capture_output=True).stdout == canonical_unit_text(unit).encode()
