"""Executed joint species/energy face primitive with an explicit collective constraint."""
import ctypes
import subprocess

import numpy as np
import pytest
from pops._ir.expr import Var
from pops._ir.application import ApplicationContext
from pops.codegen.native_constitutive import emit_native_constitutive
from pops.model import FieldSpace, Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent


@pytest.mark.compiler
@pytest.mark.native_loader
def test_joint_face_species_energy_constraint_and_distinct_contexts_execute(tmp_path, record_property):
    from pathlib import Path
    from pops.codegen.toolchain import pops_loader_build_flags, _probe_cxx_std, loader_cxx_std
    from pops.native_components import compiler_include_roots, verify_prepared_native_dependencies

    external = tmp_path / "external"
    external.mkdir()
    (external / "law.hpp").write_text(r'''#pragma once
#include <pops/core/model/native_call.hpp>
#include <atomic>
#include <cmath>
namespace species_energy {
static inline std::atomic<long> calls{0};
// This immutable native law declares and enforces its collective admissible set.
constexpr double mass_fraction_sum = 1;
constexpr double constraint_tolerance = 1e-12;
inline pops::NativeCallResult<3> evaluate(double y1,double y2,double g1,double g2,double gT) {
  ++calls;
  if (std::abs(y1+y2-mass_fraction_sum)>constraint_tolerance)
    return pops::NativeCallResult<3>::rejected(701);
  const double raw1=-g1, raw2=-2*g2;
  const double correction=raw1+raw2;
  const double j1=raw1-y1*correction, j2=raw2-y2*correction;
  if (std::abs(j1+j2)>constraint_tolerance)
    return pops::NativeCallResult<3>::rejected(702);
  pops::NativeCallResult<3> result;
  result.status=pops::EvaluationStatus::kOk;
  result.values={j1,j2,3*j1+7*j2-2*gT};
  return result;
}
}
''')
    component = PreparedNativeComponent.header_only("species.energy.collective", include_root=external,
                                                   entry_headers=("law.hpp",))
    fractions = FieldSpace("mass_fractions", components=("y1", "y2"))
    gradients = FieldSpace("face_gradients", components=("dy1", "dy2", "dT"))
    function = NativeFunction(component, "species_energy::evaluate",
        Signature((fractions, gradients), ProductSpace({
            "species": FieldSpace("species_flux", components=("j1", "j2")),
            "energy": FieldSpace("energy_flux", components=("heat",))})),
        domains=tuple(NativeInputDomain(0, i, lower=0, upper=1) for i in (0, 1)),
        effects=("fallible", "diagnostic_counter"))
    names = tuple(Var(name, "cons") for name in ("y1", "y2", "g1", "g2", "gT"))
    call = function(names[:2], names[2:], occurrence="joint-face-law",
                    context=ApplicationContext(sampling="face", location="adjacent-cell-face"))
    emitted = emit_native_constitutive((*call.species, *call.energy, call.species[0]+call.species[0]))
    body = "\n".join(emitted.lines)
    assert body.count("species_energy::evaluate(") == 1
    source = r'''#include <law.hpp>
extern "C" int face(const double* left,const double* right,double distance,double* output,unsigned* reason) {
  const double y1=.5*(left[0]+right[0]), y2=.5*(left[1]+right[1]);
  const double g1=(right[0]-left[0])/distance, g2=(right[1]-left[1])/distance;
  const double gT=(right[2]-left[2])/distance;
'''+body+"\n*reason=%s; if (%s!=0) return %s;\n" % (emitted.reason, emitted.status, emitted.status)
    source += "\n".join("output[%d]=%s;" % (i, value) for i, value in enumerate(emitted.values))
    source += '\nreturn 0;\n}\nextern "C" long count() { return species_energy::calls.load(); }\n'
    source_path, binary, depfile = tmp_path / "face.cpp", tmp_path / "face.so", tmp_path / "face.d"
    source_path.write_text(source)
    # Use the exact selected package headers, not a different checkout ABI.
    import pops
    include = Path(pops.__file__).resolve().parent / "include"
    staged = component.stage_verified(tmp_path / "stage")
    compiler, cflags, lflags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    built = subprocess.run([compiler, "-std="+standard, "-shared", "-fPIC", "-O2", *cflags,
        "-I", str(include), "-I", staged, "-MMD", "-MF", str(depfile), str(source_path),
        "-o", str(binary), *lflags], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    verify_prepared_native_dependencies(depfile, generated_source=source_path,
        pops_include_root=include, staged_components=((component, staged),),
        toolchain_include_roots=compiler_include_roots(cflags))
    native = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    native.face.argtypes = (pointer, pointer, ctypes.c_double, pointer, ctypes.POINTER(ctypes.c_uint))
    native.face.restype = ctypes.c_int
    native.count.restype = ctypes.c_long
    output, reason = (ctypes.c_double*4)(17,17,17,17), ctypes.c_uint(0)
    for left, right in (((.2,.8,3.),(.3,.7,3.2)), ((.3,.7,3.2),(.4,.6,3.5))):
        assert native.face((ctypes.c_double*3)(*left),(ctypes.c_double*3)(*right),.1,
                           output,ctypes.byref(reason)) == 0
        y = .5*(np.asarray(left[:2])+right[:2])
        gradients = (np.asarray(right)-left)/.1
        raw = -np.asarray((1.,2.))*gradients[:2]
        j = raw-y*sum(raw)
        np.testing.assert_allclose(tuple(output), (*j,3*j[0]+7*j[1]-2*gradients[2],2*j[0]),
                                   rtol=0,atol=1e-12)
        assert abs(output[0]+output[1]) < 1e-12
    assert native.count() == 2  # two adjacent face contexts, one whole call each
    before = tuple(output)
    assert native.face((ctypes.c_double*3)(.2,.7,3.),(ctypes.c_double*3)(.3,.6,3.2),.1,
                       output,ctypes.byref(reason)) == 2
    assert reason.value == 701 and tuple(output) == before and native.count() == 3
    record_property("output_widths", [2,1])
    record_property("actual_joint_calls_valid_contexts", 2)
    record_property("repeated_projected_contribution", "two times j1; no extra native call")
    record_property("collective_constraint", "y1+y2=1 and J1+J2=0, tolerance1e-12")
    record_property("program_binary_bytes", binary.stat().st_size)
