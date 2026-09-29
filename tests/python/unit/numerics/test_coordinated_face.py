"""Version-one public face authority, unchanged old paths, and host emission."""
import ctypes
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import re
from types import SimpleNamespace

import numpy as np
import pytest
import pops
from pops.math import minimum, where
from pops.numerics import CoordinatedFace, CoordinatedFiniteVolume, FaceBalance
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_emit_coordinated_face import emit_coordinated_face_members
from tests.python.support.symbolic_path_case import declarations


ROOT = Path(__file__).resolve().parents[4]


def _load(name):
    path = ROOT / "examples/migration/scientific" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _face(body):
    model, state, flux, product, path = declarations()
    return model, CoordinatedFace(flux=flux, product=product, frame=path.frame, body=body)


@pytest.mark.parametrize("order", [("h", "q", "z"), ("z", "h", "q")])
def test_public_hydrostatic_case_retains_complete_method_identity_through_loader(order):
    case, layout, _ = _load("api040_m07_saint_venant").build_case(order=order)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    source = _emit_bricks(emitter._m)[1]
    method = block.numerics.rates[0].method
    assert method.native_identity() == emitter._m._path_conservative["identity"]
    assert method.native_identity() in source
    assert "coordinated_face_contract_version = 1" in source
    assert "coordinated_face<Axis>(U, U)" in source
    assert "path_integral(" not in source
    assert str(method.runtime_spatial().flux) == "coordinated_face:v1:" + method.native_identity()
    assert method.to_data()["path"]["side_sign"] == "already_signed_cell_rhs"
    assert method.to_data()["riemann"]["native_id"] == "pops::CoordinatedFaceFlux"
    observations = [operation for operation in block.resolved_operations.operations
                    if operation.native_route == "program:path_conservative_rhs"
                    and "program_evaluation" in operation.guarantees]
    assert observations
    from pops.identity import canonical_bytes
    from pops.identity.semantic import semantic_value
    def canonical(value):
        return canonical_bytes(semantic_value(value, where="test numerical authority"))
    assert all(canonical(operation.guarantees["numerical_method"]) == canonical(method.to_data())
               for operation in observations)


def test_rejects_foreign_leaves_and_nonzero_conservative_side():
    from pops._ir.expr import Var
    with pytest.raises(ValueError, match="free variable"):
        _face(lambda left, right, axis: FaceBalance((Var("foreign", "cons"), 0.),
                                                   (0., 0.), (0., 0.), 1.))
    with pytest.raises(ValueError, match="literal zero side"):
        _face(lambda left, right, axis: FaceBalance(tuple(left), (0., 1.), (0., 0.), 1.))


def test_complete_balance_cannot_be_omitted_or_counted_twice():
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan
    model, face = _face(lambda left, right, axis:
        FaceBalance(tuple(left), (0., 0.), (0., 0.), 1.))
    state, flux, product = face.product.state, face.flux, face.product
    rate = model.rate("balance", equation=ddt(state) == -div(flux)-product)
    method = CoordinatedFiniteVolume(face=face)
    assert method.validate_rate_contract(model.rate_contract(rate))
    with pytest.raises(ValueError, match="exact flux/product/state"):
        method.validate_rate_contract(model.rate_contract(rate.select(flux)))
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    plan.rates.add(rate.select(product), method)
    with pytest.raises(ValueError, match="duplicate physical balance occurrence"):
        plan.validate_for(model)


def _compile_members(tmp_path, face):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    size = len(face.product.state.components)
    members = "\n".join(emit_coordinated_face_members(SimpleNamespace(
        n_vars=size, _path_conservative={"kernel": face.native_kernel()})))
    source = (
        "#include <array>\n#include <cmath>\n#include <limits>\n"
        "#include <pops/numerics/fv/path_result.hpp>\n"
        "#include <pops/numerics/spatial/nd/state_conversion.hpp>\n"
        "namespace Kokkos { using std::fmax; using std::fmin; using std::sqrt; using std::abs; }\n"
        "struct Model {\n"
        f"using State=std::array<double,{size}>; static constexpr int n_vars={size},dimension=2;\n"
        "pops::nd::StateConversionStatus admissibility(const State&) const {\n"
        "return pops::nd::StateConversionStatus::Success; }\n"
        + members + "\n};\n"
        'extern "C" int evaluate(const double* l,const double* r,double* out) {\n'
        "Model::State left{},right{};\n"
        f"for(int k=0;k<{size};++k) {{left[k]=l[k];right[k]=r[k];}}\n"
        "auto result=Model{}.coordinated_face<0>(left,right);\n"
        f"for(int k=0;k<{size};++k){{ out[k]=result.conservative_flux.values[k];"
        f"out[k+{size}]=result.left_ncp.values[k];out[k+{2*size}]=result.right_ncp.values[k];}}\n"
        f"out[{3*size}]=result.speed_bound; return static_cast<int>(result.status);}}\n")
    cpp, library = tmp_path/"face.cpp", tmp_path/"face.so"
    cpp.write_text(source)
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-fno-fast-math",
        "-ffp-contract=off", "-shared", "-fPIC", "-I"+str(ROOT / "include"),
        str(cpp), "-o", str(library)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    function = ctypes.CDLL(str(library)).evaluate
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes, function.restype = [ptr, ptr, ptr], ctypes.c_int
    return function


def test_host_face_keeps_asymmetric_signs_and_refuses_active_invalid_tuple_atomically(tmp_path):
    def body(left, right, axis):
        # v is declared conservative; u has asymmetric sources. NaN in a selected
        # expression must not be masked by an outer minimum.
        invalid = where(left[0] > 0., lambda: 1./(left[0]-left[0]), lambda: -2.)
        return FaceBalance((minimum(invalid, 5.), left[1]),
                           (-.3*(right[0]-left[0]), 0.),
                           (-.7*(right[0]-left[0]), 0.), 2.)
    _, face = _face(body)
    evaluate = _compile_members(tmp_path, face)
    out = np.full(7, 999., dtype=np.float64)
    assert evaluate(np.array([-1., 3.]), np.array([2., 4.]), out) == 0
    np.testing.assert_allclose(out, [-2., 3., -.9, 0., -2.1, 0., 2.], atol=1.e-15)
    assert evaluate(np.array([1., 3.]), np.array([2., 4.]), out) != 0
    np.testing.assert_array_equal(out, 0.)
    assert evaluate(np.array([-1., np.nan]), np.array([2., 4.]), out) != 0
    np.testing.assert_array_equal(out, 0.)


def test_negative_numerical_bound_cannot_publish_finite_outputs(tmp_path):
    _, face = _face(lambda left, right, axis:
        FaceBalance(tuple(left), (0., 0.), (0., 0.), -1.))
    evaluate = _compile_members(tmp_path, face)
    out = np.full(7, 999., dtype=np.float64)
    assert evaluate(np.array([1., 2.]), np.array([1., 2.]), out) != 0
    np.testing.assert_array_equal(out, 0.)


@pytest.mark.parametrize("foreign", [False, True])
def test_runtime_capture_is_qualified_and_lowered_without_baking_its_value(foreign):
    from pops.params import RuntimeParam
    from pops.math import ddt, div
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.lib.time import ForwardEuler
    from pops.time import FixedDt
    from pops.numerics import DiscretizationPlan
    model, state, flux, product, path = declarations()
    owner = declarations()[0] if foreign else model
    coefficient = owner.value(owner.param(RuntimeParam("face_coefficient", default=2.)))
    face = CoordinatedFace(flux=flux, product=product, frame=path.frame,
        body=lambda left, right, axis: FaceBalance(tuple(coefficient*q for q in left),
                                                   (0., 0.), (0., 0.), coefficient))
    rate = model.rate("balance", equation=ddt(state) == -div(flux)-product)
    plan = DiscretizationPlan()
    plan.rates.add(rate, CoordinatedFiniteVolume(face=face))
    case = pops.Case("qualified_face_capture")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(.001))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(4,4),
                                   periodic=PeriodicAxes(path.frame.axes)))
    def emit():
        resolved = pops.resolve(pops.validate(case), layout=layout)
        native = resolved.blocks[0]
        emitter, _ = lower_and_validate(native.model, state_space=native.state_spaces[0],
            resolved_operations=native.resolved_operations, numerics=native.numerics)
        return _emit_bricks(emitter._m)[1]
    if foreign:
        with pytest.raises((ValueError, KeyError), match="(?i)(param|model|declaration|owner|reference)"):
            emit()
    else:
        source = emit()
        assert "RuntimeParams params" in source and "params.get(" in source


@pytest.mark.parametrize("coordinated", [False, True])
def test_actual_generated_composite_selects_and_executes_its_exact_policy(tmp_path, coordinated):
    """Compile the real composite/adapter, including the legacy version-zero trait."""
    prefix = Path(sys.prefix)
    if not (prefix / "include/Kokkos_MathematicalFunctions.hpp").is_file():
        pytest.skip("host generated-brick check requires the configured Kokkos SDK")
    if coordinated:
        case, layout, _ = _load("api040_m07_saint_venant").build_case()
        dimension = 1
    else:
        from tests.python.support.symbolic_path_case import make_case
        case, layout = make_case()
        dimension = 2
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    source = _emit_bricks(emitter._m)[1]
    hyp, ell = re.search(r"struct (\w+GenHyp)", source)[1], re.search(r"struct (\w+GenEll)", source)[1]
    source = "#include <pops/physics/composition/no_source.hpp>\n#include <pops/physics/composition/composite.hpp>\n" + source
    source += "\n#include <pops/numerics/fv/path_flux.hpp>\nint main(){\n"
    source += f"using M=pops::CompositeModel<pops_generated::{hyp},pops::NoSource,pops_generated::{ell}>;\n"
    source += f"static_assert(pops::coordinated_face_model<M> == {'true' if coordinated else 'false'});\n"
    policy = "CoordinatedFaceFlux" if coordinated else "PathRusanovFlux"
    source += f"static_assert(std::is_same_v<pops::ModelPathFlux<M>,pops::{policy}<M::n_vars>>);\n"
    source += "M m; M::State l{},r{}; auto p=pops::bind_flux_providers<M>(pops::FluxProviderValues<M>{});\n"
    source += ("l[0]=.9;l[2]=.1;r[0]=.8;r[2]=.2;\n" if coordinated else "l[0]=1;l[1]=2;r=l;\n")
    source += "auto v=pops::ModelPathFlux<M>{}.evaluate_path<0>(m,l,p,r,p);if(!v.succeeded())return 1;\n"
    if coordinated:
        source += "if(std::abs(v.conservative_flux.values[1]-v.left_ncp.values[1]-.405)>1e-15)return 2;\n"
        source += "if(std::abs(v.conservative_flux.values[1]+v.right_ncp.values[1]-.32)>1e-15)return 3;\n"
        source += "l[0]=-1;v=pops::ModelPathFlux<M>{}.evaluate_path<0>(m,l,p,r,p);if(v.status!=pops::PathStatus::DomainFailure)return 4;\n"
    else:
        source += "if(v.conservative_flux.values[0]!=.5 || v.left_ncp.values[0]!=0 || v.right_ncp.values[0]!=0)return 5;\n"
    source += "return 0;}\n"
    cpp, exe = tmp_path / "composite.cpp", tmp_path / "composite"
    cpp.write_text(source)
    command = [shutil.which("c++"), "-std=c++20", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-DPOPS_NATIVE_DIM="+str(dimension), "-I"+str(ROOT/"include"), "-I"+str(prefix/"include"),
        str(cpp), "-o", str(exe), "-L"+str(prefix/"lib"), "-lkokkoscore",
        "-Wl,-rpath,"+str(prefix/"lib")]
    if sys.platform == "darwin":
        command.append("-lomp")
    compiled = subprocess.run(command, text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(exe)], text=True, capture_output=True)
    assert executed.returncode == 0, executed.stderr
