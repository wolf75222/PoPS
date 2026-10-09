"""Source/host acceptance; no Native extension, bind, runtime or JIT."""
import json
import pathlib
import subprocess
import sys
import pytest
import pops
from pops.solvers import Newton
from pops.solvers.tolerances import Relative, AbsoluteFloor, Absolute
from pops.solvers.nonlinear.convergence import lower_tolerance, validate_convergence
from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case, finish

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_actual_source_identity():
    assert pathlib.Path(pops.__file__).resolve() == ROOT / "python/pops/__init__.py"
    assert "pops._pops" not in sys.modules


@pytest.mark.parametrize("tol", [Relative(1e-10), Relative(1e-10, floor=AbsoluteFloor(1e-11)), Absolute(1e-11)])
def test_complete_original_public_route(tol):
    args = mixed_case(solver=Newton(tolerance=tol))
    code = finish(*args)
    assert "pops::FieldNewtonConvergence{" in code
    assert "pops::collective_field_newton_stop_tolerance(" in code
    assert "original_field_residual_recheck_failed" in code
    assert args[2]._serialize()["version"] == 22


@pytest.mark.parametrize("mutation", ["missing", "kind", "contract", "raw", "bool", "nan", "negative", "noncanonical", "extra"])
def test_policy_corruption_refused(mutation):
    data = lower_tolerance(Relative(1e-10))
    if mutation == "missing": del data["relative"]
    elif mutation == "kind": data["kind"] = "legacy"
    elif mutation == "contract": data["contract"] += "foreign"
    elif mutation == "raw": data["relative"] = 1e-10
    elif mutation == "bool": data["relative"] = True
    elif mutation == "nan": data["relative"]["value"] = "nan"
    elif mutation == "negative": data["absolute"]["value"] = (-1.0).hex()
    elif mutation == "noncanonical": data["relative"]["value"] = "0.0000000001"
    elif mutation == "extra": data["scope"] = "reduced"
    with pytest.raises((ValueError, TypeError)):
        validate_convergence(data)


def test_installed_protocol_and_authentication():
    class Runtime:
        def set_field_newton_convergence_plan(self, *args): self.args = args
    prepared = Newton(tolerance=Relative(1e-10, floor=AbsoluteFloor(1e-11))).lower_field_nonlinear(target="system", layout=None)
    runtime = Runtime()
    prepared.install(runtime, "slot")
    assert runtime.args[-3:] == (1, 1e-10, 1e-11)
    with pytest.raises(TypeError): prepared.install(object(), "slot")
    prepared.convergence["relative"]["value"] = (2e-10).hex()
    with pytest.raises(ValueError, match="identity"):
        prepared.install(runtime, "slot")


def test_prepared_identity_detects_policy_mutation():
    from pops.time._program.spatial_solve import PreparedSpatialNewton
    from pops.time._graph.base import CanonicalData
    prepared = Newton(tolerance=Relative(1e-10)).prepare_program_solve()
    changed = lower_tolerance(Absolute(1e-11))
    with pytest.raises(ValueError):
        PreparedSpatialNewton(prepared.controls, prepared.identity, convergence=CanonicalData(changed))


def test_real_cpp_header_thresholds(tmp_path):
    source = tmp_path / "criteria.cpp"
    source.write_text(r'''#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <cassert>
#include <limits>
int main() {
  pops::FieldNewtonOptions o;
  o.tolerance=1e-10;
  for(double reference: {0., .25, 1., 4.})
    assert(pops::field_newton_stop_tolerance(o,reference)==1e-10*std::max(1.,reference));
  o.convergence={pops::FieldNewtonConvergenceKind::kRelative,1e-10,0.};
  assert(pops::field_newton_stop_tolerance(o,0.)==0.);
  assert(pops::field_newton_stop_tolerance(o,.25)==2.5e-11);
  assert(pops::field_newton_stop_tolerance(o,4.)==4e-10);
  o.convergence.absolute=1e-11;
  assert(pops::field_newton_stop_tolerance(o,0.)==1e-11);
  o.convergence={pops::FieldNewtonConvergenceKind::kAbsolute,0.,1e-11};
  assert(pops::field_newton_stop_tolerance(o,4.)==1e-11);
  for(double bad: {-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;try{(void)pops::field_newton_stop_tolerance(o,bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  }
  o.convergence.kind=static_cast<pops::FieldNewtonConvergenceKind>(99);
  bool rejected=false;try{pops::validate_field_newton_options(o);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
}
''')
    exe = tmp_path / "criteria"
    subprocess.run(["c++", "-std=c++20", "-I"+str(ROOT/"include"), "-I/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/include", str(source), "-o", str(exe)], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True, capture_output=True)


@pytest.mark.parametrize("tol", [Relative(1e-10), Relative(1e-10, floor=AbsoluteFloor(1e-11)), Absolute(1e-11)])
def test_complete_original_amr_route(tol):
    from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.params import RuntimeParam
    from pops.time import every
    args = mixed_case((2,0,1), width=3, solver=Newton(tolerance=tol))
    case, field, program, current, request, block, forcing, frame = args
    transfer = AMRTransfer(); transfer.state(block[forcing], StateTransfer())
    threshold = case.param(RuntimeParam("refinement threshold", default=.5))
    layout = AMR(grid=CartesianGrid(frame=frame,cells=(16,12),periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[forcing])["f0"] > case.value(threshold)),Buffer(cells=1)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000,clock=program.clock)),transfer=transfer,execution=AMRExecution.synchronous())
    code = finish(*args,layout=layout,target="amr_system")
    assert "pops::FieldNewtonConvergence{" in code
    assert "PreparedAmrFieldResidual" in code
    assert program._serialize()["version"] == 22


def test_legacy_numeric_bytes_pinned_to_frozen_base():
    import hashlib
    solver = Newton()
    assert set(solver.numerical_options()) == {"tolerance","max_iterations","linear_tolerance","linear_max_iterations","restart","armijo","minimum_step"}
    assert solver.prepare_program_solve().identity.token == "pops.prepared-spatial-newton.v1:sha256:089d8f10917911ecd58610933c2614cd65d4d5a379b4ab012c5c7781b43e7968"
    manifest = solver.lower_field_nonlinear(target="system",layout=None).to_data()
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest() == "6cbbc8dd3812a590df1165263f66662149278ed6ece91e360542733fd13bdf16"
    args = mixed_case(solver=solver)
    code = finish(*args)
    assert hashlib.sha256(code.encode()).hexdigest() == "290c85bafbee241c291cee5b40ec904b9614b363a038adef3c0b1893477c22e2"
    assert args[2]._serialize()["version"] != 22


@pytest.mark.parametrize("kind", ["relative", "absolute", "floor"])
def test_mutated_descriptor_bool_is_not_a_numeric_coefficient(kind):
    if kind == "relative":
        tolerance = Relative(1e-10); tolerance.rel = True
    elif kind == "absolute":
        tolerance = Absolute(1e-11); tolerance.abs_tol = True
    else:
        tolerance = Relative(1e-10,floor=AbsoluteFloor(1e-11)); tolerance.floor.abs_floor = True
    with pytest.raises((ValueError,TypeError)):
        Newton(tolerance=tolerance)
