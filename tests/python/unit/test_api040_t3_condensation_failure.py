"""A singular elimination block must not become a finite condensed coefficient."""
from pathlib import Path
import re
import shutil
import subprocess

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt


def _source(diagonal=1, *, operation="coeff", target="system"):
    frame = Rectangle("box", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("block_response", frame=frame)
    state = model.state("U", components=("density", "first", "second"))
    operator = model.operator("response", returns=model.local_linear_operator(
        "response", on=state, matrix=((0, 0, 0), (0, diagonal, 0), (0, 0, diagonal))))
    case = pops.Case("elimination_domain")
    block = case.block("medium", model)
    program = pops.Program("condensation")
    q = program.state(block[state])
    common = dict(state=q.n, linear_operator=operator, subset=(1, 2), th_dt=1)
    result = q.n
    if operation == "coeff":
        program.condensed_coeffs(**common, c=1)
    elif operation == "flux":
        program.condensed_rhs(program.scalar_field("rhs"), **common, g=1, charge_component=0)
    else:
        result = program.condensed_reconstruct(**common, phi=program.scalar_field("potential"))
    program.commit(q.next, program.value("endpoint", 1*result, at=q.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                   periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    selected = resolved.blocks[0]
    emitted, _ = lower_and_validate(selected.model, state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    return emit_cpp_program(resolved.time, model=emitted, target=target)


@pytest.mark.parametrize("diagonal,expected_status", [(1, 1), (0, 0), (-1, 0)])
def test_public_condensation_checks_inverse_before_using_output(tmp_path, diagonal, expected_status):
    source = _source(diagonal)
    start = source.index("const pops::Real th_dt_")
    end = re.search(r"^\s*}\)\)?;", source[start:], re.MULTILINE)
    assert end is not None
    cell = source[start:start+end.start()]
    assert "block_inverse<2>" in cell
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang is needed to pin automatic-storage initialization in this witness")
    include = Path(__file__).resolve().parents[3] / "include"
    cpp = tmp_path / "singular.cpp"
    cpp.write_text("""#include <pops/numerics/linalg/block_inverse.hpp>
#include <cmath>
#include <limits>
struct Field {
  double values[4];
  double& operator()(int, int component) { return values[component]; }
};
int main() {
  const int index=0;
  const double rho=1.;
  Field tensorA{{NAN,NAN,NAN,NAN}};
  const double status = [&]() {
""" + cell + """
    return pops::Real(0);
  }();
  // An invalid inverse must be reported before producing a usable tensor.
  // Zero initialization is a compiler-supported choice of the old unwritten
  // automatic storage; a correct implementation must not depend on its bytes.
""" + ("""
  if (status != """ + str(expected_status) + """) return 2;
  for (double value : tensorA.values) if (std::isfinite(value)) return 1;
""" if expected_status else """
  if (status != 0) return 3;
  const double expected = 1 + 1/(1 - """ + str(float(diagonal)) + """);
  if (tensorA.values[0] != expected || tensorA.values[3] != expected ||
      tensorA.values[1] != 0 || tensorA.values[2] != 0) return 4;
""") + """
}
""")
    binary = tmp_path / "singular"
    subprocess.run([compiler, "-std=c++20", "-O2", "-ftrivial-auto-var-init=zero",
                    "-I" + str(include), str(cpp), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=False)
    assert result.returncode == 0, "invalid inverse was consumed or a valid inverse changed"


@pytest.mark.parametrize("target", ["system", "amr_system"])
@pytest.mark.parametrize("operation,label", [
    ("coeff", "condensed_coefficient_inverse"),
    ("flux", "condensed_flux_inverse"),
    ("reconstruct", "condensed_reconstruction_inverse"),
])
def test_each_public_elimination_route_converges_status_before_its_next_use(target, operation, label):
    source = _source(operation=operation, target=target)
    reduction = source.index("pops::for_each_cell_reduce_max(")
    consumption = source.index('"' + label + '"')
    assert reduction < consumption
    assert source.count('"' + label + '"') == 1
    assert re.search(r"if \(!pops::detail::block_(?:apply_)?inverse<2>", source)
    prefix = source[:consumption]
    if target == "amr_system" and operation != "reconstruct":
        assert ".grown_box()" in prefix[reduction:reduction+250]
    if operation in {"coeff", "flux"}:
        assert source.index("ctx.fill_boundary(", reduction) > consumption
    else:
        declaration = re.search(r"pops::Real (cond\d+_reconstruct_status) = 0;", source)
        assert declaration is not None
        assert declaration.start() < reduction


@pytest.mark.parametrize("diagonal", [1., 0., -1.])
def test_native_inverse_apply_failure_does_not_read_unwritten_destination(tmp_path, diagonal):
    # This is the common production fragment called by both flux/reconstruct
    # routes verified above, with the actual native inverse implementation.
    from pops.codegen.program_emit_condensed import _emit_apply_minv
    body = []
    _emit_apply_minv(body, ["3", "5"], ["first", "second"], "")
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang unavailable")
    cpp = tmp_path / "apply.cpp"
    cpp.write_text("""#include <pops/numerics/linalg/block_inverse.hpp>
#include <cmath>
int main() {
  const pops::Real M_[2][2] = {{1-DIAGONAL,0},{0,1-DIAGONAL}};
  pops::Real output[2] = {NAN,NAN};
  const auto status = [&]() {
""".replace("DIAGONAL", "(" + str(diagonal) + ")") + "\n".join(body) + """
    output[0]=first; output[1]=second;
    return pops::Real(0);
  }();
""" + ("if (status != 1 || std::isfinite(output[0]) || std::isfinite(output[1])) return 1;"
         if diagonal == 1 else
         "if (status != 0 || output[0] != 3/(1-DIAGONAL) || output[1] != 5/(1-DIAGONAL)) return 2;"
         .replace("DIAGONAL", "(" + str(diagonal) + ")")) + "\n}\n")
    binary = tmp_path / "apply"
    include = Path(__file__).resolve().parents[3] / "include"
    subprocess.run([compiler, "-std=c++20", "-O2", "-ftrivial-auto-var-init=zero",
                    "-I" + str(include), str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
