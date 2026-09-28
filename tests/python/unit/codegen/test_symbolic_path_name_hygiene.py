"""The proposal-speed adapter delegates without rebinding scientific names."""
import shutil
import subprocess

import pytest
import pops
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume, SymbolicPath,
                           reconstruction, riemann, variables)
from pops.time import AdaptiveCFL


def emitted_path(names):
    frame = Rectangle("names", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("author_names", frame=frame)
    state = model.state("Q", components=names)
    flux = model.flux("transport", state=state, frame=frame,
                      components={axis: tuple(.5*q for q in state) for axis in frame.axes})
    matrix = tuple(tuple(1. if i == j else 0. for j in range(4)) for i in range(4))
    product = model.nonconservative_product("product", state=state,
                                            matrices={axis: matrix for axis in frame.axes})
    path = SymbolicPath(product, frame=frame, quadrature=((.5, 1.),),
                        speed=lambda left, right, axis: 1.5)
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    plan = DiscretizationPlan()
    plan.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("author_names")
    block = case.block("matter", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=.25, max_dt=1e-3))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(4,4), periodic=PeriodicAxes(frame.axes))))
    resolved_block = resolved.blocks[0]
    emitter, _ = lower_and_validate(resolved_block.model,
        state_space=resolved_block.state_spaces[0],
        resolved_operations=resolved_block.resolved_operations, numerics=resolved_block.numerics)
    assert tuple(emitter._m.cons_names) == names
    return _emit_bricks(emitter._m)[1]


@pytest.mark.parametrize("names", [("a", "b", "c", "d"), ("d", "c", "a", "b"),
                                    ("g", "result", "direction", "state"),
                                    ("params", "speed", "left", "right")])
def test_proposal_speed_never_declares_unused_author_locals(names, tmp_path):
    source = emitted_path(names)
    start = source.index("POPS_HD pops::Real max_wave_speed(const State& U,")
    end = source.index("\n  }", start)
    method = source[start:end]
    assert "const pops::Real " not in method, method
    assert "path_covector<Axis>(a)" in method
    assert "path_integral(U, U, g)" in method
    assert "result.speed_bound" in method
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the generated adapter syntax check")
    # Compile the actual adapter, with only its two delegated interfaces stubbed.
    # The complete installed four-state runtime test remains a separate proof.
    cpp = tmp_path / "path_adapter.cpp"
    cpp.write_text("#include <array>\n#include <limits>\n#define POPS_HD\n"
                   "namespace pops { using Real = double; }\n"
                   "struct Model { using State = std::array<double, 4>; static constexpr int dimension=2;\n"
                   "struct Result { double speed_bound; bool succeeded() const {return true;} };\n"
                   "template<int Axis> std::array<double,2> path_covector(const auto&) const {return {};}\n"
                   "Result path_integral(const State&,const State&,const std::array<double,2>&) const {return {};}\n"
                   "template<int Axis> " + method + "\n  }\n};\n"
                   "void instantiate() { Model{}.max_wave_speed<0>(Model::State{}, 0); }\n")
    compiled = subprocess.run([compiler, "-std=c++20", "-fsyntax-only", str(cpp)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
