"""N8 « dt= as a function of cell » (ROM-89, design note ``n8/cell_step_rule``, D1, D2, C2, P3):
the step bound of a cells rule in its compiled control (schema 6), its private carrier, the
Courant budget a rule declares, and the runtime parameters that its generated kernel reads,
without a native run."""
import math
import types

import pytest

import pops
import pops.surface as surface
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_emit_params import program_param_entries
from pops.lib.numerics import two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.surface.math import ddt, div, grad
from pops.time._program.control import (
    ADVANCE_CONTROL_CELLS_SCHEMA, ADVANCE_CONTROL_STEP_RULE_SCHEMA, AdvanceControl,
)
from pops.time._program.traced_method import NotYetElaborated
from pops._ir.expr import Const, Div, Mul
from pops.mesh._amr.refinement_criterion import CellWidth
from pops.time._step.bound import StepBound, step_fraction
from pops.time._step.strategy import CellStepBound

DEFAULT = (("finite", "default"),)
IDENTITY = "pops.cells-step-bound.v1::" + "0" * 64


def _control(**options):
    values = dict(assessments=DEFAULT, max_attempts=2, max_transforms=0, first_dt=None,
                  step_bound=(0.45, IDENTITY))
    values.update(options)
    return AdvanceControl(**values)


# --- the compiled control ------------------------------------------------------------------------

def test_a_cells_rule_records_schema_six_and_round_trips():
    control = _control()
    data = control.to_data()
    assert data["schema"] == ADVANCE_CONTROL_CELLS_SCHEMA == "pops.advance-control@6"
    assert data["proposal"] == {"rule": "cells", "fraction": (0.45).hex(), "bound": IDENTITY}
    assert set(data) == {"schema", "accept", "retry", "proposal", "outputs", "memory",
                         "trial_failures"}
    assert AdvanceControl.from_data(data) == control


@pytest.mark.parametrize("change", ["schema 4", "a stability rule", "a stray key", "dt",
                                    "no bound"])
def test_a_record_that_mixes_the_rules_is_refused(change):
    data = _control().to_data()
    if change == "schema 4":
        data["schema"] = ADVANCE_CONTROL_STEP_RULE_SCHEMA
    elif change == "a stability rule":
        data["proposal"]["rule"] = "stability"
    elif change == "a stray key":
        data["proposal"]["max_dt"] = (0.01).hex()
    elif change == "dt":
        data["proposal"]["dt"] = (0.001).hex()
    else:
        del data["proposal"]["bound"]
    with pytest.raises(ValueError, match="another schema|malformed proposal"):
        AdvanceControl.from_data(data)


@pytest.mark.parametrize("options, message", [
    (dict(step_bound=(0.0, IDENTITY)), "fraction"),
    (dict(step_bound=(1.5, IDENTITY)), "fraction"),
    (dict(step_bound=(float("nan"), IDENTITY)), "fraction"),
    (dict(step_bound=(1, IDENTITY)), "fraction"),
    (dict(step_bound=(0.45, "sha")), "identity"),
    (dict(step_bound=[0.45, IDENTITY]), "step_bound"),
    (dict(step_bound=(0.45, IDENTITY), step_rule=(0.45, None)), "step"),
    (dict(step_bound=(0.45, IDENTITY), first_dt=0.001), "first_dt"),
])
def test_a_malformed_bound_is_refused(options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _control(**options)


# --- the carrier and the budget ------------------------------------------------------------------

def test_a_function_without_a_budget_takes_that_of_a_fixed_step():
    def limit(cell):
        return cell

    assert step_fraction(limit) == (1.0, "default")
    limit.step_fraction = 0.5
    assert step_fraction(limit) == (0.5, "stated")
    limit.step_fraction = 1
    assert step_fraction(limit) == (1.0, "stated")


@pytest.mark.parametrize("value, error", [(0.0, ValueError), (-0.5, ValueError),
                                          (1.0000001, ValueError), (math.inf, ValueError),
                                          (math.nan, ValueError), (True, TypeError),
                                          ("0.5", TypeError)])
def test_a_declared_budget_is_in_the_unit_interval(value, error):
    def limit(cell):
        return cell

    limit.step_fraction = value
    with pytest.raises(error, match="step_fraction of a step rule"):
        step_fraction(limit)


def test_the_carrier_reads_the_cell_widths_values_and_constants():
    bound = StepBound(Div(Mul(Const(0.45), CellWidth(0)), Const(2.0)), 2, 0.45, "stated")
    assert (bound.fraction, bound.origin, bound.dimension) == (0.45, "stated", 2)
    with pytest.raises(TypeError, match="reads no cell"):
        StepBound(Const(1e-3), 2)
    with pytest.raises(ValueError, match="width of axis 2 of a layout of dimension 2"):
        StepBound(CellWidth(2), 2)
    with pytest.raises(ValueError, match="Courant budget in"):
        StepBound(CellWidth(0), 2, fraction=1.5)
    with pytest.raises(ValueError, match="stated or default"):
        StepBound(CellWidth(0), 2, origin="library")
    with pytest.raises(TypeError, match="an expression over the cell"):
        StepBound(1e-3, 2)
    with pytest.raises(TypeError, match="number of axes"):
        StepBound(CellWidth(0), True)


def test_the_strategy_of_a_cells_rule_names_its_budget_and_bound():
    strategy = CellStepBound(0.45, IDENTITY)
    assert strategy.kind == "cell_step_bound"
    assert (strategy.fraction, strategy.bound) == (0.45, IDENTITY)


# --- the runtime parameters of the kernel (driver ruling 12/11 22:43, conditions 1 and 2) --------

def _parameter_case(read_by_kernel):
    """L1's program A (advection-diffusion, upwind() and two_point(), dt=combined_limit): nu is read
    by the two_point() stencil, whose kernel routes it; scale is read by the bound alone."""
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2)
    scale = model.parameter("scale", default=1.0)
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))

    def combined_limit(cell):
        h = cell.width(domain.x)
        limit = 1 / (abs(a) / h + 2 * nu / h**2)
        return limit if read_by_kernel else scale * limit

    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=combined_limit),
                              initial={u: 0.5})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=256)})
    return pops.resolve(pops.validate(case.elaborated.case), layout=case.elaborated.layout)


def _emit(resolved):
    return emit_cpp_program(resolved.time,
                            model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_a_parameter_that_a_kernel_reads_is_routed_to_the_bound():
    """Condition 1: the bound reads the parameter through the table the kernels route
    (program_param_entries, ctx.program_params), not a copy of it."""
    resolved = _parameter_case(read_by_kernel=True)
    entries = program_param_entries(
        resolved.time, ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert entries == [(0, "nu", 0, 1e-2)]
    cpp = _emit(resolved)
    kernel = cpp.split("struct pops_program_advance_step_bound_kernel", 1)[1].split(
        "POPS_HD pops::Real operator()", 1)[1].split("\n  }\n", 1)[0]
    assert "params.get(0)" in kernel
    assert "  kernel.params = ctx.program_params(0);" in cpp


def test_a_parameter_that_no_kernel_reads_is_refused_with_its_lot():
    """Condition 2: a parameter that only the bound reads has no installed value."""
    with pytest.raises(NotYetElaborated, match="dt= as a function of cell reads the runtime "
                                               "parameter scale, which no kernel of the method "
                                               "reads") as caught:
        _emit(_parameter_case(read_by_kernel=False))
    assert "a later lot of N8 « dt= as a function of cell »" in str(caught.value)


def test_the_runtime_knows_the_zero_elliptic_brick_by_its_codegen_identity():
    """The plan guard of a cells rule takes a generated block whose elliptic brick is the zero
    right-hand side for one without a Poisson source: the identity it compares is the one that
    codegen gives that brick (``module_codegen._emit_bricks``, ``_exact_brick_contract``)."""
    from pathlib import Path

    import pops.codegen.module_codegen as module_codegen

    root = Path(__file__).resolve().parents[4]
    header = (root / "include/pops/runtime/builders/compiled/generated_system_block.hpp").read_text()
    source = Path(module_codegen.__file__).read_text()
    assert '"zero-elliptic-rhs",' in source
    assert '"pops.codegen.zero-elliptic-rhs-brick"' in header


def test_a_cell_width_is_positive_by_the_geometry():
    """Ruling 13/11 (a): cfl_limit's (cfl * w) / s at a cell at rest is w / +0 = +infinity only
    because w > 0, which the geometry guarantees: the surface and the b6 layouts refuse an empty
    extent or no cell, and the native Geometry, which the kernel's ``spacing`` reads, refuses a
    spacing that is not finite and positive."""
    from pathlib import Path

    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid

    with pytest.raises(ValueError, match="lower < upper"):
        surface.Box(x=(1.0, 1.0))
    with pytest.raises(TypeError, match="takes a number of cells"):
        surface.Uniform(cells=(4, 0))
    with pytest.raises(ValueError, match="upper must exceed lower"):
        CartesianDomain("box", lower=(0.0, 0.0), upper=(0.0, 1.0))
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    with pytest.raises(ValueError, match="must be >= 1"):
        CartesianGrid(frame=frame, cells=(0, 4))
    root = Path(__file__).resolve().parents[4]
    geometry = (root / "include/pops/mesh/geometry/geometry.hpp").read_text()
    assert "!(cell_spacing > Real(0)))" in geometry
    assert "Geometry physical extents must produce finite positive cell spacing" in geometry
