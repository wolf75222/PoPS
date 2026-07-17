#!/usr/bin/env python3
"""Named-source predictor-corrector through the final public runtime lifecycle.

The only deliberately low-level seam in this test is ``emit_cpp_program`` in section (A): no
public pure-Python API exposes generated C++ text, and these assertions pin the named-source
lowering itself.  Even that seam consumes the Program and ``field_plans`` produced by the public
``Case -> validate -> resolve`` lifecycle and a model graph derived from that resolved plan.

Sections (B)/(C) contain only ``pops.compile -> artifact.verify -> pops.bind -> pops.run``.  The
predictor-corrector oracle replays its stages with a separately compiled public Forward-Euler rate
Program.  Toolchain absence is checked explicitly before those sections; compilation, lowering and
ABI errors are never caught or converted into a skip.
"""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys

from pops.codegen import Production
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler, PredictorCorrector
from pops.math import ddt, div, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import Flux
from pops.params import ConstParam
from pops.solvers.elliptic import GeometricMG
from pops.time import FailRun, FixedDt, Program
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    require_native_or_skip,
)


def _skip(msg):
    require_native_or_skip("test_predictor_corrector: %s" % msg)


try:
    import numpy as np
    import pops
except ImportError as exc:
    _skip("pops/numpy unavailable: %s" % exc)


ROOT = Path(__file__).resolve().parents[4]
N = 16
BZ = 3.0
DT = 0.02
CS2 = 0.5

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001 -- a wrong exception type is a failed contract assertion
        return False
    return False


def _physics(name):
    """Author the isothermal Poisson/Lorentz model through the public Model board."""
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model(name, frame=frame)
    state = model.state("U", components=("rho", "mx", "my"))
    rho, mx, my = state
    u, v = mx / rho, my / rho
    pressure = CS2 * rho
    sound_speed = CS2**0.5
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (mx, mx * u + pressure, my * u),
            y_axis: (my, mx * v, my * v + pressure),
        },
        waves={
            x_axis: (u - sound_speed, u, u + sound_speed),
            y_axis: (v - sound_speed, v, v + sound_speed),
        },
    )

    potential = model.field("potential")
    field_operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=-laplacian(potential) == rho - 1.0,
        outputs=(
            FieldOutput("phi", potential),
            GradientOutput("grad", potential, sign=1),
        ),
    )
    gx, gy = model.aux("grad_x"), model.aux("grad_y")
    # The public PredictorCorrector factory takes one exact field value.  An imposed B_z aux and
    # the electrostatic field have distinct providers and cannot honestly be collapsed into that
    # value.  This test's physical oracle is the spatially constant B_z case, so express that exact
    # coefficient as a typed constant parameter; spatially varying imposed-B_z coverage belongs to
    # a separate multi-provider Program once that public composition contract exists.
    bz = model.value(model.param(ConstParam("B_z", BZ)))
    electric_source = model.source(
        "electric", on=state, value=(0.0 * rho, -rho * gx, -rho * gy)
    )
    explicit_rate = model.rate(
        "explicit_rhs", equation=ddt(state) == -div(flux) + electric_source
    )
    lorentz_math = model.local_linear_operator(
        "lorentz",
        on=state,
        matrix=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, bz),
            (0.0, -bz, 0.0),
        ),
    )
    lorentz = model.operator("lorentz", returns=lorentz_math)
    electric = model.module.operator_handle("electric")
    return frame, model, state, flux, field_operator, electric, explicit_rate, lorentz


def _field_discretization():
    return FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=GeometricMG(),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0.0),
    )


def _manual_fe(block_state, field, electric):
    program = Program("electric_fe")
    temporal = program.state(block_state)
    fields = field(temporal.n, name="fields_n").consume(action=FailRun())
    rate = program.rhs(
        name="R", state=temporal.n, fields=fields, terms=[Flux(), electric]
    )
    endpoint = program.value(
        "U1", temporal.n + program.dt * rate, at=temporal.next.point
    )
    program.commit(temporal.next, endpoint)
    program.step_strategy(FixedDt(DT))
    return program


def _resolved_case(name, program_kind):
    frame, model, state, flux, field_operator, electric, rate, lorentz = _physics(name)
    case = pops.Case("%s-case" % name)
    block = case.block("plasma", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    field = case.field(field_operator, _field_discretization())
    block_state = block[state]
    if program_kind == "manual_fe":
        program = _manual_fe(block_state, field, electric)
    elif program_kind == "rate_fe":
        program = ForwardEuler(
            block_state, rate=rate, fields=field, solve_action=FailRun()
        )
        program.step_strategy(FixedDt(DT))
    elif program_kind == "predictor_corrector":
        program = PredictorCorrector(
            block_state,
            fields=field,
            explicit=rate,
            implicit=lorentz,
            solve_action=FailRun(),
        )
        program.step_strategy(FixedDt(DT))
    else:  # pragma: no cover -- test-local closed factory
        raise ValueError("unknown program kind %r" % program_kind)
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )


def _emit_resolved(resolved):
    """Explicit pure-codegen seam; all authorities come from the public resolved plan."""
    return emit_cpp_program(
        resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        field_plans=resolved.field_plans,
    )


def _initial_state():
    coordinates = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return np.ascontiguousarray(np.stack((rho, 0.4 * rho, -0.2 * rho)))


def _compile(resolved):
    artifact = pops.compile(resolved)
    artifact.verify()
    return artifact


def _run_one_step(artifact, state):
    runtime = pops.bind(
        artifact,
        initial_state={"plasma": np.ascontiguousarray(state)},
    )
    report = pops.run(runtime, t_end=DT, max_steps=1)
    actual = np.asarray(runtime.state_global("plasma"), dtype=np.float64).reshape(
        state.shape
    )
    return actual, report


def _reference_rhs(reference_artifact, state):
    advanced, report = _run_one_step(reference_artifact, state)
    chk(report.accepted_steps == 1, "public Forward-Euler reference accepted exactly one step")
    return (advanced - state) / DT


def _analytic_lorentz_solve(state, scale):
    k = scale * BZ
    denominator = 1.0 + k * k
    rho, mx, my = state
    return np.stack(
        (rho, (mx + k * my) / denominator, (-k * mx + my) / denominator)
    )


def _analytic_lorentz_apply(state):
    rho, mx, my = state
    return np.stack((np.zeros_like(rho), BZ * my, -BZ * mx))


# ---- (A) the only low-level section: pure generated-C++ assertions ----------------------------
print("== (A) named-source rhs codegen from a public resolved Case ==")
manual_fe_plan = _resolved_case("pc_manual_fe", "manual_fe")
src = _emit_resolved(manual_fe_plan)
chk(
    "ctx.neg_div_flux_default_into(0, " in src and "ctx.rhs_into(" not in src,
    "named electric source uses the flux-only base (no implicit default source)",
)
chk("pops::for_each_cell(" in src, "the named electric source is a per-cell kernel")
chk(
    "((-rho) * grad_x)" in src and "((-rho) * grad_y)" in src,
    "the electric kernel reads -rho*grad_x / -rho*grad_y",
)
chk(
    "auxA(i, j, 1)" in src and "auxA(i, j, 2)" in src,
    "grad_x / grad_y use canonical aux components 1 / 2",
)
chk("ctx.axpy(" in src, "the named source is accumulated onto the residual")

# The final API cannot manufacture an unknown/free source selector: rejection happens at Program
# authoring, before lowering.  This is stronger than the former late unknown-name emitter failure.
def _free_source_program():
    _, model, state, _, _, _, _, _ = _physics("pc_unknown_source")
    case = pops.Case("pc-unknown-source-case")
    block = case.block("plasma", model)
    program = Program("free_source_is_invalid")
    temporal = program.state(block[state])
    return program.rhs(state=temporal.n, terms=[Flux(), "does_not_exist"])


chk(
    raises((TypeError, ValueError), _free_source_program),
    "a free/unknown source name is rejected by the typed public Program contract",
)

# The historical test also manufactured a model carrying both a hidden non-empty "default" source
# and a named source.  The final public Model board intentionally has no such hidden-source authoring
# route: explicit sources are typed handles and rate equations compose them.  That legacy emitter
# matrix therefore stays outside this final public-runtime test rather than being recreated through
# ``pops.physics._facade``.

# Resolve the library factory through the same strict Case snapshot/authentication path as the
# manual Program above.  This also guards the generic ownership protocol used when a public model
# replaces an earlier live Module view before validation.
pc_plan = _resolved_case("pc_public", "predictor_corrector")
pc_src = _emit_resolved(pc_plan)
chk(
    pc_src.count("ctx.solve_fields_from_state(") == 2
    and ", 0, u0);" in pc_src
    and ", 0, u7);" in pc_src,
    "predictor and corrector re-solve fields from their own stage states",
)
chk(
    bool(pc_src),
    "the resolved predictor-corrector emits C++ with named sources and local Lorentz solves",
)


# ---- (B)/(C) public native lifecycle -----------------------------------------------------------
print("== compile public artifacts (Case -> validate -> resolve -> compile -> verify) ==")
missing = missing_native_compile_requirement(ROOT / "include", default_cxx())
if missing is not None:
    if fails:
        print("FAIL test_predictor_corrector: %d failure(s)" % fails)
        sys.exit(1)
    _skip("public native lifecycle unavailable: %s" % missing)
if find_spec("pops._pops") is None:
    if fails:
        print("FAIL test_predictor_corrector: %d failure(s)" % fails)
        sys.exit(1)
    _skip("public native lifecycle unavailable: pops._pops is not installed")

manual_fe_artifact = _compile(manual_fe_plan)
reference_fe_artifact = _compile(_resolved_case("pc_rate_fe", "rate_fe"))
pc_artifact = _compile(pc_plan)

initial = _initial_state()

print("== (B) named-source FE Program == public rate-operator FE Program ==")
actual_fe, fe_report = _run_one_step(manual_fe_artifact, initial)
reference_rate = _reference_rhs(reference_fe_artifact, initial)
expected_fe = initial + DT * reference_rate
fe_error = float(np.max(np.abs(actual_fe - expected_fe)))
print("  focused FE parity: max|d| = %.2e" % fe_error)
chk(fe_report.accepted_steps == 1, "public named-source FE run accepted exactly one step")
chk(
    fe_error < 1.0e-10,
    "explicit named-source rhs equals the public rate-operator path (max|d| = %.2e)"
    % fe_error,
)
chk(
    float(np.max(np.abs(actual_fe - initial))) > 1.0e-6,
    "the field-coupled electric/transport step changed the state",
)

print("== (C) full public predictor-corrector parity ==")
chk(
    pc_artifact.program_name == "PredictorCorrector",
    "artifact carries the canonical public PredictorCorrector Program name",
)
actual_pc, pc_report = _run_one_step(pc_artifact, initial)

# Replay the exact stages.  Each reference RHS is a fresh public bind/run of the independently
# compiled Forward-Euler rate Program, hence re-solves its field from the supplied stage state.
rate_n = _reference_rhs(reference_fe_artifact, initial)
predictor_rhs = initial + DT * rate_n
predicted = _analytic_lorentz_solve(predictor_rhs, DT)
rate_star = _reference_rhs(reference_fe_artifact, predicted)
lorentz_star = _analytic_lorentz_apply(predicted)
corrector_rhs = (
    initial
    + 0.5 * DT * rate_n
    + 0.5 * DT * rate_star
    + 0.5 * DT * lorentz_star
)
expected_pc = _analytic_lorentz_solve(corrector_rhs, 0.5 * DT)

pc_error = float(np.max(np.abs(actual_pc - expected_pc)))
print("  predictor-corrector parity: max|d| = %.2e" % pc_error)
chk(pc_report.accepted_steps == 1, "public predictor-corrector run accepted exactly one step")
chk(
    pc_error < 1.0e-10,
    "public predictor-corrector equals the independent staged replay (max|d| = %.2e)"
    % pc_error,
)
mass_before = float(initial[0].sum())
mass_after = float(actual_pc[0].sum())
chk(
    abs(mass_after - mass_before) < 1.0e-9,
    "mass is conserved over the periodic step (|d| = %.2e)"
    % abs(mass_after - mass_before),
)
chk(
    float(np.max(np.abs(actual_pc - initial))) > 1.0e-6,
    "the predictor-corrector step changed the state",
)

print("%s test_predictor_corrector" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
