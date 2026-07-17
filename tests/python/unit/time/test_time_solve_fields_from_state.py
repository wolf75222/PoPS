#!/usr/bin/env python3
"""Per-stage elliptic field solve in the final public runtime (ADC-409).

Each consumed callable ``FieldHandle(U_stage)`` now lowers to
``ctx.solve_fields_from_state(0, <U_stage>)``:
the elliptic fields are re-solved -- and the shared aux re-filled -- from THAT stage's state, not the
block's current state. So a field-COUPLED multi-stage scheme (Poisson feedback into the RHS) is exact:
stage k's RHS reads phi solved from stage k's own state. The compiled Program runs the stages
sequentially, so stage k's solve overwrites the shared aux before stage k's RHS reads it -- no distinct
per-stage FieldContext buffer is needed.

(A) Public IR/provenance: the detached compiled Program records two field solves with distinct
    state inputs.  The second solve consumes ``U1`` and the second RHS consumes both ``U1`` and the
    exact field context produced from ``U1``.  This is the public, typed proof that lowering cannot
    silently substitute the block's current state.

(B) Field-coupled parity (skips unless the full toolchain is present): a 2-stage Heun (RK2) scheme on a
    model whose RHS reads grad phi (a named ``electric`` source = -rho*grad phi, with
    ``-laplacian(phi) == rho - 1`` so phi depends on rho). The compiled program does:
        stage 1: solve phi(U0); R0 = rhs(U0);  U1 = U0 + dt*R0
        stage 2: solve phi(U1) [via solve_fields_from_state -- the new path]; R1 = rhs(U1)
        commit:  U_np1 = U0 + 0.5*dt*(R0 + R1)
    It is compared to a public Forward-Euler reference compiled through the same
    ``Case -> validate -> resolve -> compile -> bind -> run`` path.  Running that reference once
    from U0 and once from U1 obtains R0 and R1 while re-solving phi from each exact initial state.
    The reconstructed Heun update must match to ~1e-12.

Skips cleanly when numpy or a native compiler/Kokkos prerequisite is genuinely unavailable. A stale
installed extension or any compile/lowering/ABI failure remains a hard test failure.
"""
import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.fields import (CellCenteredSecondOrder, ConstantNullspace, FieldDiscretization,
                         FieldOutput, GradientOutput, MeanValueGauge)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model as BoardModel
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import Relative

from pops.numerics.terms import Flux as FinalFlux, SourceTerm
from pops.time import FailRun, FixedDt
import sys
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


INCLUDE = repo_include()
CXX = default_cxx()


def _skip(msg):
    require_native_or_skip("test_time_solve_fields_from_state: %s" % msg)


try:
    import numpy as np

    from pops.math import ddt, div, laplacian, sqrt
    from pops import time as adctime
except ImportError as exc:  # numpy or an installed PoPS module is genuinely absent
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0
N = 16
DT = 0.02


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _public_field_program_artifact(name="sffs_public", *, scheme="heun"):
    """Compile one field-coupled step through the complete final public lifecycle."""
    if scheme not in ("heun", "forward_euler"):
        raise ValueError("unknown public reference scheme %r" % scheme)
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = BoardModel(name, frame=frame)
    state = model.state("U", components=("rho", "mx", "my"))
    rho, mx, my = state
    u, v = mx / rho, my / rho
    pressure = 0.5 * rho
    sound_speed = sqrt(0.5 + 0.0 * rho)
    flux = model.flux(
        "transport", frame=frame, state=state,
        components={x_axis: (mx, mx * u + pressure, my * u),
                     y_axis: (my, mx * v, my * v + pressure)},
        waves={x_axis: (u - sound_speed, u, u + sound_speed),
               y_axis: (v - sound_speed, v, v + sound_speed)},
    )
    potential = model.field("potential")
    gx, gy = model.aux("grad_x"), model.aux("grad_y")
    field_operator = model.field_operator(
        "electrostatic", unknown=potential,
        equation=-laplacian(potential) == rho - 1.0,
        outputs=(FieldOutput("potential", potential), GradientOutput("grad", potential)),
    )
    electric = model.source("electric", on=state, value=(0.0 * rho, -rho * gx, -rho * gy))
    electric_operator = model.module.operator_handle("electric")
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + electric)
    case = pops.Case("%s-case" % name)
    block = case.block("plasma", model)
    field = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            # A strict solve makes the independently-bound FE stages a stable 1e-12 oracle even
            # though the Heun stage-2 solve may start from a different native MG iterate.
            solver=GeometricMG(tolerance=Relative(1e-12), max_cycles=100),
            nullspace=ConstantNullspace(), gauge=MeanValueGauge(0.0),
        ),
    )
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = adctime.Program(name)
    temporal = program.state(block[state])
    dt = program.dt
    fields0 = field(temporal.n, name="fields_0").consume(action=FailRun())
    r0 = program.rhs(
        name="R0",
        state=temporal.n,
        fields=fields0,
        terms=[FinalFlux(), SourceTerm(electric_operator)],
    )
    if scheme == "forward_euler":
        program.commit(temporal.next, program.value(
            "U_np1", temporal.n + dt * r0, at=temporal.next.point))
    else:
        stage = adctime.StagePoint(
            "heun_predictor", {"main": adctime.TimePoint(program.clock, 1)})
        stage_state = program.value("U1", temporal.n + dt * r0, at=stage)
        fields1 = field(stage_state, name="fields_1").consume(action=FailRun())
        r1 = program.rhs(
            name="R1",
            state=stage_state,
            fields=fields1,
            terms=[FinalFlux(), SourceTerm(electric_operator)],
        )
        program.commit(temporal.next, program.value(
            "U_np1", temporal.n + 0.5 * dt * r0 + 0.5 * dt * r1,
            at=temporal.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production(),
                            compile_options={"include": INCLUDE, "cxx": CXX})
    artifact = pops.compile(resolved)
    # Keep the public Program inspection value beside the artifact.  The aggregate compiled
    # artifact intentionally hides its internal executable component, while ``program_hash``
    # authenticates that this exact inspected Program is the one which was lowered.
    return artifact, program


# The public lifecycle compiles native code; skip only for an actual missing prerequisite.
missing_native = missing_native_compile_requirement(INCLUDE, CXX)
if missing_native:
    _skip(missing_native)


def _initial_state():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.4 * rho
    my = -0.2 * rho
    return np.ascontiguousarray(np.stack([rho, mx, my]))


def _run_public_step(artifact, initial_state):
    """Run one already-compiled public Program from one exact bind-owned state."""
    simulation = pops.bind(artifact, initial_state={"plasma": initial_state})
    report = pops.run(simulation, t_end=DT, max_steps=1)
    result = np.asarray(simulation.state_global("plasma"), dtype=np.float64)
    return report, result


print("== (A) public IR retains the exact per-stage field provenance ==")
compiled, heun_program = _public_field_program_artifact()
nodes = heun_program.ir_nodes()
chk(compiled.program_hash == heun_program.inspect().hash,
    "the compiled artifact authenticates the exact publicly-inspected Heun Program")
field_solves = [node for node in nodes if node["op"] == "solve_fields"]
fields0 = next((node for node in field_solves if node["name"] == "fields_0"), None)
fields1 = next((node for node in field_solves if node["name"] == "fields_1"), None)
rhs1 = next((node for node in nodes if node["op"] == "rhs" and node["name"] == "R1"), None)

chk(len(field_solves) == 2, "the compiled Heun IR contains exactly two field solves")
chk(fields0 is not None and fields1 is not None and fields0["inputs"] != fields1["inputs"],
    "the two field solves consume distinct stage states")
chk(fields1 is not None and fields1["inputs"] == ["U1"],
    "the second field solve consumes the explicit predictor state U1")

context0 = fields0["field_context"] if fields0 is not None else None
context1 = fields1["field_context"] if fields1 is not None else None
sources0 = context0.get("stage_sources", []) if context0 is not None else []
sources1 = context1.get("stage_sources", []) if context1 is not None else []
chk(len(sources0) == 1 and len(sources1) == 1 and sources0 != sources1,
    "field provenance retains two distinct owner-qualified stage-source ids")
chk(context1 is not None and {"grad_x", "grad_y"}.issubset(context1["outputs"]),
    "the second solve publishes the gradient outputs consumed by the electric source")
chk(rhs1 is not None and "U1" in rhs1["inputs"] and "fields_1" in rhs1["inputs"],
    "R1 consumes both U1 and the fields_1 outcome")
chk(rhs1 is not None and context1 is not None and rhs1["field_context"] == context1,
    "R1 carries the exact field provenance produced from U1")


print("== (B) field-coupled 2-stage parity through the public runtime ==")

U0 = _initial_state()
program_report, U_prog = _run_public_step(compiled, U0)
chk(program_report.accepted_steps == 1, "the public field-coupled Program accepted one step")

# Public replay: each Forward-Euler bind starts from the exact requested stage state and its Program
# solves the field before evaluating the same named flux + electric source.  Therefore
#   FE(U0) - U0 = dt*R0(U0, phi(U0))
#   FE(U1) - U1 = dt*R1(U1, phi(U1)).
forward_euler, forward_euler_program = _public_field_program_artifact(
    "sffs_forward_euler", scheme="forward_euler")
fe_nodes = forward_euler_program.ir_nodes()
chk(forward_euler.program_hash == forward_euler_program.inspect().hash,
    "the compiled artifact authenticates the exact publicly-inspected Forward-Euler Program")
fe_solves = [node for node in fe_nodes if node["op"] == "solve_fields"]
chk(len(fe_solves) == 1,
    "the public Forward-Euler oracle performs exactly one field solve per bound stage state")

fe0_report, U1 = _run_public_step(forward_euler, U0)
fe1_report, U2 = _run_public_step(forward_euler, U1)
chk(fe0_report.accepted_steps == 1 and fe1_report.accepted_steps == 1,
    "both public Forward-Euler reference stages accepted exactly one step")
R0 = (U1 - U0) / DT
R1 = (U2 - U1) / DT
U_ref_perstage = U0 + 0.5 * DT * R0 + 0.5 * DT * R1

e_perstage = float(np.abs(U_prog - U_ref_perstage).max())
print("  per-stage parity: max|d| = %.2e" % e_perstage)
chk(e_perstage < 1e-12,
    "compiled Heun == public per-stage Forward-Euler reconstruction (max|d| = %.2e)"
    % e_perstage)

# Sanity: the program ran, conserved mass (periodic; the electric source is momentum-only), and moved.
mass0, mass1 = float(U0[0].sum()), float(U_prog[0].sum())
chk(abs(mass1 - mass0) < 1e-9, "mass (sum rho) conserved over the step (|d| = %.2e)"
                               % abs(mass1 - mass0))
chk(float(np.abs(U_prog - U0).max()) > 1e-6, "the 2-stage step actually changed the state")

print("%s test_time_solve_fields_from_state" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
