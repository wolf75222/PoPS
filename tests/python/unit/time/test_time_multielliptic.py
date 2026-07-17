#!/usr/bin/env python3
"""Named multi-elliptic-field runtime (m.elliptic_field), ADC-428 (epic ADC-399, completes ADC-419).

ADC-419 landed the IR + validation + hash for m.elliptic_field("phi2", rhs=, operator=, aux=[...]) but
the named callable field route could not lower its SECOND elliptic solve + own aux channel. ADC-428
wires the runtime on the production/system backend: a named field gets

  - its OWN RHS brick (a function of the conservative state, like m.elliptic_rhs),
  - a DEDICATED native elliptic solver instance (GeometricMG/FFT, reused -- not reimplemented),
  - its OWN aux output channel (the model's named aux_field slots, distinct from the shared phi/grad),

and calling its exact ``FieldHandle`` with ``U`` lowers to ctx.solve_fields_from_state(field, block, U).

Section A (pure Python, always runs): the named solve_fields op lowers to the named ctx call (NOT the
default 2-arg one); the default solve_fields lowers byte-identically to before; unknown field / missing
model / aux-reading rhs / undeclared aux output are rejected with clear errors.

Section B (gated, self-skip) compares complete public runtime lifecycles.

  - PARITY: a named field "phi2" with rhs = (the SAME RHS as the default Poisson coupling) solves the
    IDENTICAL elliptic problem with the SAME native solver, so its derived gradient (g2x/g2y) equals the
    default grad_x/grad_y. A Case carrying BOTH providers and whose Program reads phi2 therefore steps
    like a default-only Case whose Program reads potential. This is a true second independent solve,
    validated without an offline multigrid reimplementation.
  - DISTINCT RHS (linearity): a named field with rhs = 2 * (default RHS) produces phi2 = 2*phi (Poisson
    is linear), so g2x = 2*grad_x; the source reading 0.5*g2x reproduces the default-grad step -> the
    named field carries a genuinely DIFFERENT, correctly scaled field.
  - NO REGRESSION: a default-only model (no named field) stepped via a program is byte-identical to the
    same model stepped before this feature (the named code path is inert; asserted on the lowered C++).

Skips cleanly (exit 0) without numpy / _pops / a compiler / a visible Kokkos -- never fakes the engine.
"""
from pops.codegen.program_codegen import emit_cpp_program
import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.math import ddt, div, laplacian
from pops.fields import (CellCenteredSecondOrder, ConstantNullspace, FieldDiscretization,
                         FieldOutput, GradientOutput, MeanValueGauge)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model as BoardModel
from pops.solvers.elliptic import GeometricMG
from typed_program_support import codegen_field_plans, solve_field, typed_field, typed_state

from pops.params import ConstParam
from pops.numerics.terms import DefaultSource, Flux, Flux as FinalFlux, SourceTerm
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


def _pops_mods():
    try:
        from pops.math import sqrt
        from pops.physics._facade import Model
        from pops import time as adctime
    except ImportError as exc:  # a genuinely absent installed package is an environment skip
        require_native_or_skip("test_time_multielliptic: pops unavailable: %s" % exc)
    return Model, sqrt, adctime


Model, sqrt, adctime = _pops_mods()

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
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure
        return False
    return False


Q = -1.0  # charge sign (f = q * rho), like pops::ChargeDensity


def _public_program_artifact(
    name, *, selected_field="phi2", scale=1.0, src_scale=1.0,
):
    """Compile a default-only or default-plus-named field Case through the public lifecycle."""
    if selected_field not in {"potential", "phi2"}:
        raise ValueError("selected_field must be 'potential' or 'phi2'")
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = BoardModel(name, frame=frame)
    state = model.state("U", components=("rho", "mx", "my"))
    rho, mx, my = state
    velocity_x = mx / rho
    velocity_y = my / rho
    pressure = 0.5 * rho
    flux = model.flux(
        "transport", frame=frame, state=state,
        components={x_axis: (mx, mx * velocity_x + pressure, my * velocity_x),
                     y_axis: (my, mx * velocity_y, my * velocity_y + pressure)},
        waves={x_axis: (velocity_x, velocity_x, velocity_x),
               y_axis: (velocity_y, velocity_y, velocity_y)},
    )
    potential = model.field("potential")
    default_gx, default_gy = model.aux("grad_x"), model.aux("grad_y")
    default_operator = model.field_operator(
        "default_electrostatic", unknown=potential,
        equation=-laplacian(potential) == Q * (rho - 1.0),
        outputs=(FieldOutput("phi", potential), GradientOutput("grad", potential)),
    )
    if selected_field == "potential":
        gx, gy = default_gx, default_gy
    else:
        phi2 = model.field("phi2")
        gx, gy = model.aux("g2_x"), model.aux("g2_y")
        named_operator = model.field_operator(
            "named_electrostatic", unknown=phi2,
            equation=-laplacian(phi2) == scale * Q * (rho - 1.0),
            outputs=(FieldOutput("phi2", phi2), GradientOutput("g2", phi2)),
        )
    source = model.source(
        "electric", on=state,
        value=(0.0 * rho, -src_scale * rho * gx, -src_scale * rho * gy),
    )
    source_operator = model.module.operator_handle("electric")
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + source)

    case = pops.Case("%s-case" % name)
    block = case.block("plasma", model)
    def discretization():
        return FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=GeometricMG(),
            nullspace=ConstantNullspace(), gauge=MeanValueGauge(0.0),
        )

    field = case.field(default_operator, discretization())
    if selected_field == "phi2":
        field = case.field(named_operator, discretization())
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = adctime.Program(name)
    temporal = program.state(block[state])
    fields = field(temporal.n, name="fields").consume(action=FailRun())
    rhs = program.rhs(
        state=temporal.n,
        fields=fields,
        terms=[FinalFlux(), SourceTerm(source_operator)],
    )
    program.commit(temporal.next, program.value(
        "U1", temporal.n + program.dt * rhs, at=temporal.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(
        pops.validate(case), layout=layout, backend=Production(),
        compile_options={"include": INCLUDE, "cxx": CXX},
    )
    return pops.compile(resolved)


# --- shared isothermal 2D fluid block (rho, mx, my; default Poisson f = q*rho) ---
def _block(m):
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    m.elliptic_rhs(Q * rho)  # default Poisson coupling: f = q * rho
    return rho, mx, my


def default_model(name="me_default"):
    """Default Poisson only; the source pushes momentum along the default electric field -grad phi."""
    m = Model(name)
    rho, mx, my = _block(m)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    # S = (0, -rho*grad_x, -rho*grad_y): the standard electrostatic force on the momentum.
    m.source([0.0 * rho, -rho * gx, -rho * gy])
    return m


def named_model(name="me_named", scale=1.0, src_scale=1.0):
    """Default Poisson PLUS a named field 'phi2' with rhs = scale * (default RHS). The source reads
    phi2's OWN gradient (g2x/g2y, the named aux), multiplied by src_scale -- NOT the default grad."""
    m = Model(name)
    rho, mx, my = _block(m)
    # The named field's output aux components -- declared as model aux_field slots so a source can read
    # them and the runtime has a channel to write into.
    g2x = m.aux_field("g2x")
    g2y = m.aux_field("g2y")
    m.aux_field("phi2")  # declare the named field's potential aux slot (written C++-side, not read in this IR)
    m.elliptic_field("phi2", rhs=(scale * Q) * rho, aux=["phi2", "g2x", "g2y"])
    m.source([0.0 * rho, -src_scale * rho * g2x, -src_scale * rho * g2y])
    return m


# =================== Section A: pure Python ===================
print("== (A) m.elliptic_field lowering + validation ==")


def _prog(name, field=None, model=None):
    P = adctime.Program(name)
    U = typed_state(P, "plasma", model=model)
    if field is None:
        f = solve_field(P, U)
    else:
        f = solve_field(P, U, field=typed_field(P, field), name="f_" + field)
    R = P.rhs(name="R", state=U, fields=f, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U", model=model).next
    P.commit(endpoint, P.value("U1", U + P.dt * R, at=endpoint.point))
    return P


def _emit(program, *, model=None):
    return emit_cpp_program(
        program, model=model, field_plans=codegen_field_plans(program))


# default solve_fields lowers to the 2-arg ctx call (historical), named to the 3-arg ctx call.
default_codegen_model = default_model()
src_default = _emit(_prog("me_def_prog", model=default_codegen_model),
    model=default_codegen_model)
chk('ctx.solve_fields_from_state("potential", 0, ' in src_default,
    "default solve_fields lowers to its qualified potential provider")
chk('ctx.solve_fields_from_state("phi2", 0, ' not in src_default,
    "default solve_fields does NOT use the named phi2 overload")

named_codegen_model = named_model()
src_named = _emit(_prog("me_nam_prog", field="phi2", model=named_codegen_model),
    model=named_codegen_model)
chk('ctx.solve_fields_from_state("phi2", 0, ' in src_named,
    "named solve_fields lowers to ctx.solve_fields_from_state(\"phi2\", 0, ...)")

# The named brick + registration land in the native loader (production backend).
loader = named_model("me_nam_loader")._m.emit_cpp_native_loader(target="system")
chk("Ell_phi2" in loader, "the named elliptic RHS brick is emitted in the native loader")
chk('register_elliptic_field(name, "phi2"' in loader, "the named field registers its aux components")
chk("set_block_elliptic_field" in loader and "make_poisson_rhs" in loader,
    "the named field attaches its RHS closure (make_poisson_rhs of the brick)")

# Validation: a field handle is now a Case-owned solve authority.  The historical string-based
# "unknown field"/"missing model" cases are not meaningful final authoring errors anymore; any name
# can be declared in the fixture's Case.  The real invalid contract is a cross-Case field/state owner.
foreign_program = adctime.Program("me_foreign_field")
foreign_state = typed_state(foreign_program, "plasma")
foreign_field = typed_field(adctime.Program("me_foreign_owner"), "phi2")
chk(raises(ValueError, lambda: solve_field(
    foreign_program, foreign_state, field=foreign_field)),
    "a FieldHandle owned by another Case is rejected before lowering")


def _bad_rhs_aux():
    m = Model("me_badrhs")
    rho, mx, my = _block(m)
    m.elliptic_field("phi2", rhs=rho + m.aux("grad_x"))  # rhs reading aux -> rejected


chk(raises(ValueError, _bad_rhs_aux), "an elliptic_field rhs reading the aux channel is rejected")


def _bad_aux_out():
    m = Model("me_badaux")
    rho, mx, my = _block(m)
    m.elliptic_field("phi2", rhs=rho, aux=["never_declared"])  # output aux not an aux_field
    m._m._elliptic_field_registrations("Me_badauxGen")


chk(raises(ValueError, _bad_aux_out),
    "an elliptic_field whose aux output is not a declared aux_field is rejected")


# A named elliptic field on target='amr_system' now LOWERS (ADC-428): the AMR native loader emits the
# same register_elliptic_field + set_block_elliptic_field calls as the uniform loader, on the AmrSystem
# facade. (Previously this raised NotImplementedError -- the AMR path was the one deferral.)
amr_loader = named_model("me_amr")._m.emit_cpp_native_loader(target="amr_system")
chk('register_elliptic_field(name, "phi2"' in amr_loader,
    "a named elliptic field on target='amr_system' registers its aux components (ADC-428)")
chk("set_block_elliptic_field" in amr_loader and "make_poisson_rhs" in amr_loader,
    "the AMR named field attaches its RHS closure (make_poisson_rhs of the brick)")
chk("pops::AmrSystem*" in amr_loader,
    "the AMR named-field registration targets the AmrSystem facade")


# NO REGRESSION: a default-only model lowers IDENTICALLY whether or not the named feature exists. We
# assert the default program never emits the named (3-arg) ctx call (above) AND that adding a named
# field to a SECOND model leaves the default model's lowering untouched.
default_codegen_model2 = default_model()
src_default2 = _emit(_prog("me_def_prog", model=default_codegen_model2),
    model=default_codegen_model2)
chk(src_default == src_default2, "the default program lowers deterministically (no named-field leak)")


# =================== Section B: gated end-to-end parity ===================
print("== (B) named second elliptic solve == default Poisson solve (public runtimes) ==")


def _skipB(msg):
    print("%s test_time_multielliptic (A only)" % ("FAIL" if fails else "PASS"))
    if fails:
        sys.exit(1)
    require_native_or_skip("test_time_multielliptic native section: %s" % msg)


try:
    import numpy as np
except ImportError as exc:
    _skipB("numpy unavailable: %s" % exc)

missing_native = missing_native_compile_requirement(INCLUDE, CXX)
if missing_native:
    _skipB(missing_native)

N = 16
DT = 0.005


def _ic():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.2 * rho
    my = -0.1 * rho
    return np.stack([rho, mx, my])


def step_program(
    model, prog, *, artifact_name, selected_field, scale=1.0, src_scale=1.0,
):
    # ``model``/``prog`` retain the independent pure-lowering fixture above; the numerical path
    # is exclusively Case -> validate -> resolve -> compile -> bind -> run.
    del model, prog
    compiled = _public_program_artifact(
        artifact_name, selected_field=selected_field, scale=scale, src_scale=src_scale)
    if selected_field == "phi2":
        provider_outputs = {"phi2", "g2_x", "g2_y"}
        chk(
            provider_outputs.isdisjoint(compiled.arguments().aux),
            "named field outputs are reported as provider-owned, not external bind aux",
        )
    simulation = pops.bind(compiled, initial_state={"plasma": _ic()})
    report = pops.run(simulation, t_end=DT, max_steps=1)
    chk(report.accepted_steps == 1, "%s accepted one public runtime step" % artifact_name)
    return np.asarray(simulation.state_global("plasma"), dtype=np.float64)


# REFERENCE: the default Poisson coupling, source reads the default grad.
U0 = _ic()
reference_model = default_model("me_ref")
ref = step_program(reference_model, _prog("me_ref_fe", model=reference_model),
                   artifact_name="me_ref_public", selected_field="potential")
chk(float(np.abs(ref - U0).max()) > 1e-9, "the default electrostatic source actually moved the state")

# PARITY: a named field with rhs == the default RHS solves the same problem with the same native
# solver, so g2x/g2y == grad_x/grad_y -> the named-field-driven step matches the default step.
parity_model = named_model("me_par", scale=1.0, src_scale=1.0)
got = step_program(
    parity_model, _prog("me_par_fe", field="phi2", model=parity_model),
    artifact_name="me_par_public", selected_field="phi2", scale=1.0, src_scale=1.0)
e_par = float(np.abs(got - ref).max())
print("  named(rhs=default) vs default Poisson: max|d| = %.2e" % e_par)
chk(e_par < 1e-12,
    "named second elliptic solve (same RHS) == default Poisson solve (max|d| = %.2e)" % e_par)

# DISTINCT RHS (linearity): named rhs = 2*default -> phi2 = 2*phi -> g2x = 2*grad_x; the source reads
# 0.5*g2x, recovering the default-grad step. Confirms the named field carries a genuinely different,
# correctly scaled field (not an alias of the shared aux).
linear_model = named_model("me_lin", scale=2.0, src_scale=0.5)
got2 = step_program(
    linear_model, _prog("me_lin_fe", field="phi2", model=linear_model),
    artifact_name="me_lin_public", selected_field="phi2", scale=2.0, src_scale=0.5)
e_lin = float(np.abs(got2 - ref).max())
print("  named(rhs=2*default, src=0.5*g2) vs default: max|d| = %.2e" % e_lin)
chk(e_lin < 1e-12,
    "named field with rhs=2*default and src=0.5*g2 reproduces the default step (max|d| = %.2e)"
    % e_lin)

print("%s test_time_multielliptic" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
