"""M7.1 synchronized composite implicit diffusion against a conservative fine-grid reference.

Native qualification retains all N=16,32,64 pairs and extends IMEX to N128, with four full
accepted steps, partial refinement, physical exchange accounting, and rejected-attempt rollback.
Source collection alone is not evidence.
"""

from __future__ import annotations
import numpy as np
import pops
import pytest
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic, Gaussian
from pops.math import ValueExpr, CoeffGradient, ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.params import RuntimeParam
from pops.solvers import Newton
from pops.time import (
    DerivativeStrategy,
    FailRun,
    FixedDt,
    ImplicitDiffusionStage,
    SolveUnknown,
    every,
)
from tests.python.support.amr_snapshots import composite_active_mask, composite_active_block_state
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
DT = 0.00025


def build(
    n,
    *,
    kind="constant",
    refined=True,
    boundary=False,
    invalid=False,
    subcycled=False,
    imex=False,
    failure_action=None,
    newton_iterations=20,
    step_dt=DT,
    periodic_witness=False,
):
    from pops.physics.diffusion import DiffusiveBoundary

    frame = Rectangle("composite-implicit-square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("composite-implicit-heat", frame=frame)
    state = model.state("U", components=("energy",))
    u = state[0]
    variable = (sqrt(1 + 4 * u) - 1) / 2 if kind == "nonlinear_accumulation" else u
    coefficient = 0.1 * (1 + 0.2 * u) if kind == "variable" else 0.1
    if kind == "diagonal":
        coefficient = ((0.1 * (1 + 0.2 * u), 0), (0, 0.07 * (1 + 0.1 * u)))
    if invalid:
        coefficient = 0.1 * (1.1 - u)
    physical = (
        tuple(
            DiffusiveBoundary(axis, side, "conormal", 0.01 if axis == 0 else 0.0)
            for axis in range(2)
            for side in ("lower", "upper")
        )
        if boundary
        else None
    )
    flux = model.diffusive_flux(
        "conduction", state=state, value=CoeffGradient(variable, coefficient), boundaries=physical
    )
    transport_rate = None
    transport_method = None
    if imex:
        from pops.numerics import FiniteVolume, reconstruction, riemann, variables

        transport = model.flux(
            "advection",
            frame=frame,
            state=state,
            components={frame.x: (0.7 * u,), frame.y: (-0.3 * u,)},
            waves={frame.x: (0.7,), frame.y: (-0.3,)},
        )
        balance = model.rate("transport_heat", equation=ddt(state) == -div(transport) + div(flux))
        transport_rate = balance.select(transport)
        rate = balance.select(flux)
        transport_method = FiniteVolume(
            flux=transport,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        )
    if not imex:
        rate = model.rate("heat", equation=ddt(state) == div(flux))
    accumulation = (
        model.local_transform("temperature_to_energy", (u + u * u,), valid_if=u > -0.5)
        if kind == "nonlinear_accumulation"
        else None
    )
    case = pops.Case("composite-implicit-" + kind)
    block = case.block("heat", model)
    numerics = DiscretizationPlan()
    if imex:
        numerics.rates.add(balance, Diffusion(flux=flux, transport=transport_method))
    else:
        numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = pops.Program("composite-implicit-stage")
    temporal = program.state(block[state])
    if invalid or failure_action is not None:
        program.store_history("prior_energy", temporal.n, depth=1)
    coordinates = program.value("coordinates", temporal.n, at=temporal.next.point)
    previous = temporal.n
    if imex:
        explicit = transport_rate(temporal.n)
        previous = program.value(
            "conservative_predictor", temporal.n + program.dt * explicit, at=temporal.next.point
        )
    request = ImplicitDiffusionStage(rate, previous, program.dt, accumulation=accumulation).request(
        unknown=SolveUnknown("coordinate", coordinates),
        seed=temporal.n,
        derivative=DerivativeStrategy("finite_difference"),
    )
    solved = program.solve(
        request,
        solver=Newton(
            tolerance=1e-12,
            max_iterations=newton_iterations,
            linear_tolerance=1e-8,
            linear_max_iterations=100,
            restart=30,
        ),
    ).consume(action=FailRun() if failure_action is None else failure_action)[0]
    conserved = (
        program.transform(solved, transform=accumulation) if accumulation is not None else solved
    )
    candidate = program.value("updated", conserved, at=temporal.next.point)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(step_dt))
    case.program(program)
    if periodic_witness:
        from pops.analytic import cos, x, y

        initial = Analytic(
            frame=frame,
            components=(
                1 + 0.3 * cos(2 * np.pi * (x(frame) - 0.5)) * (1 + 0.1 * cos(2 * np.pi * y(frame))),
            ),
        )
    else:
        initial = Gaussian(
            frame=frame,
            center={frame.x: 0.37, frame.y: 0.43},
            background=1.0,
            amplitude=0.3,
            inverse_width=40.0,
        )
    case.initials.add(
        InitialCondition(state=block[state], value=initial, projection=ConservativeCellAverage())
    )
    grid = CartesianGrid(
        frame=frame, cells=(n, n), periodic=None if boundary else PeriodicAxes(frame.axes)
    )
    if not refined:
        return case, Uniform(grid)
    threshold = case.param(
        RuntimeParam(
            "refine-threshold", default=1 + 0.3 * 1.1 / np.sqrt(2) if periodic_witness else 1.12
        )
    )
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    execution = (
        AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
        if subcycled
        else AMRExecution.synchronous()
    )
    return case, AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(block[state]) > case.value(threshold)),
                # Existing transfer contributes one lookahead cell. Scale the authored buffer
                # to keep its total physical padding exactly 3/16 at every N.
                Buffer(cells=3 * n // 16 - 1 if periodic_witness else 1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
        transfer=transfer,
        execution=execution,
    )


def bind(n, **options):
    case, layout = build(n, **options)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return pops.bind(
        artifact, resources={"execution_context": artifact_execution_context(artifact)}
    )


def mass(runtime, n):
    return sum(
        float(composite_active_block_state(runtime, "heat", level, refinement_ratio=2).sum())
        / (n * 2**level) ** 2
        for level in range(runtime.n_levels())
    )


def assert_no_second_reflux(runtime):
    ledger = tuple(tuple(map(str, row)) for row in runtime._executor.program_flux_ledger_manifest())
    assert not ledger, (
        "composite residual already closes the face flux; accepted reflux must remain empty"
    )


def _periodic_cell_averages(n):
    centers = (np.arange(n) + 0.5) / n
    cx = np.cos(2 * np.pi * (centers - 0.5)) * np.sinc(1 / n)
    cy = np.cos(2 * np.pi * centers) * np.sinc(1 / n)
    return 1 + 0.3 * cx[None, :] * (1 + 0.1 * cy[:, None])


def _conservative_reference_errors(kind, imex, *, periodic_witness, record_property=None):
    errors, independent_checks = [], []
    grids = (16, 32, 64, 128) if imex and periodic_witness else (16, 32, 64)
    for n in grids:
        amr, reference = (
            bind(n, kind=kind, imex=imex, periodic_witness=periodic_witness),
            bind(2 * n, kind=kind, refined=False, imex=imex, periodic_witness=periodic_witness),
        )
        assert amr.n_levels() == 2
        mask0 = composite_active_mask(amr, 0, refinement_ratio=2)
        assert mask0.any() and (~mask0).any(), "the matrix requires an actual coarse/fine interface"
        if periodic_witness:
            expected_mask = np.ones((n, n), dtype=bool)
            expected_mask[:, 3 * n // 16 : 13 * n // 16] = False
            np.testing.assert_array_equal(mask0, expected_mask)
            fine_mask = composite_active_mask(amr, 1, refinement_ratio=2)
            np.testing.assert_array_equal(fine_mask, (~expected_mask).repeat(2, 0).repeat(2, 1))
            for level in range(amr.n_levels()):
                exact = _periodic_cell_averages(n * 2**level)
                mask = composite_active_mask(amr, level, refinement_ratio=2)
                actual = np.asarray(amr.block_level_state_global("heat", level)).reshape(
                    exact.shape
                )
                np.testing.assert_allclose(actual[mask], exact[mask], rtol=0, atol=2e-11)
        reference_initial = np.asarray(reference.state_global("heat")).reshape(2 * n, 2 * n).copy()
        if periodic_witness:
            np.testing.assert_allclose(
                reference_initial, _periodic_cell_averages(2 * n), rtol=0, atol=2e-11
            )
        if not periodic_witness:
            from math import erf

            # The legacy Uniform Gaussian producer integrates this profile analytically.
            edges = np.arange(2 * n + 1) / (2 * n)
            integrals = [
                np.diff([erf(np.sqrt(40) * (edge - center)) for edge in edges])
                * np.sqrt(np.pi)
                / (2 * np.sqrt(40))
                * (2 * n)
                for center in (0.37, 0.43)
            ]
            exact = 1 + 0.3 * integrals[1][:, None] * integrals[0][None, :]
            np.testing.assert_allclose(reference_initial, exact, rtol=0, atol=2e-14)
        before = mass(amr, n)
        for runtime in (amr, reference):
            report = pops.run(runtime, t_end=4 * DT, max_steps=4, console=False)
            assert report.accepted_steps == 4 and report.rejected_steps == 0
        assert abs(mass(amr, n) - before) < 5e-11
        if imex:
            rows = tuple(
                tuple(map(str, row)) for row in amr._executor.program_flux_ledger_manifest()
            )
            assert rows, "explicit predictor must retain its accepted quadrature"
            assert all("provider/4" not in "/".join(row) for row in rows), (
                "implicit diffusion must not enter the accepted transport reflux ledger"
            )
        else:
            assert_no_second_reflux(amr)
        fine = np.asarray(reference.state_global("heat")).reshape(2 * n, 2 * n)
        if kind == "constant" and not imex:
            # An independent periodic centered-Laplacian diagonalization authenticates the
            # complete four-step Uniform BE reference, including the nonperiodic-data transient.
            eigenvalues = 4 * (2 * n) ** 2 * np.sin(np.pi * np.fft.fftfreq(2 * n)) ** 2
            multiplier = (1 + DT * 0.1 * (eigenvalues[:, None] + eigenvalues[None, :])) ** -4
            exact_final = np.fft.ifft2(np.fft.fft2(reference_initial) * multiplier).real
            np.testing.assert_allclose(fine, exact_final, rtol=0, atol=2e-11)
        coarse = fine.reshape(n, 2, n, 2).mean(axis=(1, 3))
        squared = 0.0
        active_fields, active_masks = [], []
        for level, expected in ((0, coarse), (1, fine)):
            mask = composite_active_mask(amr, level, refinement_ratio=2)
            actual = np.asarray(amr.block_level_state_global("heat", level)).reshape(expected.shape)
            squared += float(np.sum((actual[mask] - expected[mask]) ** 2)) / (n * 2**level) ** 2
            active_fields.append(actual[mask])
            active_masks.append(mask)
        errors.append(squared**0.5)
        if imex and periodic_witness:
            from tests.python.support.composite_imex_reference import (
                DT as reference_dt,
                constant_uniform_fourier,
                reference_pair,
            )

            assert DT == reference_dt
            independent = reference_pair(n, kind)
            np.testing.assert_array_equal(active_masks[0], independent["coarse_mask"])
            np.testing.assert_array_equal(active_masks[1], independent["fine_mask"])
            nc = int(active_masks[0].sum())
            expected_active = independent["composite_final"]
            diagnostics = independent["diagnostics"]
            diagnostics["native_pointwise_linf"] = {
                "coarse": float(np.max(np.abs(active_fields[0] - expected_active[:nc]))),
                "fine": float(np.max(np.abs(active_fields[1] - expected_active[nc:]))),
                "uniform": float(np.max(np.abs(fine - independent["uniform_final"]))),
            }
            independent_checks.append(diagnostics)
            np.testing.assert_allclose(
                np.concatenate(active_fields), expected_active, rtol=0, atol=2e-11
            )
            np.testing.assert_allclose(fine, independent["uniform_final"], rtol=0, atol=2e-11)
            if kind == "constant":
                np.testing.assert_allclose(fine, constant_uniform_fourier(2 * n), rtol=0, atol=2e-11)
    if independent_checks and record_property is not None:
        record_property(kind + "_imex_independent_reference_checks", independent_checks)
    return errors


@pytest.mark.parametrize("imex", [False, True])
@pytest.mark.parametrize("kind", ["constant", "variable", "diagonal", "nonlinear_accumulation"])
def test_composite_implicit_matches_conservative_reference_and_converges(
    kind, imex, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    # A periodic smooth solution and a fixed physical interface define this refinement sequence.
    # The former unperiodized Gaussian plus fixed-cell padding changed both seam resolution and
    # interface location with N; its N16→N32 discrepancy was not an asymptotic order witness.
    errors = _conservative_reference_errors(
        kind, imex, periodic_witness=True, record_property=record_property
    )
    # Preserve the original three-grid measurements under their existing evidence key.
    record_property(kind + ("_imex" if imex else "") + "_conservative_reference_l2_errors", errors[:3])
    if imex:
        record_property(kind + "_imex_extended_conservative_reference_l2_errors", errors)
        record_property(kind + "_imex_reference_grids", [16, 32, 64, 128])
        ratios = [errors[1] / errors[2], errors[2] / errors[3]]
        record_property(kind + "_imex_asymptotic_refinement_ratios", ratios)
        # Independent conserved-U FV trajectories reproduce the N16->N32 pre-asymptotic
        # behavior. Keep those fields and errors, and test the same 1.5 reduction on both
        # finer pairs, where the fixed-interface sequence resolves that transient.
        assert errors[2] < errors[1] / 1.5 and errors[3] < errors[2] / 1.5
    else:
        assert errors[1] < errors[0] / 1.5 and errors[2] < errors[1] / 1.5


def test_unperiodized_gaussian_conserves_against_reference_without_an_order_claim(
    isolated_native_cache, native_cxx, kokkos_root, record_property
):
    # Preserve the original six-instance adversarial lifecycle: the Gaussian has a periodic-seam
    # jump, so finite-grid monotonicity is not an appropriate accuracy oracle for this profile.
    errors = _conservative_reference_errors("constant", False, periodic_witness=False)
    record_property("unperiodized_gaussian_conservative_reference_l2_errors", errors)
    assert np.isfinite(errors).all()


def test_composite_implicit_physical_boundary_inventory(
    isolated_native_cache, native_cxx, kokkos_root
):
    n = 32
    runtime = bind(n, boundary=True)
    previous = mass(runtime, n)
    for step in range(4):
        report = pops.run(runtime, t_end=(step + 1) * DT, max_steps=1, console=False)
        assert report.accepted_steps == 1
        records = runtime._executor._program_exchange_records()
        amount = sum(row["integrated_amount"] for row in records)
        assert records and abs(amount - 0.02 * DT) < 2e-12
        current = mass(runtime, n)
        assert abs(current - previous - amount) < 5e-11
        previous = current
        assert_no_second_reflux(runtime)


def accepted_histories(runtime):
    native = runtime._executor
    histories = []
    for name in native.history_names():
        levels = tuple(native.history_levels(name))
        assert levels == tuple(range(runtime.n_levels()))
        histories.append((
            name,
            native.history_ncomp(name),
            native.history_depth(name),
            tuple((
                level,
                native.history_initialized(name, level),
                native.history_fill_count(name, level),
                tuple((
                    np.asarray(native.history_slot_dt(name, level, slot), dtype=np.float64).tobytes(),
                    np.asarray(native.history_global(name, level, slot)).tobytes(),
                ) for slot in range(native.history_depth(name))),
            ) for level in levels),
        ))
    return tuple(histories)


def accepted_envelope(runtime):
    native = runtime._executor
    return (
        runtime.time(),
        runtime.macro_step(),
        tuple(runtime.patch_boxes()),
        tuple(
            np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
            for level in range(runtime.n_levels())
        ),
        native._program_exchange_records(),
        tuple(tuple(map(str, row)) for row in native.program_flux_ledger_manifest()),
        accepted_histories(runtime),
    )


@pytest.mark.parametrize("imex", [False, True])
def test_composite_implicit_failure_restores_every_level_history_and_exchange(
    imex, isolated_native_cache, native_cxx, kokkos_root
):
    runtime = bind(32, invalid=True, imex=imex)

    before = accepted_envelope(runtime)
    with pytest.raises(RuntimeError, match="(?i)(invalid|diffus|spatial)"):
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert accepted_envelope(runtime) == before


def test_composite_implicit_refuses_true_subcycling_before_publication(
    isolated_native_cache, native_cxx, kokkos_root
):
    runtime = bind(16, subcycled=True)
    before = tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
        for level in range(runtime.n_levels())
    )
    with pytest.raises(ValueError, match="synchronized composite field stage requires every child state"):
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert runtime.time() == 0 and runtime.macro_step() == 0
    assert before == tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
        for level in range(runtime.n_levels())
    )


@pytest.mark.parametrize("imex", [False, True])
def test_composite_implicit_numerical_rejection_restores_pending_predictor(
    imex, isolated_native_cache, native_cxx, kokkos_root
):
    from pops.time import RejectAttempt

    runtime = bind(
        32,
        kind="nonlinear_accumulation",
        imex=imex,
        newton_iterations=1,
        failure_action=RejectAttempt(statuses=("iteration_limit",)),
    )
    before = accepted_envelope(runtime)
    with pytest.raises(RuntimeError, match="(?i)(spatial|iteration|reject)"):
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert accepted_envelope(runtime) == before
    report = runtime._executor._last_step_transaction_report
    assert report.action == "reject_attempt"
    assert report.attempts == 1


@pytest.mark.parametrize("imex", [False, True])
def test_composite_implicit_manual_resumption_after_numerical_rejection(
    imex, isolated_native_cache, native_cxx, kokkos_root
):
    from pops.time import RejectAttempt

    options = dict(
        kind="variable",
        imex=imex,
        newton_iterations=1,
        failure_action=RejectAttempt(statuses=("iteration_limit",)),
    )
    runtime = bind(32, step_dt=16 * DT, **options)
    before = accepted_envelope(runtime)
    with pytest.raises(RuntimeError, match="(?i)(spatial|iteration|reject)"):
        pops.run(runtime, t_end=16 * DT, max_steps=1, console=False)
    assert runtime._executor._last_step_transaction_report.action == "reject_attempt"
    assert accepted_envelope(runtime) == before
    # FixedDt's public driver clips to t_end. This is a new caller-requested attempt
    # on the same instance, not an adaptive retry or a changed solver tolerance.
    endpoint = DT / 64
    reference = bind(32, step_dt=endpoint, **options)
    for current in (runtime, reference):
        report = pops.run(current, t_end=endpoint, max_steps=1, console=False)
        assert report.accepted_steps == 1 and report.rejected_steps == 0
    assert runtime.time() == reference.time() == endpoint
    for level in range(runtime.n_levels()):
        np.testing.assert_array_equal(
            runtime.block_level_state_global("heat", level),
            reference.block_level_state_global("heat", level),
        )
    assert (
        runtime._executor._program_exchange_records()
        == reference._executor._program_exchange_records()
    )
    assert accepted_histories(runtime) == accepted_histories(reference)
