"""Public generated-Program proof for a rational, three-level, two-block AMR hierarchy.

The case in this module is intentionally assembled only through the public lifecycle
``validate -> resolve -> compile -> bind``.  Its spatial hierarchy has two transitions while
its temporal authority is independent: ``5/2`` with an explicit final substep followed by ``2/1``.
The native run therefore proves that the generated Program, not a low-level Engine fixture, owns
the two blocks, all three levels, accepted clocks, flux ledger, and reflux transaction.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRemainderPolicy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import Flux, SourceTerm
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import ErrorControlledDt, FixedDt, GuardRole, Program, RejectAttempt


ROOT = Path(__file__).resolve().parents[4]
N = 24
DT = 2.0e-3
STEPS = 3
RETRY_DT = 0.125
SOURCE_RATE = 0.5

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _profile(center: tuple[float, float]) -> np.ndarray:
    """Return a positive, localized scalar profile that tags every requested level."""
    points = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(points, points, indexing="ij")
    value = 1.0 + 0.5 * np.exp(
        -80.0 * ((x - center[0]) ** 2 + (y - center[1]) ** 2)
    )
    return np.ascontiguousarray(value[None, ...], dtype=np.float64)


def _author_case(attempt_policy: str, *, native_cxx: str):
    """Author one public case; policy variants share the exact AMR topology."""
    if attempt_policy not in {"accepted", "forced_reject", "error_retry"}:
        raise ValueError("unknown attempt policy %r" % attempt_policy)

    frame = Rectangle(
        "rational-hierarchy-domain-%s" % attempt_policy,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("rational-hierarchy-model-%s" % attempt_policy, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    if attempt_policy == "accepted":
        flux = model.flux(
            "transport",
            frame=frame,
            state=state,
            components={x_axis: (0.4 * rho,), y_axis: (-0.15 * rho,)},
            waves={x_axis: (0.4 + 0.0 * rho,), y_axis: (-0.15 + 0.0 * rho,)},
        )
        rate = model.rate("transport-rate", equation=ddt(state) == -div(flux))
        source_operator = None
    else:
        # A constant source makes the controller's first rejection deterministic while the
        # zero numerical flux still exercises the complete AMR transaction and rollback path.
        flux = model.flux(
            "zero-transport",
            frame=frame,
            state=state,
            components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
            waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        )
        source = model.source(
            "forcing",
            on=state,
            value=(SOURCE_RATE + 0.0 * rho,),
        )
        source_operator = model.module.operator_handle("forcing")
        rate = model.rate(
            "transport-rate",
            equation=ddt(state) == -div(flux) + source,
        )

    case = pops.Case("rational-hierarchy-case-%s" % attempt_policy)
    rows = []
    for name in ("left", "right"):
        block = case.block(name, model, states=(state,))
        instance = block[state]
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
        rows.append((name, instance))

    if attempt_policy == "accepted":
        routes = tuple(libtime.RungeKuttaRoute(instance, rate) for _, instance in rows)
        program = libtime.RungeKutta(
            routes=routes,
            tableau=libtime.SSPRK2_TABLEAU,
        )
        program.step_strategy(FixedDt(DT))
    else:
        program = Program("rational-hierarchy-program-%s" % attempt_policy)
        strategy = (
            FixedDt(RETRY_DT)
            if attempt_policy == "forced_reject"
            else ErrorControlledDt(
                dt_init=RETRY_DT,
                rtol=1.0e-3,
                atol=1.0e-8,
                dt_min=0.01,
                dt_max=RETRY_DT,
                max_rejections=2,
                shrink=0.5,
                growth=1.25,
            )
        )
        for route, (_name, instance) in enumerate(rows):
            temporal = program.state(instance)
            terms = [Flux()]
            if source_operator is not None:
                terms.append(SourceTerm(source_operator))
            rhs = program.rhs(state=temporal.n, terms=terms)
            candidate = program.value(
                "candidate_%d" % route,
                temporal.n + program.dt * rhs,
                at=temporal.next.point,
            )
            if route == 0:
                if attempt_policy == "forced_reject":
                    candidate = program.guard(
                        "forced_rejection",
                        candidate,
                        program.norm_inf(candidate) < 0.0,
                        action=RejectAttempt(),
                    )
                else:
                    increment = program.value(
                        "candidate_increment",
                        candidate - temporal.n,
                        at=temporal.next.point,
                    )
                    candidate = program.guard(
                        "dt_dependent_error_estimate",
                        candidate,
                        program.norm_inf(increment)
                        <= SOURCE_RATE * strategy.dt_init * strategy.shrink,
                        action=RejectAttempt(),
                        role=GuardRole.ERROR_ESTIMATE,
                    )
            program.commit(temporal.next, candidate)
        program.step_strategy(strategy)
    case.program(program)

    instances = {}
    for name, instance in rows:
        instances[name] = instance
        case.initials.add(
            InitialCondition(
                state=instance,
                value=BindArray(),
                projection=ConservativeCellAverage(),
            )
        )

    threshold = case.param(RuntimeParam("rational_refine_threshold", default=1.05))
    transfer = AMRTransfer()
    for instance in instances.values():
        transfer.state(instance, StateTransfer())

    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=3, ratios=(2, 2)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(instances["left"]) > case.value(threshold)),
                Tag(ValueExpr(instances["right"]) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.subcycled(
            (
                AMRClockRelation(
                    0,
                    1,
                    Fraction(5, 2),
                    AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP,
                ),
                AMRClockRelation(1, 2, 2),
            )
        ),
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    return resolved, instances


def _initial_values(resolved) -> dict[object, np.ndarray]:
    centers = {"left": (0.30, 0.50), "right": (0.70, 0.50)}
    values = {}
    for binding in resolved.initial_condition_plan.bindings:
        block = binding.subject.block_ref.local_id
        values[binding.subject] = _profile(centers[block])
    return values


def _bind(resolved):
    artifact = pops.compile(resolved)
    threshold_slots = tuple(
        slot.handle
        for slot in resolved.bind_schema.runtime_slots
        if slot.handle.local_id == "rational_refine_threshold"
    )
    assert len(threshold_slots) == 1
    runtime = pops.bind(
        artifact,
        params={threshold_slots[0]: 1.05},
        initial_values=_initial_values(resolved),
    )
    return artifact, runtime


def _assert_static_contract(resolved) -> None:
    assert resolved.target == "amr_system"
    assert len(resolved.blocks) == 2
    assert resolved.resolved_hierarchy.plan.level_count == 3
    assert resolved.amr_execution.to_data() == {
        "schema_version": 2,
        "authority_type": "amr_execution",
        "mode": "subcycled",
        "relations": [
            {
                "parent_level": 0,
                "child_level": 1,
                "temporal_ratio": {"numerator": 5, "denominator": 2},
                "remainder_policy": "explicit_final_substep",
            },
            {
                "parent_level": 1,
                "child_level": 2,
                "temporal_ratio": {"numerator": 2, "denominator": 1},
                "remainder_policy": "integral_only",
            },
        ],
    }
    assert len(resolved.initial_condition_plan.bindings) == 2
    assert len(resolved.bootstrap_plan.actions) >= 16


def test_public_generated_rational_three_level_two_block_program(
    native_cxx,
    isolated_native_cache,
    kokkos_root,
):
    """Generated Program execution authenticates topology, clocks, reflux, and conservation."""
    del isolated_native_cache, kokkos_root
    resolved, instances = _author_case("accepted", native_cxx=native_cxx)
    _assert_static_contract(resolved)
    _artifact, runtime = _bind(resolved)

    assert runtime.block_names() == ("left", "right")
    assert runtime.n_levels() == 3
    boxes = tuple(runtime.patch_boxes())
    assert boxes and {int(row[0]) for row in boxes} == {0, 1, 2}
    before_state = {
        (name, level): np.asarray(
            runtime.block_level_state_global(name, level), dtype=np.float64
        ).copy()
        for name in instances
        for level in range(3)
    }
    before_mass = {
        name: runtime.integral(name, levels=(0,))
        for name in instances
    }

    report = pops.run(
        runtime,
        t_end=STEPS * DT,
        max_steps=STEPS,
        console=False,
    )
    assert report.accepted_steps == STEPS
    assert report.final_macro_step == STEPS
    assert report.final_time == pytest.approx(STEPS * DT, rel=0.0, abs=2.0e-15)
    assert runtime.n_levels() == 3

    for name in instances:
        after = runtime.integral(name, levels=(0,))
        assert abs(after - before_mass[name]) < 1.0e-8
        for level in range(3):
            values = np.asarray(
                runtime.block_level_state_global(name, level), dtype=np.float64
            )
            assert values.size == before_state[(name, level)].size
            assert np.isfinite(values).all()

    program_report = runtime.program_report()
    assert program_report.level_relations == [
        {
            "parent_level": 0,
            "child_level": 1,
            "temporal_ratio": {"numerator": 5, "denominator": 2},
            "remainder_policy": "explicit_final_substep",
        },
        {
            "parent_level": 1,
            "child_level": 2,
            "temporal_ratio": {"numerator": 2, "denominator": 1},
            "remainder_policy": "integral_only",
        },
    ]
    level_clocks = [row for row in program_report.clocks if row["kind"] == "level"]
    assert {row["level"] for row in level_clocks} == {0, 1, 2}
    assert all(row["macro_step"] == STEPS for row in level_clocks)
    assert {row["level"] for row in program_report.flux_ledger} == {0, 1, 2}
    assert all(row["substep_duration"] > 0.0 for row in program_report.flux_ledger)

    phase_denominators = {
        level: {
            row["phase"]["denominator"]
            for row in program_report.flux_ledger
            if row["level"] == level
        }
        for level in range(3)
    }
    # The 5/2 relation materializes 0, 2/5, 4/5, 1 child windows; the next 2/1 relation
    # materializes the half-window on level two.  These are accepted ledger identities, not
    # inferred from the authored descriptor alone.
    assert 5 in phase_denominators[1]
    assert 2 in phase_denominators[2]

    sync_pairs = Counter(
        (row["parent_level"], row["child_level"])
        for row in program_report.synchronization
    )
    assert set(sync_pairs) == {(0, 1), (1, 2)}
    assert sync_pairs[(0, 1)] == 4
    assert sync_pairs[(1, 2)] == 4

    # The accepted state must differ from the bind image while preserving its exact shape, proving
    # that the generated candidates actually crossed the hidden publish/reflux boundary.
    assert any(
        not np.array_equal(
            before_state[(name, 0)],
            np.asarray(runtime.block_level_state_global(name, 0), dtype=np.float64),
        )
        for name in instances
    )

    rejected_plan, rejected_instances = _author_case(
        "forced_reject", native_cxx=native_cxx
    )
    _assert_static_contract(rejected_plan)
    _rejected_artifact, rejected = _bind(rejected_plan)
    rejected_before = {
        (name, level): np.asarray(
            rejected.block_level_state_global(name, level), dtype=np.float64
        ).copy()
        for name in rejected_instances
        for level in range(3)
    }
    rejected_clock = (rejected.time(), rejected.macro_step())
    from pops._bootstrap import StepAttemptRejected

    with pytest.raises(StepAttemptRejected):
        pops.run(rejected, t_end=RETRY_DT, max_steps=1, console=False)
    assert (rejected.time(), rejected.macro_step()) == rejected_clock
    for key, expected in rejected_before.items():
        name, level = key
        np.testing.assert_array_equal(
            np.asarray(rejected.block_level_state_global(name, level), dtype=np.float64),
            expected,
        )
    assert rejected.program_report().temporal["transaction_stats"] == {
        "accepted": 0,
        "rejected": 1,
        "failed": 0,
    }

    retry_plan, retry_instances = _author_case("error_retry", native_cxx=native_cxx)
    _assert_static_contract(retry_plan)
    _retry_artifact, retrying = _bind(retry_plan)
    retry_initial = {
        name: np.asarray(
            retrying.block_level_state_global(name, 0), dtype=np.float64
        ).copy()
        for name in retry_instances
    }
    retry_report = pops.run(
        retrying,
        t_end=RETRY_DT,
        max_steps=2,
        console=False,
    )
    assert retry_report.accepted_steps == 2
    assert retry_report.rejected_steps == 1
    assert retrying.macro_step() == 2
    assert retrying.time() == pytest.approx(RETRY_DT, rel=0.0, abs=2.0e-15)
    for name in retry_instances:
        actual = np.asarray(
            retrying.block_level_state_global(name, 0), dtype=np.float64
        )
        np.testing.assert_allclose(
            actual,
            retry_initial[name] + SOURCE_RATE * RETRY_DT,
            rtol=0.0,
            atol=2.0e-13,
        )
    assert retrying.program_report().temporal["transaction_stats"] == {
        "accepted": 2,
        "rejected": 1,
        "failed": 0,
    }


def test_public_rational_hierarchy_rejects_implicit_fractional_remainder_before_mutation():
    """The typed authoring boundary refuses a 5/2 relation without an explicit remainder policy."""
    with pytest.raises(ValueError, match="EXPLICIT_FINAL_SUBSTEP"):
        AMRClockRelation(0, 1, Fraction(5, 2))

    hierarchy = AMRHierarchy(max_levels=3, ratios=(2, 2))
    relation = AMRClockRelation(
        0,
        1,
        Fraction(5, 2),
        AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP,
    )
    execution = AMRExecution.subcycled(
        (relation, AMRClockRelation(1, 2, 2))
    )
    # Both valid descriptors remain detached immutable values; no native bind/mutation occurred
    # while the invalid relation was rejected.
    assert hierarchy.to_data()["max_levels"] == 3
    assert execution.to_data()["relations"][0]["remainder_policy"] == (
        "explicit_final_substep"
    )
