"""Inventory claims use exact post-affine recipient weights and physical maps."""
from fractions import Fraction

import pytest
import pops
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_interaction_exchanges import accepted_interaction_quadrature
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.numerics import DiscretizationPlan, JointEvaluation
from pops.physics.interaction_inventories import InventoryProjection, PhysicalInventoryMap
from tests.python.support.layout_plan import cartesian_grid


def interaction_case(*, n=16, left_weight=1, right_weight=1, repeated=False,
                     inventories=True, native_function=None, stages=1, unshared=False,
                     transform_accepted=False, guarded=False):
    frame = Rectangle("domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("pair", frame=frame)
    left = model.species("a", state=("p", "E"))
    right = model.species("b", state=("p", "E", "m"))
    force = right[0] / right[2] - left[0]
    power = (left[0] + right[0] / right[2]) * force / 2
    outputs = {left: (force, power), right: (-force, -power, 0)}
    if native_function is not None:
        call = native_function(left, right)
        outputs = {left: call.left, right: call.right}
    maps = (
        PhysicalInventoryMap("momentum", (InventoryProjection(left, (1, 0)),
                                          InventoryProjection(right, (1, 0, 0)))),
        PhysicalInventoryMap("total_energy", (InventoryProjection(left, (0, 1)),
                                              InventoryProjection(right, (0, 1, 0)))),
    )
    if unshared and inventories:
        raise ValueError("unshared cost baseline omits inventory instrumentation on both compared routes")
    application = model.interaction("exchange", outputs=outputs, preserves=maps if inventories else None)
    right_application = model.interaction("independent_exchange", outputs=outputs) if unshared else application
    left_rate = model.rate("a_rate", equation=ddt(left) == (
        application[left] + application[left] if repeated else application[left]))
    right_rate = model.rate("b_rate", equation=ddt(right) == right_application[right])
    if guarded and (inventories or stages != 1):
        raise ValueError("guard witness declares one-stage execution without inventory claims")
    transform = model.local_transform("square_momentum", (left[0] ** 2, left[1]), on=left) if transform_accepted else None
    case = pops.Case("pair")
    blocks = [case.block("left", model, states=(left,)), case.block("right", model, states=(right,))]
    for block, rate, state in zip(blocks, (left_rate, right_rate), (left, right), strict=True):
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, JointEvaluation(state))
        case.numerics(numerics, block=block)
    program = pops.Program("step")._bind_operators(model.module)
    a, b = (program.state(block[state]) for block, state in zip(blocks, (left, right), strict=True))
    if guarded:
        an,bn = a.n,b.n
        condition = program.norm2(an) > 0
        a_next = program.branch(condition,
            lambda P: P.value("a_active",an+P.dt*left_rate(an,bn),at=a.next.point),
            lambda P: P.value("a_inactive",an,at=a.next.point))
        b_next = program.branch(condition,
            lambda P: P.value("b_active",bn+P.dt*right_rate(bn,an),at=b.next.point),
            lambda P: P.value("b_inactive",bn,at=b.next.point))
        program.commit(a.next,a_next)
        program.commit(b.next,b_next)
        program.step_strategy(pops.time.FixedDt(.001))
        case.program(program)
        return case,program,model,maps
    ra, rb = left_rate(a.n, b.n), right_rate(b.n, a.n)
    if stages == 2:
        stage = program.stage("predictor", c=1)
        ap = program.value("a_predictor", a.n + program.dt * ra, at=stage)
        bp = program.value("b_predictor", b.n + program.dt * rb, at=stage)
        ra2, rb2 = left_rate(ap, bp), right_rate(bp, ap)
        ra = program.value("a_quadrature", Fraction(1, 2) * ra + Fraction(1, 2) * ra2, at=a.next.point)
        rb = program.value("b_quadrature", Fraction(1, 2) * rb + Fraction(1, 2) * rb2, at=b.next.point)
    elif stages != 1:
        raise ValueError("matrix supports explicitly declared Euler or two-stage Heun")
    a_next = program.value("a_next", a.n + left_weight * program.dt * ra, at=a.next.point)
    if transform is not None:
        a_next = transform(a_next)
    program.commit(a.next, a_next)
    program.commit(b.next, program.value("b_next", b.n + right_weight * program.dt * rb, at=b.next.point))
    program.step_strategy(pops.time.FixedDt(.001))
    case.program(program)
    return case, program, model, maps


def test_post_affine_inventory_quadrature_rejects_unequal_weights_and_repeated_use():
    for options in ({"left_weight": Fraction(1, 2)}, {"repeated": True}):
        _case, program, _model, _maps = interaction_case(**options)
        with pytest.raises(ValueError, match="nonconservative"):
            accepted_interaction_quadrature(program)



def test_accepted_state_transform_cannot_drop_joint_inventory():
    _case, program, _model, _maps = interaction_case(transform_accepted=True)
    with pytest.raises(ValueError, match="unproved local_transform quadrature"):
        accepted_interaction_quadrature(program)


def test_nonlinear_fresh_stage_rate_does_not_inherit_predictor_inventory():
    _case, program, _model, _maps = interaction_case(stages=2)
    rows = accepted_interaction_quadrature(program)
    assert len(rows) == 8
    assert all(weights == {1: Fraction(1, 2)} for _map, _row, weights in rows)

def test_repeated_occurrences_preserved_with_compensating_accepted_weight():
    _case, program, _model, _maps = interaction_case(repeated=True, left_weight=Fraction(1, 2))
    rows = accepted_interaction_quadrature(program)
    assert len(rows) == 6  # three mathematical occurrences in each of two inventory rows
    assert sorted(weights[1] for _map, _row, weights in rows) == [Fraction(1, 2)] * 4 + [1, 1]


def test_full_case_source_has_one_joint_kernel_and_exact_physical_ledger():
    case, _program, _model, _maps = interaction_case()
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    graph = build_program_model_graph(resolved)
    for block in resolved.blocks:
        lowered = graph.model_for_block(block.name)
        assert lowered._m._program_only_storage_axes == ("x", "y")
        assert not lowered._m._flux_terms
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert source.count("consume_pointwise_evaluation_status") == 1
    assert source.count("ctx.stage_exchange(") == 4
    assert source.index("ctx.stage_exchange(") < source.index("ctx.commit_many(")
    assert "ctx.sum_component(0," in source and "ctx.sum_component(1," in source
    assert source.count("ctx.geometry().spacing(axis)") == 1
    from pops.time._program.detach import detach_compiled_program
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    assert len(accepted_interaction_quadrature(detached)) == 4


def test_inventory_refuses_foreign_same_named_recipient():
    _case, _program, model, _maps = interaction_case(inventories=False)
    foreign = pops.Model("foreign").state("a", components=("p", "E"))
    bad = PhysicalInventoryMap("momentum", (InventoryProjection(foreign, (1, 0)),))
    a, b = model._states.values()
    with pytest.raises(ValueError, match="exact interaction recipient"):
        model.interaction("bad", outputs={a: (b[0], 0), b: (-b[0], 0, 0)}, preserves=bad)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_heterogeneous_inventory_updates_and_accepted_ledger(record_property):
    import time
    from pathlib import Path
    import numpy as np
    from pops._native_selector import select_native_dimension
    from tests.python.support.native_execution_context import artifact_execution_context
    select_native_dimension(2)
    case, _program, _model, _maps = interaction_case(n=16)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    start = time.perf_counter()
    artifact = pops.compile(resolved)
    record_property("compile_seconds", time.perf_counter() - start)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)
    initial = {"left": np.stack([np.full((16, 16), value) for value in (2., 5.)]),
               "right": np.stack([np.full((16, 16), value) for value in (1., 3., 2.)])}
    simulation = pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)})
    before = [simulation.integral("left", component) + simulation.integral("right", component)
              for component in range(2)]
    report = pops.run(simulation, t_end=.1, max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    a, b = np.asarray((2., 5.)), np.asarray((1., 3., 2.))
    for _ in range(100):
        force = b[0] / b[2] - a[0]
        power = (a[0] + b[0] / b[2]) * force / 2
        a += .001 * np.asarray((force, power))
        b += .001 * np.asarray((-force, -power, 0))
    for name, expected in (("left", a), ("right", b)):
        actual = np.asarray(simulation.state_global(name))
        np.testing.assert_allclose(actual, expected[:, None, None] * np.ones((1, 16, 16)),
                                   rtol=0, atol=1e-11)
    after = [simulation.integral("left", component) + simulation.integral("right", component)
             for component in range(2)]
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-11)
    thermal = a[1] - a[0] ** 2 / 2 + b[1] - b[0] ** 2 / (2 * b[2])
    assert thermal > 3 + 2.75
    engine = simulation._executor_for_block("left")
    rows = engine._program_exchange_records()
    assert len(rows) == 4
    for inventory in ("momentum", "total_energy"):
        selected = [row for row in rows if row["occurrence_identity"].endswith(":" + inventory)]
        assert len(selected) == 2
        assert abs(sum(row["integrated_amount"] for row in selected)) < 1e-11
    record_property("accepted_steps", report.accepted_steps)
    record_property("accepted_exchange_records_last_step", len(rows))
