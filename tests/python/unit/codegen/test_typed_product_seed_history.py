"""History halos match actual typed-product allocations before and after detach."""
from __future__ import annotations

import pytest
import pops

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
from pops.codegen.program_history_identity import history_ghost_depth
from pops.codegen.program_models import ProgramModelGraph
from pops.time import FailRun
from pops.time._program.detach import detach_compiled_program
from tests.python.support.ns_conduction import build, CONTROLS, FD_STEP


def _resolved(case, layout):
    plan = pops.resolve(pops.validate(case), layout=layout)
    return plan.time, ProgramModelGraph.from_resolved_blocks(plan.blocks), detach_compiled_program(plan.time)


def _seed(program, name):
    return next(value.inputs[0] for value in program._values
                if value.op == "store_history" and value.attrs["history"] == name)


def _halo_metadata(source):
    return source.split("pops_program_checkpoint_history_ghost_depth", 1)[1].split("\n}", 1)[0]


def _assert_storage(program, source, name, width, depth, seed_id):
    assert history_ghost_depth(program, name) == depth
    registration = next(line for line in source.splitlines()
                        if 'ctx.register_history("%s"' % name in line)
    assert registration.endswith(", %d);" % depth)
    seed_allocation = next(line for line in source.splitlines()
                           if 'seed_output_%d = &ctx.scalar_scratch' % seed_id in line)
    assert seed_allocation.endswith(", %d, 0);" % width)


@pytest.mark.parametrize("repeat_store", (False, True))
def test_real_ns_seed_history_preserves_cpp_and_checkpoint_halo(repeat_store):
    case, layout, program = build(8)
    seed = _seed(program, "primitive-seed")
    if repeat_store:
        program.store_history("primitive-seed", seed, depth=1)
    live, graph, detached = _resolved(case, layout)
    metadata_images = []
    for program in (live, detached):
        source = emit_cpp_program(program, model_graph=graph)
        _assert_storage(program, source, "primitive-seed", 3, 0, seed.id)
        assert history_ghost_depth(program, "issued-convective-Q") == -1
        assert history_ghost_depth(program, "Euler-RHS") == -1
        for name in ("density", "velocity", "temperature"):
            assert history_ghost_depth(program, name) == 1
        metadata_images.append({name: history_ghost_depth(program, name)
                                for name in program._histories})
    assert metadata_images[0] == metadata_images[1]


def _scalar_product(*, extra_store):
    from pops._ir.quantity import PhysicalDimension, PhysicalSupport
    from pops.domain import CartesianDomain
    from pops.fields import FieldBoundary, FieldDiscretization, bcs
    from pops.fields.methods import CellCenteredNonlinearCoupled
    from pops.frames import Cartesian1D
    from pops.layouts import Uniform
    from pops.math import Reaction
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.model.spaces import FieldSpace
    from pops.solvers import Newton
    from pops.time import EvolvedOriginalFieldStage, EvolvedOriginalFieldRate, FixedDt

    frame = CartesianDomain("scalar-product-domain", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("scalar-product-material", frame=frame)
    state = model.state("Q", components=("amount",), sampling="cell_average",
                        support=PhysicalSupport((("x", "scalar-product-domain"),)))
    unknown = model.field("phi")
    case = pops.Case("scalar-product-history")
    block = case.block("material", model)
    program = pops.Program("scalar-product-step")
    temporal = program.state(block[state])
    issued = program.value("issued", 1*temporal.n, at=temporal.next.point)
    stage = EvolvedOriginalFieldStage("scalar-product-stage", unknowns=(unknown,),
        evolved_unknowns=(unknown,), accumulation=(Reaction(unknown, 1),),
        spatial_rhs=(EvolvedOriginalFieldRate(spatial=None, additive=0),),
        previous=tuple(state), previous_scope="issued",
        tau=program.temporal_tau(program.dt, at=temporal.next.point),
        boundaries=(FieldBoundary(unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),))
    solver = Newton(**CONTROLS)
    field = case.field(stage.problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=FD_STEP, face_policy="Arithmetic@1", coefficient_evaluation="PerCandidate@1"),
        boundaries=(), solver=solver))
    space = FieldSpace("scalar-product", components=("phi",), representation="primitive",
        centering="cell", sampling="cell_value", units=(PhysicalDimension(),),
        frame=state.space.frame, clock=state.space.clock, support=state.space.support)
    request = field.bind_program_inputs(program=program, values={block[state]: issued},
                                       at=temporal.next.point, solver=solver)
    request = request.seed_product(program=program, name="scalar-seed", space=space,
                                   expressions=(issued[0],))
    seed = request.seeds["field_tuple"]
    program.store_history("seed", seed, depth=1)
    if extra_store == "guarded":
        guarded = program.guard("seed-domain", seed, program.norm2(seed) > 0, action=FailRun())
        program.store_history("seed", guarded, depth=1)
    else:
        named = program.value("named-seed", seed)
        program.store_history("seed" if extra_store == "mixed" else "named-seed", named, depth=1)
    if extra_store in ("copy", "transitive-copy", "anchored-cycle", "copy-conflict"):
        loaded = program.history("seed", ncomp=1, block=seed.block)
        program.store_history("named-seed" if extra_store == "copy-conflict" else "copy",
                              loaded, depth=1)
        if extra_store == "transitive-copy":
            copied = program.history("copy", ncomp=1, block=seed.block)
            program.store_history("second-copy", copied, depth=1)
        elif extra_store == "anchored-cycle":
            copied = program.history("copy", ncomp=1, block=seed.block)
            program.store_history("seed", copied, depth=1)
    elif extra_store == "reverse-anchor":
        loaded = program.history("unwritten", ncomp=1, block=seed.block)
        program.store_history("seed", loaded, depth=1)
    elif extra_store == "read-first-cycle":
        a = program.history("cycle-a", ncomp=1, block=seed.block)
        b = program.history("cycle-b", ncomp=1, block=seed.block)
        program.store_history("cycle-a", b, depth=1)
        program.store_history("cycle-b", a, depth=1)
    elif extra_store == "state-copy":
        program.store_history("accepted-state", issued, depth=1)
        loaded = program.history("accepted-state", ncomp=1, block=seed.block)
        program.store_history("state-copy", loaded, depth=1)
    observed = field.observe(program.solve(request, solver=solver).consume(action=FailRun()))
    program.commit(temporal.next, observed.evolved_state(target=temporal.next))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8,), periodic=PeriodicAxes(frame.axes)))
    live, graph, detached = _resolved(case, layout)
    return live, graph, detached, seed.id


@pytest.mark.parametrize("extra_store", ("guarded", "named"))
def test_scalar_product_aliases_keep_their_actual_allocation(extra_store):
    live, graph, detached, seed_id = _scalar_product(extra_store=extra_store)
    metadata_images = []
    for program in (live, detached):
        source = emit_cpp_program(program, model_graph=graph)
        _assert_storage(program, source, "seed", 1, 0, seed_id)
        if extra_store == "named":
            assert history_ghost_depth(program, "named-seed") == 1
        checkpoint_metadata = _emit_checkpoint_shape_metadata(program)
        index = sorted(program._histories).index("seed")
        assert "case %d: return 0;" % index in _halo_metadata(checkpoint_metadata)
        metadata_images.append(_halo_metadata(checkpoint_metadata))
    assert metadata_images[0] == metadata_images[1]


def test_same_history_refuses_distinct_scalar_seed_and_named_allocation_halos():
    live, graph, detached, _seed_id = _scalar_product(extra_store="mixed")
    for program in (live, detached):
        with pytest.raises(ValueError, match="stores disagree on their exact ghost depth"):
            history_ghost_depth(program, "seed")
        with pytest.raises(ValueError, match="stores disagree on their exact ghost depth"):
            emit_cpp_program(program, model_graph=graph)


@pytest.mark.parametrize("route,expected", (
    ("copy", {"copy": 0}),
    ("transitive-copy", {"copy": 0, "second-copy": 0}),
    ("anchored-cycle", {"copy": 0}),
    ("reverse-anchor", {"unwritten": 0}),
    ("read-first-cycle", {"cycle-a": 1, "cycle-b": 1}),
    ("state-copy", {"accepted-state": -1, "state-copy": -1}),
))
def test_public_history_copies_and_cycles_follow_exact_allocation_anchors(route, expected):
    live, graph, detached, seed_id = _scalar_product(extra_store=route)
    expected = {"seed": 0, "named-seed": 1, **expected}
    metadata_images = []
    for program in (live, detached):
        source = emit_cpp_program(program, model_graph=graph)
        _assert_storage(program, source, "seed", 1, 0, seed_id)
        assert "ctx.history_zero_start(" in source
        metadata = _halo_metadata(_emit_checkpoint_shape_metadata(program))
        for name, depth in expected.items():
            assert history_ghost_depth(program, name) == depth
            registration = next(line for line in source.splitlines()
                                if 'ctx.register_history("%s"' % name in line)
            assert registration.endswith(", %d);" % depth)
            index = sorted(program._histories).index(name)
            assert "case %d: return %d;" % (index, depth) in metadata
        metadata_images.append(metadata)
    assert metadata_images[0] == metadata_images[1]


def test_history_copy_refuses_contradictory_anchors_across_rings():
    live, graph, detached, _seed_id = _scalar_product(extra_store="copy-conflict")
    for program in (live, detached):
        for name in ("seed", "named-seed"):
            with pytest.raises(ValueError, match="stores disagree on their exact ghost depth"):
                history_ghost_depth(program, name)
        with pytest.raises(ValueError, match="stores disagree on their exact ghost depth"):
            emit_cpp_program(program, model_graph=graph)


@pytest.mark.parametrize("alias", ("guarded", "named"))
def test_ns_width_three_aliases_preserve_the_existing_width_refusal(alias):
    _case, _layout, program = build(8)
    seed = _seed(program, "primitive-seed")
    value = (program.guard("seed-domain", seed, program.norm2(seed) > 0, action=FailRun())
             if alias == "guarded" else program.value("named-seed", seed))
    with pytest.raises(ValueError, match="scalar history component width changed"):
        program.store_history("primitive-seed", value, depth=1)
