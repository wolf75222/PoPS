"""Public read-only dt-bound capture, exact routing and refusal witnesses."""
import hashlib
import json

import pops
import pytest

from pops.codegen.program_codegen import _emit_dt_bound, emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt
from pops.time._program.dt_bound import readonly_dt_bound_nodes


def _case(kind):
    frame = Rectangle("dt_frame", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("dt_model", frame=frame)
    state = model.state("U", components=("a", "b"))
    case = pops.Case("dt_case")
    block = case.block("fluid", model)
    program = pops.Program("dt_program")
    u = program.state(block[state])
    old = u.n
    program.commit(u.next, program.value("identity", (old[0], old[1]), at=u.next.point))
    program.step_strategy(FixedDt(.1))
    if kind == "readonly_block":
        bound = program.state(case.block("bound_data", model)[state])
        program.set_dt_bound(lambda P, cfl: cfl / (1 + P.dot_all(bound.n, bound.n)))
    elif kind == "existing_state":
        program.set_dt_bound(lambda P, cfl: cfl / (1 + P.dot_all(u.n, u.n)))
    elif kind == "scalar_capture":
        norm = program.dot_all(old, old)
        scalar = 1 + norm
        program.set_dt_bound(lambda P, cfl: cfl / (scalar * scalar))
    elif kind == "legacy":
        program.set_dt_bound(lambda P, cfl: cfl * P.hmin())
    elif kind != "unbounded":
        raise ValueError(kind)
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, program, layout, u


@pytest.mark.parametrize("kind", ["existing_state", "readonly_block", "scalar_capture"])
def test_public_resolved_bound_closes_captures_and_routes_readonly_states(kind):
    case, program, layout, _ = _case(kind)
    initial_nodes = tuple(program._values)
    initial_bound = tuple(program._dt_bound[0])
    initial_hash = program._ir_hash()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    _, body, _ = _emit_dt_bound(resolved.time, graph)
    assert body.count("ctx.dot_all(") == 1
    assert source.count("ctx.dot_all(") == (2 if kind == "scalar_capture" else 1)
    routes = resolved.time._block_indices()
    assert [block.local_id for block in routes] == (
        ["fluid", "bound_data"] if kind == "readonly_block" else ["fluid"])
    owner = 1 if kind == "readonly_block" else 0
    assert f"ctx.state({owner})" in body and f"ctx.dot_all({owner}," in body
    assert "commit" not in body
    if owner:
        assert "ctx.state(0)" not in body
    assert tuple(program._values) == initial_nodes
    assert tuple(program._dt_bound[0]) == initial_bound
    assert program._ir_hash() == initial_hash
    assert len(program.commits()) == 1
    if kind == "scalar_capture":
        closure = readonly_dt_bound_nodes(program)
        ids = [node.id for node in closure]
        assert len(ids) == len(set(ids))
        position = {node.id: i for i, node in enumerate(closure)}
        assert all(position[dep.id] < position[node.id]
                   for node in closure for dep in node.inputs)


@pytest.mark.parametrize("kind", ["legacy", "unbounded"])
def test_existing_graph_identity_and_source_remain_exact(kind):
    case, program, layout, _ = _case(kind)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    semantic = json.dumps(program._semantic_serialize(), sort_keys=True, separators=(",", ":"))
    actual = (program._ir_hash(), hashlib.sha256(semantic.encode()).hexdigest(),
              hashlib.sha256(source.encode()).hexdigest())
    assert actual == BASELINE[kind]


@pytest.mark.parametrize("capture", ["update", "derived_update", "branch"])
def test_inadmissible_captured_dag_refuses_atomically(capture):
    _, program, _, u = _case("unbounded")
    if capture == "branch":
        value = program.branch(program.norm2(u.n) > 0, lambda P: u.n, lambda P: u.n)
    else:
        value = program.value("captured_update", 2 * u.n, at=u.next.point)
    scalar = program.norm2(value)
    before = (program._next_id, tuple(program._values), program._ir_hash())
    builder = (lambda P, cfl: cfl / (1 + scalar)) if capture != "update" else (
        lambda P, cfl: cfl / (1 + P.norm2(value)))
    with pytest.raises(ValueError, match="body and captures.*op .*not allowed"):
        program.set_dt_bound(builder)
    assert program._dt_bound is None
    assert (program._next_id, tuple(program._values), program._ir_hash()) == before


def test_long_shared_capture_dag_has_no_python_recursion_limit():
    _, program, _, u = _case("unbounded")
    scalar = program.norm2(u.n)
    for _ in range(1500):
        scalar = scalar + 1
    program.set_dt_bound(lambda P, cfl: cfl / (scalar + scalar))
    closure = readonly_dt_bound_nodes(program)
    ids = [node.id for node in closure]
    assert len(ids) == len(set(ids)) == 1505


@pytest.mark.parametrize("mutation", ["strategy", "cadence", "cell_time", "integral",
                                      "freeze", "stage", "history", "retime"])
def test_callback_cannot_hide_temporal_or_publication_metadata_changes(mutation):
    _, program, _, u = _case("unbounded")
    before = (program._serialize(), program._next_id, program._next_region)

    def builder(P, cfl):
        if mutation == "strategy":
            P.step_strategy(FixedDt(.2))
        elif mutation == "cadence":
            P.cadence(substeps=2)
        elif mutation == "cell_time":
            P.cell_local_time(tick_denominator=2)
        elif mutation == "integral":
            P.integral_state("hidden", initial=1.)
        elif mutation == "freeze":
            result = cfl * P.hmin()
            P.freeze()
            return result
        elif mutation == "stage":
            P.stage("hidden", c=.5)
        elif mutation == "history":
            P.keep_history(u, depth=2)
        elif mutation == "retime":
            P.value("hidden_retime", u.n, at=u.next.point)
        return cfl * P.hmin()

    error = RuntimeError if mutation == "freeze" else ValueError
    diagnostic = "cannot run while" if mutation == "freeze" else (
        "read-only callback changed authoring metadata")
    with pytest.raises(error, match=diagnostic):
        program.set_dt_bound(builder)
    assert (program._serialize(), program._next_id, program._next_region) == before
    assert not program.has_dt_bound() and not program._frozen
    # Exact retry observes the restored graph and preserves its block route.
    program.set_dt_bound(lambda P, cfl: cfl / (1 + P.norm2(u.n)))
    assert list(program._block_indices().values()) == [0]


def test_callback_cannot_add_a_hidden_commit_of_a_preexisting_candidate():
    frame = Rectangle("mutation_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("mutation_model", frame=frame)
    state = model.state("U", components=("a", "b"))
    case = pops.Case("mutation_case")
    program = pops.Program("mutation_program")
    q = program.state(case.block("fluid", model)[state])
    passive = program.state(case.block("side", model)[state])
    program.commit(q.next, program.value("identity", (q.n[0], q.n[1]), at=q.next.point))
    candidate = program.value("side_update", (2 * passive.n[0], 3 * passive.n[1]),
                              at=passive.next.point)
    program.step_strategy(FixedDt(.1))
    before = program._serialize(), program._next_id

    def builder(P, cfl):
        P.commit(passive.next, candidate)
        return cfl * P.hmin()

    with pytest.raises(ValueError, match="read-only callback.*_commits"):
        program.set_dt_bound(builder)
    assert (program._serialize(), program._next_id) == before
    assert len(program.commits()) == 1 and not program.has_dt_bound()
    program.set_dt_bound(lambda P, cfl: cfl / (1 + P.norm2(passive.n)))
    assert len(program.commits()) == 1


# Filled from public source-only emission on the exact pre-fix 5d97b7a5 tree.
BASELINE = {
    "legacy": (
        "18429f0fec1e96ade028876bfa701f515c1daf1cc0dd3ba31641fc24b9eb26b9",
        "a6d201843cba0abde900150c0046591000079bd8ff7a5c5e55a4898696b44fb8",
        "d1c04d58f131accf87f542560dc875a9fbbd61420413dbe3f6b0606dbc628d17"),
    "unbounded": (
        "7f5df6088cd631eb81e94f5cb59c9bd9db0dd46289563d8ef7d4aad7db94cd03",
        "1738ef39f5061963cc4fc1862c4cacb68f20b34f9a55cdc55a54e793cef7c14b",
        "def72c8c684d815d60a5ffc523e3162a087363b07ea99f4e3860093a7d046088"),
}
