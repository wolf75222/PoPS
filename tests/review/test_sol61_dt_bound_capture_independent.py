"""Independent resolved emission and atomic capture refusal, without native JIT."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pops
import pytest

from pops.codegen.program_codegen import _emit_dt_bound, emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.time import FixedDt
from tests.python.integration.runtime.test_dt_bound_capture_independent_runtime import (
    arrays, authored_capture, bound_duration,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "9c4209a5e5ecb05ac71a62f4d2c21b6e3acd4300"


@pytest.mark.parametrize("capture", ("main_current", "query_only"))
@pytest.mark.parametrize("order", ((0, 1, 2), (2, 0, 1)))
def test_native_fixture_is_authentic_resolved_and_routes_query_owner(capture, order):
    case, layout, program, _, _ = authored_capture(capture, order)
    before = copy.deepcopy(program._serialize())
    resolved = pops.resolve(pops.validate(case), layout=layout)
    bindings = resolved.initial_condition_plan.bindings
    expected = {"fluid"} if capture == "main_current" else {"fluid", "query_only"}
    assert {b.subject.block_ref.local_id for b in bindings} == expected
    assert all(b.subject.is_resolved and b.subject.kind == "state" for b in bindings)
    assert all(b.source.options.to_data()["native_route"] == "bound_level_zero"
               for b in bindings)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    cpp = emit_cpp_program(resolved.time, model_graph=graph)
    _, bound, _ = _emit_dt_bound(resolved.time, graph)
    owner = 0 if capture == "main_current" else 1
    assert f"ctx.dot_all({owner}," in bound and f"ctx.state({owner})" in bound
    assert "commit" not in bound and bound.count("ctx.dot_all(") == 1
    assert "pops_install_program" in cpp and program._serialize() == before
    routes = resolved.time._block_indices()
    assert [(b.local_id, i) for b, i in routes.items()] == (
        [("fluid", 0)] if owner == 0 else [("fluid", 0), ("query_only", 1)])
    initial, query = arrays(order)
    assert bound_duration(initial) == .25 / 1249.
    assert bound_duration(query) == .25 / 9265.


@pytest.mark.parametrize("derived", (False, True))
def test_temporal_capture_refuses_without_publishing_a_bound(derived):
    _, _, program, _, candidate = authored_capture("main_current", install_bound=False)
    captured = program.dot_all(candidate, candidate) if derived else candidate
    before = copy.deepcopy(program._serialize())
    ids = program._next_id

    def builder(P, cfl):
        contraction = captured if derived else P.dot_all(captured, captured)
        return cfl / (1 + contraction)

    with pytest.raises(ValueError, match="body and captures.*not allowed"):
        program.set_dt_bound(builder)
    assert program._serialize() == before and program._next_id == ids
    assert not program.has_dt_bound()


def test_callback_cannot_hide_a_commit_outside_its_scalar_dependencies():
    frame = Rectangle("mutation_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("mutation_model", frame=frame)
    state = model.state("U", components=("a", "b"))
    case = pops.Case("mutation_case")
    fluid, side = case.block("fluid", model), case.block("side", model)
    program = pops.Program("mutation_program")
    q, passive = program.state(fluid[state]), program.state(side[state])
    next_q = program.value("identity", (q.n[0], q.n[1]), at=q.next.point)
    side_update = program.value("side_update", (2 * passive.n[0], 3 * passive.n[1]),
                                 at=passive.next.point)
    program.commit(q.next, next_q)
    program.step_strategy(FixedDt(.1))
    before = copy.deepcopy(program._serialize())

    def builder(P, cfl):
        P.commit(passive.next, side_update)
        return cfl * P.hmin()

    with pytest.raises(ValueError, match="read.only|commit|mutation|captures"):
        program.set_dt_bound(builder)
    assert program._serialize() == before and len(program.commits()) == 1


def test_legacy_serialization_hash_and_cpp_match_exact_parent_archive(tmp_path):
    archive = subprocess.check_output(("git", "archive", CANDIDATE + "^", "python"), cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as saved:
        saved.extractall(tmp_path, filter="data")
    script = '''
import json
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_dt_bound_capture_independent_runtime import authored_capture
rows = {}
for kind in ("unbounded", "legacy"):
    case, layout, program, _, _ = authored_capture("main_current", install_bound=False)
    if kind == "legacy":
        program.set_dt_bound(lambda P, cfl: cfl * P.hmin())
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rows[kind] = {"ir": program._serialize(), "semantic": program._semantic_serialize(),
                  "hash": program._ir_hash(), "cpp": emit_cpp_program(resolved.time, model_graph=graph)}
print(json.dumps(rows, sort_keys=True))
'''
    def received(package):
        environment = dict(os.environ, PYTHONPATH=str(package))
        environment.pop("POPS_NATIVE_DIM", None)
        return json.loads(subprocess.check_output((sys.executable, "-c", script),
                          cwd=ROOT, env=environment, text=True))
    assert received(tmp_path / "python") == received(ROOT / "python")
