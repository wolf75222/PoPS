"""SOURCE_ONLY reception a02: no Native/JIT/DSO substitution."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARENT = "378f29084c915ae316308481949ea174858197b2"


def helper():
    source = ROOT / "tests/review/test_sol61_history_storage_owner_received.py"
    spec = importlib.util.spec_from_file_location("independent_original_storage", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unused(name="unused-owner"):
    data = helper()._storage_only(5)
    case, program, blocks, states, problem, value, field, point, owner, issued = data
    before = tuple(program._values)
    old_order = legacy_order(program)
    solve_inputs = tuple(value.inputs[0].inputs)
    program.store_history(name, value, depth=3, owner_block=owner)
    assert tuple(program._values[:-1]) == before
    assert tuple(value.inputs[0].inputs) == solve_inputs
    assert owner not in old_order
    assert not any(node.op == "state" and node.block is owner for node in program._values)
    assert not any(node.op == "commit" and node.block is owner for node in program._values)
    return data


def legacy_order(program):
    source = subprocess.check_output(
        ["rtk", "proxy", "git", "show", PARENT + ":python/pops/time/_program/serialization.py"],
        cwd=ROOT,
    ).decode()
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_block_indices"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "actual parent block indices", "exec"),
        namespace,
    )
    return namespace["_block_indices"](program)


def test_unused_TimeState_owner_is_appended_without_physical_promotion():
    _, program, _, _, _, value, _, point, owner, issued = unused()
    old = legacy_order(program)
    before = program._serialize()
    order = program._block_indices()
    assert list(order.items())[:-1] == list(old.items())
    assert order[owner] == len(old) == 3
    assert len(issued.space.components) == 5
    assert program._histories_ncomp["unused-owner"] == 1
    assert value.block is value.state_ref is value.space is None and value.point == point
    assert program._serialize() == before
    from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
    from pops.time._program.global_history_storage import descriptor

    cpp = _emit_checkpoint_shape_metadata(program)
    assert json.dumps(descriptor(program, "unused-owner")) in cpp


@pytest.mark.parametrize("boundary", ("freeze", "graph"))
def test_detached_snapshots_preserve_exact_storage_route_and_no_State_node(boundary):
    _, program, *_ = unused()
    expected = [(block.name, index) for block, index in program._block_indices().items()]
    if boundary == "freeze":
        program = program.freeze()
    else:
        program.to_graph()
    assert [(block.name, index) for block, index in program._block_indices().items()] == expected
    owner = next(block for block in program._block_indices() if block.name == "storage-only")
    assert not any(node.op == "state" and node.block is owner for node in program._values)


@pytest.mark.parametrize(
    "mutation",
    ("owner", "bool_width", "clock", "qualification", "lost_store", "clone", "lost_scope"),
)
def test_storage_route_authenticates_before_returning_indices(mutation):
    _, program, blocks, _, _, _, _, _, owner, _ = unused()
    node = next(node for node in program._values if node.op == "store_history")
    attrs = dict(node.attrs)
    if mutation == "owner":
        program._history_blocks["unused-owner"] = blocks[0]
    elif mutation == "bool_width":
        program._histories_ncomp["unused-owner"] = True
    elif mutation == "clock":
        from pops.time.points import Clock

        metadata = dict(attrs["global_field_storage"])
        metadata["clock"] = Clock("foreign", owner=program.owner_path)
        attrs["global_field_storage"] = metadata
        object.__setattr__(node, "attrs", attrs)
    elif mutation == "clone":
        from pops.problem.handles import BlockHandle

        cloned = BlockHandle(
            owner.local_id, owner=owner.owner_path, model_owner=owner.model_owner_path
        )
        metadata = dict(attrs["global_field_storage"])
        metadata["owner_block"] = cloned
        attrs["global_field_storage"] = metadata
        program._history_blocks["unused-owner"] = cloned
        object.__setattr__(node, "attrs", attrs)
        object.__setattr__(node, "block", cloned)
    elif mutation == "lost_scope":
        program._time_states = {
            key: state for key, state in program._time_states.items() if state.block is not owner
        }
    elif mutation == "qualification":
        del attrs["global_field_storage"]
        object.__setattr__(node, "attrs", attrs)
    else:
        program._values[:] = [value for value in program._values if value is not node]
    with pytest.raises((ValueError, TypeError)):
        program._block_indices()
    assert not any(value.op == "state" and value.block is owner for value in program._values)


def resolved(layout_kind):
    import pops
    from pops.analytic import x
    from pops.initial import InitialCondition
    from pops.layouts import AMR, Uniform
    from pops.lib.initial import Analytic
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, every

    case, program, blocks, states, _, _, _, _, owner, issued = unused()
    extra = issued.state.declaration_ref or issued.state
    frame = blocks[0]._instance_registry._blocks[blocks[0].local_id]["model"].frame
    all_blocks, all_states = (*blocks, owner), (*states, extra)
    for index, (block, state) in enumerate(zip(blocks, states, strict=True)):
        # Real existing physical update only. Never read/commit the storage owner.
        current = program.state(block[state])
        program.commit(
            current.next, program.value("physical-%d" % index, 1 * current.n, at=current.next.point)
        )
    for block, state in zip(all_blocks, all_states, strict=True):
        case.initials.add(
            InitialCondition(
                state=block[state],
                value=Analytic(
                    frame=frame,
                    components=tuple(
                        0 * x(frame)
                        for _ in range(
                            len(issued.space.components) if block is owner else len(tuple(state))
                        )
                    ),
                ),
                projection=ConservativeCellAverage(),
            )
        )
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(3, 5), periodic=PeriodicAxes(frame.axes))
    if layout_kind == "system":
        layout = Uniform(grid)
    else:
        from pops.amr import (
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
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        from pops.params import RuntimeParam

        transfer = AMRTransfer()
        for block, state in zip(all_blocks, all_states, strict=True):
            transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("storage-tag-threshold", default=0.5))
        layout = AMR(
            grid=grid,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(Tag(ValueExpr(owner[extra])[0] > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
    return pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_public_Case_resolve_and_real_CPP_emission_contains_exact_four_routes(target):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    plan = resolved(target)
    assert [block.name for block in plan.blocks] == [
        "thermal",
        "matter",
        "spectator",
        "storage-only",
    ]
    assert len(plan.initial_condition_plan.bindings) == 4
    program = plan.time
    before = program._serialize()
    assert before["version"] == 16
    assert [(block.name, index) for block, index in program._block_indices().items()] == [
        ("thermal", 0),
        ("matter", 1),
        ("spectator", 2),
        ("storage-only", 3),
    ]
    cpp = emit_cpp_program(
        program, model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target=target
    )
    assert 'case 3: return "storage-only";' in cpp
    assert '"storage_state_witness' in cpp
    assert not any(
        node.op == "state" and node.block.name == "storage-only" for node in program._values
    )
    assert program._serialize() == before


def test_multiple_ring_names_do_not_duplicate_or_reorder_existing_routes():
    _, program, _, _, _, value, _, _, owner, _ = unused("z-ring")
    before = program._block_indices()
    program.store_history("a-ring", value, depth=1, owner_block=owner)
    assert program._block_indices() == before
    assert sum(block is owner for block in program._block_indices()) == 1


def legacy_images(nonlinear, endpoint):
    import hashlib
    import pops
    from pops.analytic import x
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import Analytic
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    case, program, blocks, states, problem, observation, field, _ = helper()._baseline(
        nonlinear=nonlinear, endpoint=endpoint
    )
    program.store_history("legacy-global", observation[field[problem.unknowns[1]]], depth=1)
    for index, (block, state) in enumerate(zip(blocks, states, strict=True)):
        current = program.state(block[state])
        program.commit(
            current.next,
            program.value("legacy-physical-%d" % index, 1 * current.n, at=current.next.point),
        )
        frame = block._instance_registry._blocks[block.local_id]["model"].frame
        case.initials.add(
            InitialCondition(
                state=block[state],
                value=Analytic(frame=frame, components=tuple(0 * x(frame) for _ in state)),
                projection=ConservativeCellAverage(),
            )
        )
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    plan = pops.resolve(
        pops.validate(case),
        layout=Uniform(CartesianGrid(frame=frame, cells=(3, 5), periodic=PeriodicAxes(frame.axes))),
    )
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    cpp = emit_cpp_program(plan.time, model=graph, target="system")
    return {
        "IR": plan.time._serialize(),
        "hash": plan.time._ir_hash(),
        "cpp": cpp,
        "cpp_sha256": hashlib.sha256(cpp.encode()).hexdigest(),
        "modules": [module.module_hash() for module in graph.source_modules_by_owner.values()],
    }


@pytest.fixture(scope="module")
def fresh_legacy_pairs(tmp_path_factory):
    import io
    import os
    import sys
    import tarfile

    path = tmp_path_factory.mktemp("exact-parent-python")
    archive = subprocess.check_output(
        ["rtk", "proxy", "git", "archive", PARENT, "python"], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        assert all(
            (item.name == "python" or item.name.startswith("python/"))
            and ".." not in Path(item.name).parts
            for item in files.getmembers()
        )
        files.extractall(path)
    command = [
        sys.executable,
        "-c",
        "import json,runpy; n=runpy.run_path(" + repr(str(Path(__file__).resolve())) + "); "
        "print(json.dumps([n['legacy_images'](a,b) for a in (False,True) for b in (False,True)],"
        "sort_keys=True,separators=(',',':')))",
    ]
    rows = []
    for source in (path / "python", ROOT / "python"):
        environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
        rows.append(json.loads(subprocess.check_output(command, env=environment, cwd=ROOT)))
    return rows


@pytest.mark.parametrize("profile", range(4))
def test_fresh_parent_candidate_same_callsite_full_legacy_bytes(fresh_legacy_pairs, profile):
    parent, candidate = fresh_legacy_pairs
    assert candidate[profile] == parent[profile]
