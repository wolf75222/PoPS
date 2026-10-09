"""SOURCE_ONLY counter-reception of ea13; real Program and emitter bodies.

No DSO, Native simulation, compiler or capacity-provider substitute is run.
The two historical emitter functions are explicitly read from local git0abb.
"""

import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARENT = "0abbe25395a44e570d8d5525693b8e2dcbf4d387"


def old(path):
    return subprocess.check_output(
        ["rtk", "proxy", "git", "show", PARENT + ":" + path], cwd=ROOT
    ).decode()


def extract(text, name, **namespace):
    nodes = [
        x for x in ast.walk(ast.parse(text)) if isinstance(x, ast.FunctionDef) and x.name == name
    ]
    assert len(nodes) == 1
    # Actual body, defaults and imports; closure variables are real Program/indices.
    module = ast.Module(body=nodes, type_ignores=[])
    namespace.update(Any=object, json=json)
    exec(compile(module, "actual emitter body", "exec"), namespace)
    return namespace[name]


def authored(*, global_storage=True):
    path = ROOT / "tests/review/test_sol61_history_storage_owner_received.py"
    spec = importlib.util.spec_from_file_location("independent_existing_history_witness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, program, blocks, states, problem, observation, field, _ = module._baseline()
    value = observation[field[problem.unknowns[2]]]
    owner = blocks[0]
    if global_storage:
        program.store_history("exact-global-history", value, depth=1, owner_block=owner)
    else:
        # A legacy physical State history, not a fabricated global descriptor.
        current = program.state(blocks[0][states[0]])
        program.store_history("exact-state-history", current.n, depth=1)
    return case, program, value, owner


def shape(program, *, historical=False):
    path = "python/pops/codegen/program_emit_amr.py"
    text = old(path) if historical else (ROOT / path).read_text()
    return extract(text, "_emit_checkpoint_shape_metadata")(program)


def registrations(program, *, historical=False):
    path = "python/pops/codegen/program_emit_control.py"
    text = old(path) if historical else (ROOT / path).read_text()
    return extract(text, "registrations", program=program, block_idx=program._block_indices())()


def strings(cpp, symbol):
    match = re.search(r"const char\* " + symbol + r"\(int index\) \{(.*?)\n\}", cpp, re.S)
    assert match is not None
    return [json.loads(s) for s in re.findall(r'case \d+: return ("(?:\\.|[^"\\])*");', match[1])]


def declared_descriptor(program):
    # Direct observation of genuine serializer output, not descriptor() or raw CP.
    ir = program._serialize(include_provenance=False)
    stores = [n for n in ir["nodes"] if n["op"] == "store_history"]
    assert len(stores) == 1
    return json.dumps(
        stores[0]["attrs"]["global_field_storage"], sort_keys=True, separators=(",", ":")
    ), ir


def test_exact_issued_IR_identity_for_shape_and_every_hierarchy_registration():
    _, program, value, owner = authored()
    expected, ir = declared_descriptor(program)
    assert ir["version"] >= 16 and value.block is None and value.state_ref is None
    assert json.loads(expected)["ncomp"] == 1
    assert owner in program._block_indices()
    before = program._ir_hash()
    cpp = shape(program)
    assert strings(cpp, "pops_program_checkpoint_history_state_identity") == [expected]
    lines = registrations(program)
    assert len(lines) == 1 and json.dumps(expected) in lines[0]
    assert program._ir_hash() == before
    # This is exactly the old mismatch; the authoritative capacity guard is retained.
    assert strings(
        shape(program, historical=True), "pops_program_checkpoint_history_state_identity"
    ) == ["scalar-history:exact-global-history"]
    assert json.dumps(expected) not in registrations(program, historical=True)[0]


def test_legacy_physical_state_history_shape_and_phase_bytes_exact_parent():
    _, program, _, _ = authored(global_storage=False)
    before = program._ir_hash()
    assert shape(program) == shape(program, historical=True)
    assert registrations(program) == registrations(program, historical=True)
    assert program._ir_hash() == before


@pytest.mark.parametrize("boundary", ("raw", "freeze", "graph"))
def test_snapshot_boundaries_keep_same_IR_issued_descriptor(boundary):
    _, program, _, _ = authored()
    if boundary == "freeze":
        program = program.freeze()
    elif boundary == "graph":
        assert program.to_graph() is not None
    expected, _ = declared_descriptor(program)
    assert strings(shape(program), "pops_program_checkpoint_history_state_identity") == [expected]
    assert json.dumps(expected) in registrations(program)[0]


@pytest.mark.parametrize(
    "mutation", ("owner", "point", "bool_ncomp", "remove_qualification", "registered_bool")
)
@pytest.mark.parametrize("site", ("shape", "hierarchy"))
def test_resealed_storage_changes_fail_before_returning_any_emission(mutation, site):
    _, program, _, _ = authored()
    node = next(n for n in program._values if n.op == "store_history")
    attrs = dict(node.attrs)
    metadata = dict(attrs["global_field_storage"])
    if mutation == "owner":
        metadata["owner_block"] = next(
            state.block
            for state in program._time_states.values()
            if state.block is not metadata["owner_block"]
        )
    elif mutation == "point":
        metadata["point"] = program.stage("foreign-current-point", c=0)
    elif mutation == "bool_ncomp":
        metadata["ncomp"] = True
    elif mutation == "remove_qualification":
        del attrs["global_field_storage"]
    else:
        program._histories_ncomp["exact-global-history"] = True
    if mutation != "remove_qualification":
        attrs["global_field_storage"] = metadata
    object.__setattr__(node, "attrs", attrs)
    with pytest.raises((ValueError, TypeError)):
        (shape if site == "shape" else registrations)(program)


def test_generated_CPP_bytes_are_in_real_artifact_spec_cache_projection():
    import hashlib

    _, program, _, _ = authored()
    current, historical = shape(program), shape(program, historical=True)
    assert hashlib.sha256(current.encode()).digest() != hashlib.sha256(historical.encode()).digest()
    spec = ast.parse((ROOT / "python/pops/codegen/_artifact_identity.py").read_text())
    fn = next(
        n for n in spec.body if isinstance(n, ast.FunctionDef) and n.name == "program_artifact_spec"
    )
    namespace = dict(hashlib=hashlib, Any=object, Mapping=object)

    # Actual function/identity primitive. Only local toolchain/catalog inputs are
    # explicit SOURCE_ONLY values; do not query an incompatible installed DSO.
    class NoImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    fn = NoImports().visit(fn)
    from pops.identity import artifact_spec_identity
    from pops.identity.semantic import semantic_identity, program_semantic_data, model_semantic_data
    from pops.codegen.program_models import ProgramModelGraph

    calls = []

    def record(semantic, **kw):
        calls.append((semantic, kw))
        return artifact_spec_identity(semantic, **kw)

    namespace.update(
        _precision_cache_key=lambda: "binary64",
        _registry_cache_key=lambda: "SOURCE_ONLY-registry",
        _native_feature_key=lambda: "SOURCE_ONLY-features",
        program_semantic_data=program_semantic_data,
        model_semantic_data=model_semantic_data,
        semantic_identity=semantic_identity,
        artifact_spec_identity=record,
        ProgramModelGraph=ProgramModelGraph,
    )
    from types import SimpleNamespace

    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "actual artifact-spec body", "exec"),
        namespace,
    )
    kwargs = dict(
        snapshot=None,
        model_authority=None,
        program=program,
        program_graph=SimpleNamespace(graph_hash="sameGraph"),
        target="amr_system",
        abi_key="sameABI",
        compiler="sameCompiler",
        standard="c++17",
        cflags=[],
        lflags=[],
        optflags=[],
        libraries=[],
        native_components=[],
        native_dimension=2,
    )
    a = namespace["program_artifact_spec"](source=historical, **kwargs)
    c = namespace["program_artifact_spec"](source=current, **kwargs)
    assert (
        a[0] == c[0]
        and a[1] != c[1]
        and calls[0][1]["components"]["generated_source"]
        != calls[1][1]["components"]["generated_source"]
    )
    assert {k: v for k, v in calls[0][1].items() if k != "components"} == {
        k: v for k, v in calls[1][1].items() if k != "components"
    }
    driver = ast.parse((ROOT / "python/pops/codegen/_compile_drivers.py").read_text())
    call = next(
        n
        for n in ast.walk(driver)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "program_artifact_spec"
    )
    assert any(
        k.arg == "source" and isinstance(k.value, ast.Name) and k.value.id == "src"
        for k in call.keywords
    )


def test_native_capacity_guard_headers_and_source_byte_identical_parent():
    for path in (
        "src/runtime/amr/amr_system.cpp",
        "include/pops/runtime/program/amr_program_checkpoint.hpp",
        "include/pops/runtime/program/amr_program_context_history_checkpoint_public.inc",
    ):
        assert (ROOT / path).read_text() == old(path)
    body = (ROOT / "src/runtime/amr/amr_system.cpp").read_text()
    assert "live.histories != shape.histories" in body
    assert "live AMR Program checkpoint state differs from its frozen capacity metadata" in body
