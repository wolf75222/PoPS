"""Exercise public resolve/compile across the real detached Program boundary.

Only toolchain execution is substituted; semantic resolution and emission are real.
"""
from pathlib import Path
import hashlib
import json
import sys
from types import ModuleType

import pops
import pytest

from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback
from tests.python.unit.codegen.test_module_lowering import _stub_toolchain


class _EmissionReached(Exception):
    """Stop at the actual problem compilation seam, without running a compiler."""


@pytest.mark.parametrize("physical_global", (False, True))
def test_public_compile_preserves_global_authority(monkeypatch, tmp_path, physical_global):
    drivers = _stub_toolchain(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "pops._bootstrap", ModuleType("pops._bootstrap"))
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "pops_header_signature", lambda inc: "TESTSIG")
    monkeypatch.setattr(toolchain, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(toolchain, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx=None: "c++")
    emitted = []
    def no_compile(command, what):
        output = Path(command[command.index("-o") + 1])
        source = next(arg for arg in command if arg.endswith(".cpp"))
        if source.endswith("problem.cpp"):
            emitted.append(Path(source).read_text())
            raise _EmissionReached
        output.write_bytes(b"SOURCE-ONLY-TEST-DSO")
        if "-MF" in command:
            Path(command[command.index("-MF") + 1]).write_text(f"{output}: {source}\n")
    monkeypatch.setattr(toolchain, "_run_compile", no_compile)
    monkeypatch.setattr(drivers, "_run_compile", no_compile)
    case, layout, _, _, _ = build_feedback(
        physical_global=physical_global, periodic=False, initial_condition=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(_EmissionReached):
        pops.compile(plan)
    assert len(emitted) == 1
    source = emitted[0]
    assert source.count("ctx.integral_candidate_value(") == 1
    assert ("physical_global_" in source) == physical_global
    assert source.count("ctx.capture_integral_candidate(") == 1
    assert source.count("ctx.consume_external_trace(") == 1


def _prepared():
    case, layout, program, _, _ = build_feedback(physical_global=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    return plan, program


@pytest.mark.parametrize("mutation", ("clone", "units", "index", "version", "point", "scope", "input"))
def test_resolved_reseal_cannot_replace_live_authority(mutation):
    from pops.identity import make_identity
    plan, program = _prepared()
    source = next(value for value in program._values if value.op == "source")
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    capture = source.inputs[-1]
    if mutation == "clone":
        port = rows[0]["port"]
        rows[0]["port"] = port._with_owner(port.owner_path)
    elif mutation == "units":
        rows[0]["units"] += " "
    elif mutation == "index":
        rows[0]["input"] = 0
    elif mutation == "version":
        rows[0]["version"] = True
    elif mutation == "point":
        object.__setattr__(capture, "point", next(iter(program._time_states.values())).next.point)
    elif mutation == "scope":
        object.__setattr__(capture, "attrs", {**capture.attrs, "scope": "accepted"})
    else:
        object.__setattr__(source, "inputs", (*source.inputs[:-1], source.inputs[0]))
    object.__setattr__(source, "attrs", {**source.attrs, "physical_global_inputs_v1": rows})
    object.__setattr__(plan, "plan_identity", make_identity("resolved-plan", plan._payload()))
    with pytest.raises((ValueError, TypeError)):
        plan.verify()


def test_detached_clone_and_unissued_detached_program_refuse():
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.global_source_plan import prepare_source_globals, require_detached_source_global
    plan, program = _prepared()
    detached = detach_compiled_program(program, physical_global_sources=plan._physical_global_sources)
    source = next(value for value in detached._values if value.op == "source")
    require_detached_source_global(source)
    port = source.attrs["physical_global_inputs_v1"][0]["port"]
    assert port.block_ref._instance_registry is None
    assert program._ir_hash() == detached._ir_hash()
    proof = plan._physical_global_sources
    assert not hasattr(proof, "__dict__")
    assert all(isinstance(item, (str, int)) for row in proof.signature[1] for item in row)
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    rows[0]["port"] = port._with_owner(port.owner_path)
    object.__setattr__(source, "attrs", {**source.attrs, "physical_global_inputs_v1": rows})
    assert program._ir_hash() == detached._ir_hash()
    with pytest.raises(ValueError, match="bound detached port"):
        require_detached_source_global(source)
    # Setting the detached marker alone cannot mint a proof from equal metadata.
    plain = detach_compiled_program(program)
    object.__delattr__(plain, "_physical_global_source_bindings")
    with pytest.raises(ValueError, match="no prepared authority"):
        prepare_source_globals(plain)


def test_redetachment_preserves_bound_authority_and_ir():
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.global_source_plan import require_detached_source_global
    _, program = _prepared()
    first = detach_compiled_program(program)
    second = detach_compiled_program(first)
    assert program._ir_hash() == first._ir_hash() == second._ir_hash()
    require_detached_source_global(next(value for value in second._values if value.op == "source"))


def test_source_body_reseal_cannot_refresh_prepared_authority():
    from pops.identity import make_identity
    plan, _ = _prepared()
    module = plan.blocks[0].model.module
    operator = module.operator_registry().get("reaction")
    object.__setattr__(operator, "body", tuple(.5 * expression for expression in operator.body))
    # The body is independently pinned by the live-issued proof, even if the
    # caller recomputes every public plan identity it is allowed to observe.
    with pytest.raises((ValueError, TypeError)):
        object.__setattr__(plan, "plan_identity", make_identity("resolved-plan", plan._payload()))
        plan.verify()


def test_foreign_module_cannot_supply_detached_body():
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.global_source_plan import require_detached_source_global
    plan, program = _prepared()
    detached = detach_compiled_program(program)
    value = next(value for value in detached._values if value.op == "source")
    require_detached_source_global(value, module=plan.blocks[0].model.module)
    foreign = build_feedback(physical_global=True, gamma=.31)[0]._block_registry.spec("fluid")["model"].module
    with pytest.raises(ValueError, match="Module/body authority changed"):
        require_detached_source_global(value, module=foreign)


def test_lowered_expr_body_cannot_change_without_its_registered_module():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    plan, program = _prepared()
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    model = graph.models_by_block["fluid"]
    module_hash = model.module.module_hash()
    terms = model._m._source_terms
    object.__setattr__(model._m, "_source_terms", {
        **terms, "reaction": tuple(.5 * expression for expression in terms["reaction"])})
    assert model.module.module_hash() == module_hash
    with pytest.raises(ValueError, match="lowered body changed"):
        emit_cpp_program(detach_compiled_program(program), model_graph=graph)


def test_detached_global_proof_does_not_retain_case_or_source_program():
    import gc
    import weakref
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.global_source_plan import require_detached_source_global
    def build():
        case, layout, program, _, _ = build_feedback(physical_global=True)
        plan = pops.resolve(pops.validate(case), layout=layout)
        refs = (weakref.ref(case), weakref.ref(program))
        return detach_compiled_program(program, physical_global_sources=plan._physical_global_sources), refs
    detached, refs = build()
    gc.collect()
    assert all(ref() is None for ref in refs)
    require_detached_source_global(next(value for value in detached._values if value.op == "source"))


@pytest.mark.parametrize("physical_global", (False, True))
@pytest.mark.parametrize("periodic", (False, True))
def test_detached_emission_preserves_exact_parent_ir_manifest_and_cpp(physical_global, periodic):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    parent = Path(__file__).resolve().parents[4] / "docs/development/api_040/sol61_global_authority_parent_parity.json"
    expected = json.loads(parent.read_text())[str((physical_global, periodic))]
    case, layout, program, _, _ = build_feedback(physical_global=physical_global, periodic=periodic)
    manifest = case._block_registry.spec("fluid")["model"].module.manifest().to_dict()
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(plan.time, physical_global_sources=plan._physical_global_sources)
    assert program._ir_hash() == detached._ir_hash() == expected["ir"]
    assert plan.plan_identity.token == expected["plan"]
    assert manifest["schema_version"] == expected["manifest_version"]
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == expected["manifest"]
    for target in ("system", "amr_system"):
        cpp = emit_cpp_program(detached, model_graph=graph, target=target)
        assert hashlib.sha256(cpp.encode()).hexdigest() == expected[target]
