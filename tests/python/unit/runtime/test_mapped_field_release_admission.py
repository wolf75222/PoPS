"""Source admission and real provider emission; no Native artifact is loaded or built."""
from types import SimpleNamespace

import pytest

from pops._generated_release_contract import NATIVE_ABI_VERSION
from pops.runtime import _mapped_field_capability as admission


def facts(abi=NATIVE_ABI_VERSION, **changes):
    return dict(abi_version=abi, mapped_consumed_field_output=True,
                mapped_consumed_field_output_amr=True, **changes)


@pytest.mark.parametrize("abi,adaptive", [(9, False), (10, False), (10, True),
                                        (NATIVE_ABI_VERSION, False), (NATIVE_ABI_VERSION, True)])
def test_known_endpoint_abis_through_published_release(abi, adaptive):
    calls = []
    admission.require_mapped_field_native_facts(
        adaptive=adaptive, capability_reader=lambda target: calls.append(target) or facts(abi))
    assert calls == ["production"]


@pytest.mark.parametrize("abi", [None, True, False, 11.0, "11", -1, 0, 8,
                               NATIVE_ABI_VERSION + 1, 999])
@pytest.mark.parametrize("adaptive", [False, True])
def test_unreleased_or_untyped_abi_refused(abi, adaptive):
    with pytest.raises(RuntimeError, match="mapped-consumed-output@1"):
        admission.require_mapped_field_native_facts(
            adaptive=adaptive, capability_reader=lambda _: facts(abi))


@pytest.mark.parametrize("adaptive,key", [(False, "mapped_consumed_field_output"),
                                        (True, "mapped_consumed_field_output"),
                                        (True, "mapped_consumed_field_output_amr")])
@pytest.mark.parametrize("value", [None, False, 0, 1, "true"])
def test_each_endpoint_needs_its_exact_true_capability(adaptive, key, value):
    value_facts = facts()
    value_facts[key] = value
    with pytest.raises(RuntimeError, match=key):
        admission.require_mapped_field_native_facts(
            adaptive=adaptive, capability_reader=lambda _: value_facts)


@pytest.mark.parametrize("value", [None, {}, {"abi_version": NATIVE_ABI_VERSION}])
def test_missing_native_facts_refused(value):
    with pytest.raises(RuntimeError):
        admission.require_mapped_field_native_facts(capability_reader=lambda _: value)


def test_abi9_cannot_claim_the_amr_endpoint():
    with pytest.raises(RuntimeError, match="AMR scalar endpoint@1"):
        admission.require_mapped_field_native_facts(
            adaptive=True, capability_reader=lambda _: facts(9))


def test_native_newer_than_python_release_is_stale(monkeypatch):
    monkeypatch.setattr(admission, "NATIVE_ABI_VERSION", NATIVE_ABI_VERSION - 1)
    with pytest.raises(RuntimeError):
        admission.require_mapped_field_native_facts(capability_reader=lambda _: facts())


def test_exact_artifact_target_still_controls_amr_admission():
    serialized = {"nodes": [{"op": "field_map_pack", "attrs": {"contract": "mapped-consumed-output@1"}}]}
    program = SimpleNamespace(program=SimpleNamespace(_serialize=lambda: serialized))
    artifact = SimpleNamespace(layout_programs=(SimpleNamespace(program=program, target="amr_system"),))
    value = facts()
    value["mapped_consumed_field_output_amr"] = False
    with pytest.raises(RuntimeError, match="mapped_consumed_field_output_amr"):
        admission.require_mapped_consumed_field_output(artifact, capability_reader=lambda _: value)
    artifact.layout_programs[0].target = "foreign"
    with pytest.raises(ValueError, match="exact native target"):
        admission.require_mapped_consumed_field_output(artifact, capability_reader=lambda _: facts())


class SourceEmissionComplete(Exception):
    """Stop before any SDK authentication, C++ build, provider installation or binding."""


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
def test_compiler_dispatch_reaches_actual_provider_source_only_after_admission(tmp_path, monkeypatch,
                                                                            adaptive, allowed):
    from tests.review.test_sol61_mapped_field_route import route
    from pops.codegen import _compile_drivers as compiler
    from pops.codegen import program_graph_lowering as lowering
    from pops.codegen import toolchain
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops import _capabilities_report, _native_selector

    resolved = route(tmp_path, adaptive=adaptive)
    assignments = {row.subject.local_id: row.layout for row in resolved.layout_plan.assignments
                   if row.subject_kind == "block"}
    for layout in resolved.layout_plan.layouts:
        blocks = tuple(block for block in resolved.blocks if assignments[block.name] == layout.handle)
        program = detach_compiled_program(slice_program(resolved.time, tuple(block.name for block in blocks)))
        if admission.requires_mapped_consumed_field_output(program._serialize()):
            break
    else:
        pytest.fail("real resolved route has no mapped Program")
    calls = []
    value = facts()
    value["mapped_consumed_field_output_amr" if adaptive else "mapped_consumed_field_output"] = allowed
    monkeypatch.setattr(_capabilities_report, "_module_capabilities",
                        lambda target: calls.append(target) or value)
    # Isolate the Native boundary, rather than creating a mock _pops module/artifact.
    monkeypatch.setattr(_native_selector, "select_native_dimension", lambda dimension: None)
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    emitted = []
    actual_emitter = lowering.emit_program_graph

    def emit(*args, **kwargs):
        source = actual_emitter(*args, **kwargs)
        emitted.append(source)
        assert "mapped_field_" in source
        assert "#include <pops/runtime/dynamic/abi_key.hpp>" in source
        assert "pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }" in source
        mapping_sources = tuple(tmp_path.rglob("physical_map.cpp"))
        assert mapping_sources
        assert all("POPS_ABI_KEY_LITERAL" in path.read_text() for path in mapping_sources)
        raise SourceEmissionComplete

    monkeypatch.setattr(lowering, "emit_program_graph", emit)
    expected = SourceEmissionComplete if allowed else RuntimeError
    with pytest.raises(expected):
        compiler._compile_problem_impl(
            model_graph=ProgramModelGraph.from_resolved_blocks(blocks), time=program,
            target="amr_system" if adaptive else "system", native_dimension=2,
            field_plans={}, shared_interface_codegen_evidence=None)
    assert calls == ["production"]
    assert bool(emitted) is allowed
