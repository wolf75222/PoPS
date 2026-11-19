"""One canonical form per frozen authoring Program (N11, F5): bit for bit, and its witness."""
import gc
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import copy
from dataclasses import dataclass
from threading import Barrier, get_ident
from types import MappingProxyType
from typing import ClassVar
from weakref import ref

import pytest

import pops
from examples.api_v02.finite_structural_depth import build_finite_structural_depth
from examples.api_v02.profile_contracts import SELECTED_PROFILES, named_profiles
from examples.api_v02.profile_dispatch import build_profile
from pops.time._program import canonical_form
from pops.time._program.serialization import ir_identity_reuse

GALLERY = [(identifier, choice.name) for identifier in SELECTED_PROFILES
           for choice in named_profiles(identifier)]
# These projections hold a tuple (history_field_metadata_budget), so they are not closed builtin
# JSON: a decoded copy would hold a list, and the Program keeps no form.
NOT_CLOSED_JSON = {
    ("M06", "one-phase-stefan"), ("T01", "stage-constrained-ssprk2"),
    ("T12", "uniform-spatial-face"), ("T12", "field-after-replacement"),
}


def _texts(program):
    return tuple(json.dumps(program._serialize(include_provenance=variant), sort_keys=False)
                 for variant in (True, False)) + (program._ir_hash(),)


def _snapshot_texts(snapshot):
    return (snapshot._canonical_json, snapshot._artifact_canonical_json,
            snapshot._semantic_canonical_json, snapshot.hash, snapshot.artifact_hash,
            snapshot.semantic_identity.token)


def _resolved(spec, validated):
    try:
        plan = pops.resolve(validated, layout=spec.layout)
    except Exception as error:  # a profile whose resolve needs native code or a capability
        return ("refused", type(error).__name__, str(error))
    return (plan.plan_identity.token, *_snapshot_texts(plan.snapshot))


@pytest.fixture
def form_switch(monkeypatch):
    def switch(enabled):
        monkeypatch.setattr(canonical_form, "_ENABLED", enabled)
    return switch


def _validated(identifier, profile):
    spec = build_profile(identifier, profile=profile)
    validated = pops.validate(spec.case)
    return spec, validated, spec.case._time


@pytest.mark.parametrize(("identifier", "profile"), GALLERY)
def test_gallery_form_is_bit_identical_and_kept(identifier, profile, form_switch):
    spec, validated, program = _validated(identifier, profile)
    form_switch(False)
    expected = _texts(program), _resolved(spec, validated)
    assert not canonical_form.has_form(program)
    form_switch(True)
    first = _texts(program)
    # Every gallery Program reaches only immutable objects and sealed registries: a new mutable
    # leaf class fails here instead of silently disabling the form.
    if (identifier, profile) in NOT_CLOSED_JSON:
        assert type(program.history_field_metadata_budget) is tuple
        assert not canonical_form.has_form(program)
    else:
        assert canonical_form.has_form(program)
    assert (first, _resolved(spec, validated)) == expected
    assert _texts(program) == expected[0]


def test_structural_depth_form_is_bit_identical(form_switch):
    spec = build_finite_structural_depth(depth=8)
    validated = pops.validate(spec.case)
    program = spec.case._time
    form_switch(False)
    expected = _texts(program), _resolved(spec, validated)
    form_switch(True)
    assert (_texts(program), _resolved(spec, validated)) == expected
    assert canonical_form.has_form(program)


def _frozen_with_form():
    spec, _validated_case, program = _validated("M01", "periodic-advection")
    reference = _texts(program)
    assert canonical_form.has_form(program)
    return spec, program, reference


def _projected(program):
    return canonical_form.projected_ir_hash(program)


def test_reads_decode_a_fresh_dict_in_insertion_order():
    _spec, program, reference = _frozen_with_form()
    data = program._serialize(include_provenance=False)
    assert list(data) == list(program._canonical_projection(include_provenance=False))
    data["nodes"].clear()
    data["name"] = "mutated by a caller"
    assert _texts(program) == reference


def test_rebinding_a_program_attribute_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    object.__setattr__(program, "name", "renamed out of contract")
    assert program._ir_hash() == _projected(program) != reference[2]


def test_rebinding_a_node_attribute_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    node = next(value for value in program._values if value.name)
    object.__setattr__(node, "name", node.name + "-edited")
    assert program._ir_hash() == _projected(program) != reference[2]


def test_rebinding_inside_a_leaf_object_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    node = program._values[-1]
    record = node.provenance
    span = record.primary
    object.__setattr__(span, "line", span.line + 1000)
    with_provenance = json.dumps(program._serialize(include_provenance=True), sort_keys=False)
    assert with_provenance != reference[0]
    assert with_provenance == json.dumps(
        program._canonical_projection(include_provenance=True), sort_keys=False)


def test_a_mutable_object_in_the_ir_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    pops.validate(spec.case)
    program = spec.case._time
    node = program._values[-1]
    object.__setattr__(node, "attrs", MappingProxyType({**node.attrs, "extension": [1, 2]}))
    program._ir_hash()
    assert not canonical_form.has_form(program)


def test_a_mutable_program_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    program = spec.case._time
    assert program._frozen is False
    program._ir_hash()
    assert not canonical_form.has_form(program)


def test_a_frozen_program_of_an_open_case_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    program = spec.case._time
    program.freeze()
    program._ir_hash()
    # Its Case registries are still open: nothing is kept until pops.validate seals them.
    assert not canonical_form.has_form(program)
    pops.validate(spec.case)
    program._ir_hash()
    assert canonical_form.has_form(program)


def test_a_changed_definition_gives_the_new_identity(monkeypatch):
    _spec, program, reference = _frozen_with_form()
    form = canonical_form._attached_form(program)
    assert form is not None
    assert form.authorities
    authority, fingerprint = form.authorities[0]
    changed = "test:sha256:" + "0" * 64
    assert fingerprint != changed
    monkeypatch.setattr(type(authority), "fingerprint", lambda self: changed)
    assert not canonical_form._witness_holds(program, form)


def test_copy_and_rebuild_do_not_retain_the_source_form():
    program = pops.Program("collected-form").freeze()
    expected = _texts(program)
    reference = ref(program)
    form = canonical_form._attached_form(program)
    assert form is not None
    twin = copy(program)
    rebuilt = program._rebuild(lambda _value: True)
    assert not canonical_form.has_form(twin)
    assert not canonical_form.has_form(rebuilt)
    object.__setattr__(twin, "_sealed_canonical_form", form)
    assert not canonical_form.has_form(twin)
    assert not canonical_form._witness_holds(twin, form)
    assert _texts(twin) == _texts(rebuilt) == expected
    assert canonical_form._attached_form(twin) is not form
    with ir_identity_reuse("pops.resolve"):
        assert program._ir_hash() == expected[-1]
        saved_context = copy_context()
    del program, form
    gc.collect()
    assert reference() is None
    assert saved_context.run(twin._ir_hash) == expected[-1]


@pytest.mark.parametrize("include_provenance", [False, True])
def test_projections_from_different_authorities_cannot_publish(monkeypatch, include_provenance):
    from pops.model.ownership import UnresolvedOwnershipError, _AuthoringAuthority

    authority = _AuthoringAuthority()
    first = "test:sha256:" + "a" * 64
    second = "test:sha256:" + "b" * 64
    authority.bind_fingerprint(first, priority=1)
    program = pops.Program("two-variants").freeze()

    def projection(self, *, include_provenance: bool):
        if include_provenance is not first_variant:
            authority.bind_fingerprint(second, priority=1)
        return {"authority": authority.fingerprint(), "provenance": include_provenance}

    first_variant = include_provenance
    monkeypatch.setattr(pops.Program, "_canonical_projection", projection)
    with pytest.raises(UnresolvedOwnershipError, match="changed during canonical projection"):
        program._serialize(include_provenance=first_variant)
    assert not canonical_form.has_form(program)
    assert program._serialize(include_provenance=first_variant)["authority"] == second
    assert program._serialize(include_provenance=not first_variant)["authority"] == second


def test_parallel_builds_attach_one_complete_form(monkeypatch):
    program = pops.Program("parallel-forms").freeze()
    original = pops.Program._canonical_projection
    barrier = Barrier(2)

    def simultaneous(self, *, include_provenance=True):
        barrier.wait(timeout=10)
        return original(self, include_provenance=include_provenance)

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as workers:
        patch.setattr(pops.Program, "_canonical_projection", simultaneous)
        left = workers.submit(program._serialize, include_provenance=False)
        right = workers.submit(program._serialize, include_provenance=True)
        variants = left.result(), right.result()
    form = canonical_form._attached_form(program)
    assert form is not None
    assert all(variant == json.loads(form.texts[index]) for index, variant in enumerate(variants))
    assert _texts(program)[-1] == canonical_form.projected_ir_hash(program)


def test_failed_witness_removes_only_the_observed_attachment(monkeypatch):
    program = pops.Program("competing-attachment").freeze()
    program._ir_hash()
    old = canonical_form._attached_form(program)
    assert old is not None
    object.__delattr__(program, "_sealed_canonical_form")
    program._ir_hash()
    replacement = canonical_form._attached_form(program)
    assert replacement is not None and replacement is not old
    object.__setattr__(program, "_sealed_canonical_form", old)
    original = canonical_form._witness_holds

    def invalidate(value, observed):
        if observed is old:
            object.__setattr__(value, "_sealed_canonical_form", replacement)
            return False
        return original(value, observed)

    monkeypatch.setattr(canonical_form, "_witness_holds", invalidate)
    assert _texts(program)[-1] == replacement.ir_hash
    assert canonical_form._attached_form(program) is replacement


def test_exact_sealed_registry_guard_is_the_only_opaque_method():
    from pops.model.registry import OperatorRegistry

    _spec, program, _reference = _frozen_with_form()
    form = canonical_form._attached_form(program)
    assert form is not None and form.registry_guards
    registry, guard = form.registry_guards[0]
    module = guard.__self__
    assert canonical_form._sealed_registry_guard(registry, guard)
    assert not canonical_form._sealed_registry_guard(registry, module.freeze)
    sealed_class = type(registry)
    try:
        object.__setattr__(registry, "__class__", OperatorRegistry)
        assert not canonical_form._sealed_registry_guard(registry, guard)
        assert not canonical_form._witness_holds(program, form)
    finally:
        object.__setattr__(registry, "__class__", sealed_class)
    sealed_module_class = type(module)
    try:
        object.__setattr__(module, "__class__", sealed_module_class.__bases__[0])
        assert not canonical_form._sealed_registry_guard(registry, guard)
        assert not canonical_form._witness_holds(program, form)
    finally:
        object.__setattr__(module, "__class__", sealed_module_class)
    object.__setattr__(registry, "_mutation_guard", module.freeze)
    assert not canonical_form._witness_holds(program, form)
    assert program._ir_hash() == canonical_form.projected_ir_hash(program)
    assert not canonical_form.has_form(program)


def test_copied_context_checks_a_kept_form_in_its_own_thread(monkeypatch):
    program = pops.Program("thread-scoped-form").freeze()
    expected = _texts(program)[0]
    threads = []
    original = canonical_form._witness_holds

    def checked(value, form):
        threads.append(get_ident())
        return original(value, form)

    monkeypatch.setattr(canonical_form, "_witness_holds", checked)
    author_thread = get_ident()
    with ir_identity_reuse("pops.resolve"), ThreadPoolExecutor(max_workers=1) as worker:
        assert json.dumps(program._serialize()) == expected
        assert json.dumps(program._serialize()) == expected
        saved_context = copy_context()
        reader, data = worker.submit(
            saved_context.run, lambda: (get_ident(), program._serialize())
        ).result()
        assert reader != author_thread
        assert json.dumps(data) == expected
        assert threads == [author_thread, reader]
    assert json.dumps(saved_context.run(program._serialize)) == expected
    assert threads == [author_thread, reader, author_thread]


@pytest.mark.parametrize("read_from", ["proxy", "provider"])
def test_registered_strategy_reprojects_current_application_data(read_from):
    from pops.time import references
    from pops.time._step.strategy import (
        StepStrategy, register_step_strategy_type, validate_step_strategy_manifest,
    )
    from pops.time._program.serialization import _IR_IDENTITY_REUSE

    settings = {"dt": 0.01}
    observations = []

    @register_step_strategy_type
    @dataclass(frozen=True, slots=True)
    class Extension(StepStrategy):
        kind: ClassVar[str] = "test_canonical_strategy_" + read_from
        view: Mapping[str, float] | None

        def to_data(self):
            observations.append(references._REGISTRY_READS.get())
            data = settings if self.view is None else self.view
            return {"kind": self.kind, "dt": data["dt"]}

        @classmethod
        def from_data(cls, data):
            return cls(MappingProxyType({"dt": data["dt"]}))

    spec = build_profile("M01", profile="periodic-advection")
    program = spec.case._time
    strategy = Extension(MappingProxyType(settings) if read_from == "proxy" else None)
    program.step_strategy(strategy)
    validated = pops.validate(spec.case)
    assert validated.frozen and program.transaction_plan().strategy is strategy

    def assert_projection(expected):
        data = {"kind": strategy.kind, "dt": expected}
        for provenance in (False, True):
            projected = program._serialize(include_provenance=provenance)
            assert projected == program._canonical_projection(include_provenance=provenance)
            assert projected["step_transaction"]["strategy"] == data
        manifest = {"strategy": data, "controls": {}}
        assert validate_step_strategy_manifest(manifest) == manifest
        assert not canonical_form.has_form(program)
        assert program._ir_hash() == canonical_form.projected_ir_hash(program)
        return program._ir_hash()

    before = assert_projection(0.01)
    settings["dt"] = 0.02
    current = assert_projection(0.02)
    assert current != before
    with pytest.raises(RuntimeError, match="changed during pops.resolve"):
        with ir_identity_reuse("pops.resolve"):
            assert program._ir_hash() == current
            saved_context = copy_context()
            settings["dt"] = 0.03
    assert assert_projection(0.03) != current
    assert observations and all(scope is None for scope in observations)
    reference = ref(program)
    del validated, program, spec
    gc.collect()
    assert reference() is None
    expired_scope = saved_context.run(_IR_IDENTITY_REUSE.get)
    assert expired_scope is not None and not expired_scope.applies()


@pytest.mark.parametrize("fails", [False, True])
def test_registry_observation_stays_in_its_projection_and_thread(monkeypatch, fails):
    from pops.model.ownership import _FINGERPRINT_SCOPE
    from pops.time import references

    with monkeypatch.context() as patch:
        patch.setattr(canonical_form, "_ENABLED", False)
        owner, _valid_owner, program = _validated("M01", "periodic-advection")
        other, _valid_other, nested = _validated("M01", "periodic-advection")
    owned = owner.subjects["fluid"]
    foreign = other.subjects["fluid"]
    expected_foreign = references.canonical_handle(foreign).canonical_identity()
    scoped = []
    original = pops.Program._canonical_projection

    with ThreadPoolExecutor(max_workers=1) as worker:
        def capture(target, *, include_provenance=True):
            reads = references._REGISTRY_READS.get()
            if target is nested:
                assert reads is None
                return original(target, include_provenance=include_provenance)
            assert target is program and reads is not None and reads.applies()
            fingerprints = _FINGERPRINT_SCOPE.get()
            assert fingerprints is not None and fingerprints.applies()
            copied = copy_context()
            scoped.append((copied, reads, fingerprints))
            assert not reads.records
            resolved = worker.submit(copied.run, references.canonical_handle, foreign).result()
            assert resolved.canonical_identity() == expected_foreign
            assert not reads.records
            with monkeypatch.context() as patch:
                patch.setattr(canonical_form, "_ENABLED", False)
                nested._serialize()
            assert references._REGISTRY_READS.get() is reads and not reads.records
            references.canonical_handle(owned)
            assert tuple(reads.records.values()) == (owned.block_ref._instance_registry,)
            result = original(target, include_provenance=include_provenance)
            if fails:
                raise RuntimeError("projection refused in test")
            return result

        monkeypatch.setattr(pops.Program, "_canonical_projection", capture)
        if fails:
            with pytest.raises(RuntimeError, match="projection refused in test"):
                program._serialize()
        else:
            program._serialize()
        assert canonical_form.has_form(program) is not fails
        assert references._REGISTRY_READS.get() is None
        assert len(scoped) == (1 if fails else 2)
        for copied, reads, fingerprints in scoped:
            assert not reads.applies() and not reads.records
            assert not fingerprints.applies() and not fingerprints.observed
            assert copied.run(references.canonical_handle, foreign).canonical_identity() == expected_foreign
            worker.submit(copied.run, references.canonical_handle, foreign).result()
            assert not reads.records


@pytest.mark.parametrize("provider", ["stock", "schedule", "trigger"])
def test_program_schedule_observes_its_actual_projection_sources(monkeypatch, provider):
    from examples.api_v02 import models_M01_M14
    from pops.time import AcceptedStep, Always, FixedDt, Schedule

    settings = {"revision": 1}

    @dataclass(frozen=True, slots=True)
    class LiveSchedule(Schedule):
        component_uri = "test://pops/time/schedule/canonical-source"
        component_version = 1

        def schedule_payload(self):
            return {"configuration_revision": settings["revision"]}

    @dataclass(frozen=True, slots=True)
    class LiveTrigger(Always):
        component_uri = "test://pops/time/schedule/trigger/canonical-source"
        component_version = 1

        def schedule_payload(self):
            return {"configuration_revision": settings["revision"]}

    def attach_euler(case, block, subject, rate, dt, *, name):
        program = pops.Program(name)
        state = program.state(block[subject])
        domain = AcceptedStep(program.clock)
        trigger = LiveTrigger(domain) if provider == "trigger" else Always(domain)
        schedule = LiveSchedule(trigger) if provider == "schedule" else Schedule(trigger)
        slope = rate(state.n, schedule=schedule)
        endpoint = state.next
        result = program.value("euler", state.n + program.dt * slope, at=endpoint.point)
        program.commit(endpoint, result)
        program.step_strategy(FixedDt(dt))
        case.program(program)

    monkeypatch.setattr(models_M01_M14, "_traced_ssprk2", attach_euler)
    spec = models_M01_M14.build_m01()
    assert pops.validate(spec.case).frozen
    program = spec.case._time

    def check_schedule(expected):
        for variant in (False, True):
            data = program._serialize(include_provenance=variant)
            assert data == program._canonical_projection(include_provenance=variant)
            scheduled = next(node["attrs"]["schedule"] for node in data["nodes"]
                             if node["op"] == "rhs")
            payload = scheduled["trigger"]["payload"] if provider == "trigger" else scheduled["payload"]
            assert payload == ({} if provider == "stock" else {"configuration_revision": expected})
        assert canonical_form.has_form(program) is (provider == "stock")
        assert program._ir_hash() == canonical_form.projected_ir_hash(program)
        return program._ir_hash()

    before = check_schedule(1)
    settings["revision"] = 2
    after = check_schedule(2)
    assert (after == before) is (provider == "stock")


def test_mapping_witness_requires_passive_dict_storage_and_checks_entry_order():
    from collections import UserDict

    class ExternalMapping(UserDict):
        def items(self):
            raise AssertionError("cache witness must not call an application Mapping")

    assert canonical_form._mapping_witness(MappingProxyType(ExternalMapping())) is None
    source = {"first": 1, "second": 2}
    record = canonical_form._mapping_witness(MappingProxyType(source))
    assert record is not None
    assert canonical_form._mappings_hold((record,))
    source["first"] = source.pop("first")
    assert not canonical_form._mappings_hold((record,))
    record = canonical_form._mapping_witness(MappingProxyType(source))
    assert record is not None
    source["first"] = 2
    assert not canonical_form._mappings_hold((record,))


def test_mapping_witness_rechecks_a_kept_form_inside_an_operation():
    _spec, program, _reference = _frozen_with_form()
    value = program._values[-1]
    backing = {**value.attrs, "observed_metadata": (1,)}
    object.__setattr__(value, "attrs", MappingProxyType(backing))

    with ir_identity_reuse("pops.resolve"):
        before = program._serialize()
        assert canonical_form.has_form(program)
        assert before["nodes"][-1]["attrs"]["observed_metadata"] == [1]
        assert program._serialize() == before
        backing["observed_metadata"] = (2,)
        after = program._serialize()
        assert after["nodes"][-1]["attrs"]["observed_metadata"] == [2]
        assert after == program._canonical_projection()


def test_a_reuse_scope_trusts_a_checked_form_until_it_ends():
    _spec, program, reference = _frozen_with_form()
    with pytest.raises(RuntimeError, match="changed during pops.compile"):
        with ir_identity_reuse("pops.compile"):
            assert program._ir_hash() == reference[2]
            object.__setattr__(program, "name", "renamed out of contract")
            assert program._ir_hash() == reference[2]
    assert program._ir_hash() != reference[2]


def test_the_compile_scope_projects_a_kept_form_for_real(monkeypatch):
    _spec, program, reference = _frozen_with_form()
    projected = []
    original = canonical_form.projected_ir_hash

    def counted(value):
        projected.append(value)
        return original(value)

    monkeypatch.setattr(canonical_form, "projected_ir_hash", counted)
    for operation in ("pops.resolve", "pops.compile"):
        with ir_identity_reuse(operation):
            assert program._ir_hash() == reference[2]
    assert projected == [program]


def test_only_closed_finite_json_and_refusing_classes_are_kept():
    assert canonical_form._closed_json({"a": [1, 2.5, None, True, "x"]})
    assert not canonical_form._closed_json({"a": float("nan")})
    assert not canonical_form._closed_json({"a": (1,)})
    assert not canonical_form._closed_json({1: "a"})

    class Claims:
        __pops_ir_immutable__ = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    claim = Claims()
    assert not canonical_form._immutable_class(Claims, claim)
    assert not hasattr(claim, canonical_form._PROBE)

    class Refuses:
        __slots__ = ("value",)

        def __setattr__(self, name, value):
            raise AttributeError("immutable")

    assert canonical_form._immutable_class(Refuses, Refuses())
    import numpy as np
    from pops.time import CopyCurrent

    for mutable in ([], {}, set(), bytearray(), np.zeros(2), lambda: None):
        assert not canonical_form._immutable_class(type(mutable), mutable)
    assert canonical_form._immutable_class(CopyCurrent, CopyCurrent())
