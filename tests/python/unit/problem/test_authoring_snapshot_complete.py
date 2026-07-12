"""ADC-655: the authoring snapshot captures the complete effective compile transaction."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from collections import OrderedDict

import pytest

pops = pytest.importorskip("pops")

import pops.problem._snapshot as snapshot_module  # noqa: E402
import pops.lib.time as libtime  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import Uniform  # noqa: E402
from pops.model import Handle, Module, OwnerKind, OwnerPath  # noqa: E402
from pops.problem._snapshot import AuthoringSnapshot, build_authoring_snapshot  # noqa: E402
from pops.problem._detached import detached_frozen  # noqa: E402


_GLOBAL_CALLABLE_DATA = {"scale": 2, "offsets": [1, 3]}
_CLASS_METHOD_DATA = {"bias": 4}
_OPAQUE_CALLABLE_GLOBAL = object()


def _operator_using_mutable_global(value):
    return _GLOBAL_CALLABLE_DATA["scale"] * value + _GLOBAL_CALLABLE_DATA["offsets"][0]


def _operator_using_opaque_global():
    return _OPAQUE_CALLABLE_GLOBAL


def _problem():
    module = Module("transport")
    module.state_space("U", ("u",))
    return pops.Problem(name="complete-snapshot").block("fluid", physics=module)


def _program(problem, scheme):
    spec = problem._blocks.spec("fluid")
    module = spec["model"]
    block = problem.blocks()["fluid"]
    state = module.state_handle(module.state_spaces()["U"])
    if scheme == "euler":
        return libtime.forward_euler(block, state)
    if scheme == "ssprk2":
        return libtime.SSPRK2(block, state)
    raise ValueError("unknown test scheme %r" % scheme)


def test_authoring_snapshot_is_the_only_snapshot_type_name():
    assert pops.AuthoringSnapshot is AuthoringSnapshot
    assert not hasattr(snapshot_module, "ProblemSnapshot")


def test_effective_layout_and_time_have_distinct_complete_identities():
    problem = _problem()
    base = build_authoring_snapshot(
        problem,
        layout=Uniform(CartesianMesh(n=16, L=1.0)),
        time=_program(problem, "euler"),
    )
    other_layout = build_authoring_snapshot(
        problem,
        layout=Uniform(CartesianMesh(n=32, L=1.0)),
        time=_program(problem, "euler"),
    )
    other_time = build_authoring_snapshot(
        problem,
        layout=Uniform(CartesianMesh(n=16, L=1.0)),
        time=_program(problem, "ssprk2"),
    )

    assert base.hash != other_layout.hash
    assert base.hash != other_time.hash
    assert base.artifact_hash != other_layout.artifact_hash
    assert base.artifact_hash != other_time.artifact_hash
    assert base.to_dict()["compile_context"]["layout"] \
        != other_layout.to_dict()["compile_context"]["layout"]
    assert base.to_dict()["compile_context"]["time"] \
        != other_time.to_dict()["compile_context"]["time"]


_SUBPROCESS_PROBE = textwrap.dedent(
    """
    import json
    import pops
    import pops.lib.time as libtime
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.model import Module
    from pops.problem._snapshot import build_authoring_snapshot

    model = Module("transport")
    state = model.state_space("U", ("u",))
    problem = pops.Problem(name="deterministic")
    block = problem.add_block("fluid", model)
    time = libtime.SSPRK2(block, model.state_handle(state))
    snapshot = build_authoring_snapshot(
        problem, layout=Uniform(CartesianMesh(n=16, L=1.0)), time=time)
    print(json.dumps({
        "hash": snapshot.hash,
        "artifact_hash": snapshot.artifact_hash,
        "payload": snapshot.to_dict(),
    }, allow_nan=False, sort_keys=True, separators=(",", ":")))
    """
)


def _probe(hash_seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    return subprocess.check_output(
        [sys.executable, "-c", _SUBPROCESS_PROBE],
        env=env,
        text=True,
    ).strip()


def test_complete_snapshot_is_deterministic_across_fresh_processes():
    assert _probe(1) == _probe(99991)


def test_bound_callable_instance_state_participates_in_both_identities():
    class Scale:
        def __init__(self, factor):
            self.factor = factor

        def apply(self, value):
            return self.factor * value

    first = AuthoringSnapshot({"operator": Scale(2).apply})
    second = AuthoringSnapshot({"operator": Scale(3).apply})

    assert first.hash != second.hash
    assert first.artifact_hash != second.artifact_hash


def test_referenced_mutable_global_participates_in_callable_identity():
    original = {"scale": 2, "offsets": [1, 3]}
    try:
        first = AuthoringSnapshot({"operator": _operator_using_mutable_global})
        _GLOBAL_CALLABLE_DATA["scale"] = 5
        _GLOBAL_CALLABLE_DATA["offsets"].append(8)
        second = AuthoringSnapshot({"operator": _operator_using_mutable_global})
    finally:
        _GLOBAL_CALLABLE_DATA.clear()
        _GLOBAL_CALLABLE_DATA.update(original)

    assert first.hash != second.hash
    assert first.artifact_hash != second.artifact_hash


def test_callable_container_kinds_and_mapping_order_have_distinct_identities():
    def operator(default):
        def apply(value=default):
            return value

        return apply

    ordered_a = {"left": 1, "right": 2}
    ordered_b = {"right": 2, "left": 1}

    assert AuthoringSnapshot({"fn": operator([1])}).hash \
        != AuthoringSnapshot({"fn": operator((1,))}).hash
    assert AuthoringSnapshot({"fn": operator({1})}).hash \
        != AuthoringSnapshot({"fn": operator(frozenset({1}))}).hash
    assert AuthoringSnapshot({"fn": operator(ordered_a)}).hash \
        != AuthoringSnapshot({"fn": operator(ordered_b)}).hash
    assert AuthoringSnapshot({"fn": operator(ordered_a)}).hash \
        != AuthoringSnapshot({"fn": operator(OrderedDict(ordered_a))}).hash


def test_distinct_mapping_keys_with_one_canonical_identity_are_rejected():
    class IdentityKey:
        def __init__(self, name):
            self.name = name

    first = IdentityKey("same")
    second = IdentityKey("same")

    with pytest.raises(ValueError, match="distinct keys with the same canonical identity"):
        AuthoringSnapshot({"mapping": {first: "first", second: "second"}})


def test_empty_closure_cell_cannot_collide_with_user_mapping():
    def factory(empty):
        value = {"empty_cell": True}

        def apply():
            return value

        if empty:
            del value
        return apply

    empty = factory(True)
    populated = factory(False)

    assert empty.__code__ is populated.__code__
    assert AuthoringSnapshot({"fn": empty}).hash != AuthoringSnapshot({"fn": populated}).hash


def test_exception_table_participates_in_callable_identity():
    def guarded(value):
        try:
            return 1 / value
        except ZeroDivisionError:
            return 0

    if not hasattr(guarded.__code__, "co_exceptiontable"):
        pytest.skip("code exception tables are unavailable on this Python")
    without_handlers = types.FunctionType(
        guarded.__code__.replace(co_exceptiontable=b""), guarded.__globals__)

    assert guarded.__code__.co_code == without_handlers.__code__.co_code
    assert AuthoringSnapshot({"fn": guarded}).hash \
        != AuthoringSnapshot({"fn": without_handlers}).hash


def test_inherited_callable_implementation_participates_in_identity():
    class BasePolicy:
        def __call__(self, value):
            return value + 1

    class Policy(BasePolicy):
        pass

    original = BasePolicy.__call__
    try:
        first = AuthoringSnapshot({"policy": Policy()})

        def changed(self, value):
            return value + 2

        BasePolicy.__call__ = changed
        second = AuthoringSnapshot({"policy": Policy()})
    finally:
        BasePolicy.__call__ = original

    assert first.hash != second.hash
    assert first.artifact_hash != second.artifact_hash


def test_module_used_as_opaque_callable_value_is_rejected():
    def operator():
        return os

    with pytest.raises(TypeError, match="module is used as a value"):
        AuthoringSnapshot({"operator": operator})


def test_explicit_module_attribute_dependency_is_structurally_accepted():
    def operator(value):
        return os.path.basename(value)

    snapshot = AuthoringSnapshot({"operator": operator})

    assert len(snapshot.hash) == 64


def test_opaque_referenced_global_is_refused_instead_of_omitted():
    with pytest.raises(TypeError, match=r"cannot encode opaque builtins\.object"):
        AuthoringSnapshot({"operator": _operator_using_opaque_global})


def _constant_operator(source):
    namespace = {"__name__": "pops_snapshot_constant_probe"}
    exec("def operator(value):\n    return %s\n" % source, namespace)
    return namespace["operator"]


def test_frozenset_and_complex_code_constants_have_content_identity():
    first_set = _constant_operator("value in {1, 2}")
    second_set = _constant_operator("value in {3, 4}")
    first_complex = _constant_operator("value * (1 + 2j)")
    second_complex = _constant_operator("value * (1 + 3j)")

    assert first_set.__code__.co_code == second_set.__code__.co_code
    assert first_complex.__code__.co_code == second_complex.__code__.co_code
    set_a = AuthoringSnapshot({"operator": first_set})
    set_b = AuthoringSnapshot({"operator": second_set})
    complex_a = AuthoringSnapshot({"operator": first_complex})
    complex_b = AuthoringSnapshot({"operator": second_complex})

    assert set_a.hash != set_b.hash
    assert set_a.artifact_hash != set_b.artifact_hash
    assert complex_a.hash != complex_b.hash
    assert complex_a.artifact_hash != complex_b.artifact_hash


def test_class_method_defaults_kwdefaults_and_globals_have_content_identity():
    class Policy:
        def apply(self, value, scale=2, *, offset=1):
            return scale * value + offset + _CLASS_METHOD_DATA["bias"]

    original = dict(_CLASS_METHOD_DATA)
    try:
        baseline = AuthoringSnapshot({"policy": Policy()})
        Policy.apply.__defaults__ = (5,)
        Policy.apply.__kwdefaults__ = {"offset": 7}
        changed_defaults = AuthoringSnapshot({"policy": Policy()})
        _CLASS_METHOD_DATA["bias"] = 11
        changed_global = AuthoringSnapshot({"policy": Policy()})
    finally:
        _CLASS_METHOD_DATA.clear()
        _CLASS_METHOD_DATA.update(original)

    assert baseline.hash != changed_defaults.hash
    assert baseline.artifact_hash != changed_defaults.artifact_hash
    assert changed_defaults.hash != changed_global.hash
    assert changed_defaults.artifact_hash != changed_global.artifact_hash


def test_structural_projection_is_sampled_once_for_full_and_artifact_views():
    class StatefulProjection:
        def __init__(self):
            self.calls = 0

        def to_data(self):
            self.calls += 1
            return {"sample": self.calls}

    value = StatefulProjection()
    snapshot = AuthoringSnapshot({"value": value})

    assert value.calls == 1
    assert snapshot.to_dict()["value"]["$object"]["projections"]["to_data"]["sample"] \
        == {"$scalar": {"kind": "integer", "value": "1"}}
    artifact_value = snapshot.artifact_to_dict()["payload"]["value"]
    assert artifact_value["$object"]["projections"]["to_data"]["sample"] \
        == {"$scalar": {"kind": "integer", "value": "1"}}


def test_library_location_is_provenance_but_not_artifact_identity():
    from pops.codegen.library import LibraryManifest, _content_hash

    content_hash = _content_hash("transport", "production", "abi", ())
    first_manifest = LibraryManifest(
        "transport", "production", "abi", (), (), content_hash,
        so_path="/tmp/first/transport.so")
    second_manifest = LibraryManifest(
        "transport", "production", "abi", (), (), content_hash,
        so_path="/opt/other/transport.so")
    first = AuthoringSnapshot({"library": first_manifest})
    second = AuthoringSnapshot({"library": second_manifest})

    assert first.hash != second.hash
    assert first.artifact_hash == second.artifact_hash


def test_stale_authoring_mutations_cannot_change_an_existing_snapshot():
    problem = _problem()
    stale_block_spec = problem._blocks.spec("fluid")
    layout = Uniform(CartesianMesh(n=16, L=1.0, periodic=True))
    time = _program(problem, "ssprk2")
    snapshot = build_authoring_snapshot(problem, layout=layout, time=time)
    before = snapshot.to_dict()
    before_hash = snapshot.hash

    layout.mesh.n = 4096
    time.linear_combine("late-presentation-node", time._values[0])
    stale_block_spec["spatial"] = {"scheme": "mutated-after-snapshot"}
    caller_copy = snapshot.to_dict()
    caller_copy["compile_context"]["libraries"] = ["mutated-copy"]

    assert snapshot.hash == before_hash
    assert snapshot.to_dict() == before


def test_plain_mutable_extension_record_cannot_cross_compiled_boundary():
    class PlainRecord:
        def __init__(self):
            self.options = {"order": [1, 2]}

    with pytest.raises(TypeError, match="must implement freeze"):
        detached_frozen(PlainRecord())


def test_noop_extension_freeze_hook_is_rejected():
    class LyingRecord:
        def __init__(self):
            self.options = {"order": [1, 2]}

        def freeze(self):
            return self

    with pytest.raises(TypeError, match="did not make retained state immutable"):
        detached_frozen(LyingRecord())


def test_unresolved_handle_cannot_cross_compiled_boundary():
    owner = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "unresolved")
    handle = Handle("u", kind="state", owner=owner)

    with pytest.raises(ValueError, match="Problem.resolve"):
        detached_frozen({"state": handle})


@pytest.mark.parametrize("kind", ["mapping", "sequence"])
def test_cyclic_authoring_container_is_rejected_without_exposing_mutable_shell(kind):
    if kind == "mapping":
        value = {}
        value["self"] = value
    else:
        value = []
        value.append(value)

    with pytest.raises(ValueError, match="cyclic authoring value"):
        detached_frozen(value)
