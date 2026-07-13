"""ADC-563: the real freeze lifecycle -- Descriptor / Problem / Program / snapshot.

A Problem is MUTABLE while authored and FROZEN by pops.compile. After freeze, every mutating setter
RAISES (naming the frozen object), the member descriptors are sealed, and Problem.freeze() returns a
stable AuthoringSnapshot whose .hash enters the real compile cache identity. There is NO warning, NO
shallow-copy escape. Pure Python; needs only ``import pops``.
"""
import pytest
from decimal import Decimal
from fractions import Fraction

pops = pytest.importorskip("pops", exc_type=ImportError)
from pops.params import ConstParam  # noqa: E402

from pops.problem._snapshot import AuthoringSnapshot, build_problem_snapshot  # noqa: E402
from pops.numerics.riemann import HLL  # noqa: E402
from pops.descriptors import BrickDescriptor, Descriptor  # noqa: E402
from pops.ir import Const  # noqa: E402
from pops.math import ddt, div  # noqa: E402
from pops.model import (  # noqa: E402
    Module,
    Operator,
    OperatorRegistry,
    Rate,
)
from pops.physics import Model as PhysicsModel  # noqa: E402


def _model_and_state(name="isothermal", *, speed=1):
    """Return a small owner-qualified model and its public state declaration."""
    model = PhysicsModel(name)
    state = model.state("U", components=["u"])
    flux = model.flux(
        "F", on=state, x=[speed * state[0]], y=[state[0]],
        waves={"x": [speed], "y": [1]},
    )
    model.rate("A", ddt(state) == -div(flux))
    return model, state


def _model(name="isothermal", *, speed=1):
    return _model_and_state(name, speed=speed)[0]


def _state_handle(model, name="U"):
    """Read the registry-issued state handle without reaching into model internals."""
    return next(
        handle for handle in model.declaration_index().records()
        if handle.kind == "state" and handle.local_id == name
    )


def _problem(name="plasma"):
    return pops.Problem(name=name).block("ne", physics=_model(), spatial=pops.FiniteVolume())


def _operator_module(name="deep-operator-module"):
    module = Module(name)
    state = module.state_space("U", ("rho",))
    capabilities = {"routes": {"backends": ["cpu"]}}
    requirements = {"aux": ["B_z"]}
    lowering = {"sources": ["default"], "stages": {"order": [1, 2]}}
    body = {"x": [Const(0.0)], "y": [Const(0.0)]}
    module.operator(
        "flux",
        signature=(state,) >> Rate(state),
        kind="grid_operator",
        capabilities=capabilities,
        requirements=requirements,
        lowering=lowering,
        expr=body,
    )
    return module, state, capabilities, requirements, lowering, body


# ---------------------------------------------------------------------------
# AuthoringSnapshot: stable hash, mutation-sensitive, JSON-ready.
# ---------------------------------------------------------------------------

def test_snapshot_hash_is_a_stable_sha256():
    snap = build_problem_snapshot(_problem())
    assert isinstance(snap, AuthoringSnapshot)
    assert isinstance(snap.hash, str) and len(snap.hash) == 64
    assert all(c in "0123456789abcdef" for c in snap.hash)


def test_snapshot_hash_stable_across_identical_assemblies():
    assert build_problem_snapshot(_problem()).hash == build_problem_snapshot(_problem()).hash


def test_snapshot_hash_changes_on_a_different_assembly():
    p2 = _problem()
    p2.param(ConstParam("gamma", 1.4))
    assert build_problem_snapshot(_problem()).hash != build_problem_snapshot(p2).hash


def test_snapshot_is_json_ready():
    import json
    d = build_problem_snapshot(_problem()).to_dict()
    assert d["schema_version"] == 7
    assert json.loads(json.dumps(d, sort_keys=True)) == d  # no runtime object, no numpy array


def test_snapshot_to_dict_has_no_mutable_escape_from_cached_identity():
    snap = AuthoringSnapshot({"blocks": [{"name": "a"}]})
    before = snap.hash
    external = snap.to_dict()
    external["blocks"][0]["name"] = "mutated"

    assert snap.to_dict()["blocks"][0]["name"] == "a"
    assert snap.hash == before
    with pytest.raises(AttributeError, match="immutable"):
        snap._hash = "changed"


def test_snapshot_preserves_exact_parameter_literals():
    p = _problem()
    p.param(ConstParam("third", Fraction(1, 3)))
    payload = build_problem_snapshot(p).to_dict()

    assert payload["params"]["third"]["default"] == {
        "state": "value",
        "value": {
            "kind": "rational",
            "numerator": "1",
            "denominator": "3",
            "target": "Real",
        }
    }


def test_structural_objects_with_different_state_never_collide():
    class C:
        def __init__(self, value):
            self.value = value

    one = AuthoringSnapshot({"descriptor": C(1)})
    two = AuthoringSnapshot({"descriptor": C(2)})

    assert one.hash != two.hash
    assert one.to_dict()["descriptor"]["$object"]["projections"]["fields"]["value"] != \
        two.to_dict()["descriptor"]["$object"]["projections"]["fields"]["value"]


def test_private_slotted_state_is_structural_not_a_type_only_token():
    class C:
        __slots__ = ("__value",)

        def __init__(self, value):
            self.__value = value

    assert AuthoringSnapshot({"value": C(1)}).hash != AuthoringSnapshot({"value": C(2)}).hash


def test_problem_cache_identity_does_not_collapse_same_named_models():
    first = pops.Problem(name="first").block(
        "u", physics=_model("same-display-name", speed=1))
    second = pops.Problem(name="first").block(
        "u", physics=_model("same-display-name", speed=2))

    assert build_problem_snapshot(first).hash != build_problem_snapshot(second).hash


def test_problem_cache_identity_covers_composed_model_formulas():
    def scalar_problem(speed):
        model = _model("same-model-name", speed=speed)
        return pops.Problem(name="same-problem-name").block("u", physics=model)

    first = build_problem_snapshot(scalar_problem(1))
    same = build_problem_snapshot(scalar_problem(1))
    different = build_problem_snapshot(scalar_problem(2))

    assert first.hash == same.hash
    assert first.hash != different.hash


def test_structural_projection_is_deeply_detached_from_descriptor_state():
    class Descriptor:
        def __init__(self):
            self.settings = {"levels": [1, 2]}

        def options(self):
            return self.settings

    descriptor = Descriptor()
    snapshot = AuthoringSnapshot({"descriptor": descriptor})
    descriptor.settings["levels"].append(3)

    captured = snapshot.to_dict()["descriptor"]["$object"]["projections"]
    assert len(captured["options"]["levels"]) == 2
    assert len(captured["fields"]["settings"]["levels"]) == 2


def test_mapping_valued_descriptor_options_participate_in_hash():
    first = BrickDescriptor("scheme", "native", options={"order": 1})
    second = BrickDescriptor("scheme", "native", options={"order": 2})

    assert AuthoringSnapshot({"brick": first}).hash != AuthoringSnapshot({"brick": second}).hash


def test_snapshot_envelope_schema_version_cannot_be_shadowed_by_payload():
    with pytest.raises(ValueError, match="reserved key 'schema_version'"):
        AuthoringSnapshot({"schema_version": 999})


@pytest.mark.parametrize("accessor", ["options", "to_dict"])
def test_failing_structural_projection_is_never_swallowed(accessor):
    class Broken:
        pass

    def fail(_self):
        raise RuntimeError("projection exploded")

    setattr(Broken, accessor, fail)
    with pytest.raises(RuntimeError, match="projection exploded") as exc:
        AuthoringSnapshot({"broken": Broken()})
    assert any("AuthoringSnapshot" in note and "%s()" % accessor in note
               for note in getattr(exc.value, "__notes__", ()))


def test_non_finite_decimal_is_refused_instead_of_entering_json_hash():
    with pytest.raises(ValueError, match="non-finite Decimal"):
        AuthoringSnapshot({"bad": Decimal("NaN")})


# ---------------------------------------------------------------------------
# Problem.freeze(): idempotent, mutation-after-freeze RAISES.
# ---------------------------------------------------------------------------

def test_problem_freeze_returns_snapshot_and_is_idempotent():
    p = _problem()
    snap = p.freeze()
    assert p.frozen and p.snapshot is snap
    assert p.freeze() is snap  # idempotent: the same snapshot


def test_failed_freeze_is_atomic_and_problem_can_be_edited_then_retried():
    class Freezable(Descriptor):
        def __init__(self, name):
            self.label = name
            self.frozen = False

        @property
        def name(self):
            return self.label

        def options(self):
            return {"name": self.label}

        def freeze(self):
            self.frozen = True

        def _pops_freeze_snapshot(self, capability):
            from pops.problem._freeze_transaction import _require_freeze_capability
            _require_freeze_capability(capability)
            return (self.label, self.frozen)

        def _pops_freeze_restore(self, capability, state):
            from pops.problem._freeze_transaction import _require_freeze_capability
            _require_freeze_capability(capability)
            self.label, self.frozen = state

    class RepairableOptions:
        ready = False

        def options(self):
            if not self.ready:
                raise RuntimeError("not serializable yet")
            return {"value": 2}

    layout = pops.mesh.layouts.Uniform(pops.mesh.CartesianMesh(n=8))
    spatial = Freezable("spatial")
    repairable = RepairableOptions()
    p = (pops.Problem(layout=layout, name="atomic")
         .block("ne", physics=_model(), spatial=spatial)
         .aux("repairable", repairable))

    with pytest.raises(RuntimeError, match="not serializable yet"):
        p.freeze()

    assert not p.frozen
    assert p.snapshot is None
    assert not getattr(layout, "_frozen", False)
    assert not spatial.frozen
    # Neither the Problem facade, its registries nor its member descriptors/layout were sealed.
    p.block("ion", physics=_model("isothermal-ion"))
    spatial.label = "repaired-spatial"
    layout.mesh.n = 16
    repairable.ready = True
    snapshot = p.freeze()
    assert p.frozen
    assert getattr(layout, "_frozen", False)
    assert spatial.frozen
    assert snapshot is p.snapshot
    assert "ion" in p.blocks()


def test_freeze_rolls_back_every_prior_member_when_a_later_member_raises():
    class FailingBrick(Descriptor):
        def __init__(self, name, *, fail=False):
            self.label = name
            self.scheme = "test"
            self.fail = fail

        @property
        def name(self):
            return self.label

        def options(self):
            return {"scheme": self.scheme, "fail": self.fail}

        def freeze(self):
            super().freeze()
            if self.fail:
                raise RuntimeError("second member failed")
            return self

    first = FailingBrick("first")
    second = FailingBrick("second", fail=True)
    program = pops.time.Program("rollback-program")
    model = _model("rollback-u")
    p = pops.Problem(name="rollback")
    block = p.block("u", model, spatial=first, time=program, diagnostics=[second])

    with pytest.raises(RuntimeError, match="second member failed"):
        p.freeze()

    assert not p.frozen and p.snapshot is None
    assert not getattr(first, "_frozen", False)
    assert not getattr(second, "_frozen", False)
    assert not program._frozen
    assert not p._block_registry._frozen
    p.block("v", physics=_model("rollback-v"))  # registry editable after exact rollback
    first.scheme = "edited-after-rollback"
    program.state(block, _state_handle(model))
    second.fail = False
    snapshot = p.freeze()
    assert snapshot is p.snapshot
    assert getattr(first, "_frozen", False)
    assert getattr(second, "_frozen", False)
    assert program._frozen


def test_opaque_irreversible_freezer_is_refused_before_any_member_mutates():
    class Opaque:
        def __init__(self):
            self.frozen = False

        def freeze(self):
            self.frozen = True

    first = pops.FiniteVolume()
    opaque = Opaque()
    p = pops.Problem(name="preflight")
    p.block("u", _model(), spatial=first, diagnostics=[opaque])

    with pytest.raises(TypeError, match="opaque non-transactional"):
        p.freeze()

    assert not getattr(first, "_frozen", False)
    assert not opaque.frozen
    assert not p.frozen and p.snapshot is None
    p.block("still-editable", physics=_model("still-editable"))


def test_deep_freeze_covers_block_members_and_global_program_without_stale_snapshot():
    model = Module("model")
    state = model.state_space("U", ("u",))
    spatial = pops.FiniteVolume()
    diagnostic = Descriptor()
    block_time = pops.time.Program("block-time")
    global_time = pops.time.Program("global-time")
    p = pops.Problem(name="deep-freeze")
    block = p.block(
        "u", model, spatial=spatial, time=block_time, diagnostics=[diagnostic])
    p.time(global_time)

    snapshot = p.freeze()

    for descriptor in (spatial, diagnostic):
        with pytest.raises(RuntimeError, match="frozen"):
            descriptor.changed = True
    with pytest.raises(RuntimeError, match="frozen"):
        model.state_space("late", ("u",))
    for program in (block_time, global_time):
        with pytest.raises(RuntimeError, match="frozen"):
            program.state(block, model.state_handle(state))
    assert build_problem_snapshot(p).hash == snapshot.hash


def test_python_physics_model_is_deeply_frozen_without_changing_snapshot_identity():
    model, state = _model_and_state("frozen-scalar")
    p = pops.Problem(name="physics-freeze").block("u", physics=model)

    snapshot = p.freeze()

    assert model.frozen and model.dsl.frozen and model.dsl._m.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        model.scalar("twice_u", 2 * state[0])
    with pytest.raises(RuntimeError, match="frozen"):
        model.dsl._m.gamma = 1.2
    assert build_problem_snapshot(p).hash == snapshot.hash


def test_raw_module_freeze_seals_registry_and_operator_records_deeply():
    module, state, *_ = _operator_module("raw-module-freeze")
    registry = module.operator_registry()
    operator = registry.get("flux")
    problem = pops.Problem(name="raw-module-freeze").block("fluid", physics=module)

    snapshot = problem.freeze()

    assert module.frozen and registry.frozen and operator.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(Operator(
            "late", "grid_operator", (state,) >> Rate(state), body={"x": (), "y": ()}))
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register_alias("late_alias", "flux")
    with pytest.raises(RuntimeError, match="frozen"):
        operator.capabilities = {}
    with pytest.raises(RuntimeError, match="frozen"):
        module._frozen = False
    with pytest.raises(RuntimeError, match="frozen"):
        registry._frozen = False
    with pytest.raises(RuntimeError, match="frozen"):
        operator._frozen = False
    with pytest.raises(TypeError):
        operator.capabilities["late"] = True
    with pytest.raises(AttributeError):
        operator.capabilities["routes"]["backends"].append("gpu")
    assert build_problem_snapshot(problem).hash == snapshot.hash


def test_raw_module_freeze_detaches_all_stale_operator_metadata_aliases():
    module, _, capabilities, requirements, lowering, body = _operator_module(
        "raw-module-aliases")
    operator = module.operator_registry().get("flux")
    stale_capabilities = operator.capabilities
    stale_requirements = operator.requirements
    stale_lowering = operator.lowering
    problem = pops.Problem(name="raw-module-aliases").block("fluid", physics=module)

    snapshot = problem.freeze()
    frozen_hash = module.module_hash()
    stale_capabilities["late"] = True
    stale_requirements["aux"].append("late_aux")
    stale_lowering["sources"].append("late_source")
    capabilities["routes"]["backends"].append("gpu")
    requirements["aux"].append("foreign_aux")
    lowering["stages"]["order"].append(3)
    body["x"].append(Const(1.0))

    assert dict(operator.capabilities) == {"routes": {"backends": ("cpu",)}}
    assert dict(operator.requirements) == {"aux": ("B_z",)}
    assert operator.lowering["sources"] == ("default",)
    assert operator.lowering["stages"]["order"] == (1, 2)
    assert len(operator.body["x"]) == 1
    assert module.module_hash() == frozen_hash
    assert build_problem_snapshot(problem).hash == snapshot.hash


def test_module_eigenvalues_never_expose_a_live_authoring_alias():
    module = Module("eigenvalue-aliases")
    module.state_space("U", ("rho",))
    x_input, y_input = [Const(1.0)], [Const(-1.0)]
    returned = module.eigenvalues(x_input, y_input)
    initial_hash = module.module_hash()

    returned["x"].append(Const(2.0))
    x_input.append(Const(3.0))
    assert len(module._eigenvalues["x"]) == 1
    assert module.module_hash() == initial_hash

    stale_internal = module._eigenvalues
    problem = pops.Problem(name="eigenvalue-aliases").block("fluid", physics=module)
    snapshot = problem.freeze()
    frozen_hash = module.module_hash()
    stale_internal["x"] = stale_internal["x"] + (Const(4.0),)

    assert len(module._eigenvalues["x"]) == 1
    assert module.module_hash() == frozen_hash
    assert build_problem_snapshot(problem).hash == snapshot.hash
    with pytest.raises(TypeError):
        module._eigenvalues["x"] = ()


def test_failed_problem_freeze_restores_raw_module_registry_and_operator_mutability():
    class FailingSpatial(Descriptor):
        def options(self):
            return {"kind": "failing"}

        def freeze(self):
            super().freeze()
            raise RuntimeError("later spatial freeze failed")

    module, state, *_ = _operator_module("raw-module-rollback")
    registry = module.operator_registry()
    operator = registry.get("flux")
    problem = pops.Problem(name="raw-module-rollback").block(
        "fluid", physics=module, spatial=FailingSpatial())

    with pytest.raises(RuntimeError, match="later spatial freeze failed"):
        problem.freeze()

    assert not module.frozen and not registry.frozen and not operator.frozen
    operator.capabilities["after_rollback"] = True
    registry.register(Operator(
        "late", "grid_operator", (state,) >> Rate(state), body={"x": (), "y": ()}))
    module.eigenvalues([Const(0.0)], [Const(0.0)])
    assert module.list_operators() == ["flux", "late"]


def test_adopted_registry_is_bound_to_one_module_mutation_lifecycle():
    module = Module("adopted-registry")
    state = module.state_space("U", ("rho",))
    registry = OperatorRegistry(owner=module.owner_path)
    registry.register(Operator(
        "rate", "local_rate", (state,) >> Rate(state), body=(Const(0.0),)))

    module.adopt_registry(registry)
    module.freeze()

    assert registry.frozen and registry.get("rate").frozen
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register_alias("default", "rate")


def test_later_freeze_failure_restores_python_physics_cascade_exactly():
    class FailingSpatial(Descriptor):
        def options(self):
            return {"kind": "failing"}

        def freeze(self):
            super().freeze()
            raise RuntimeError("spatial freeze failed")

    model, state = _model_and_state("rollback-scalar")
    p = pops.Problem(name="physics-rollback").block(
        "u", physics=model, spatial=FailingSpatial())
    before = build_problem_snapshot(p).hash

    with pytest.raises(RuntimeError, match="spatial freeze failed"):
        p.freeze()

    assert not model.frozen and not model.dsl.frozen and not model.dsl._m.frozen
    assert build_problem_snapshot(p).hash == before
    # Complex child/container state was restored, not merely the facade bit.
    model.scalar("twice_u", 2 * state[0])


@pytest.mark.parametrize("mutate", [
    lambda p: p.block("extra", physics=_model()),
    lambda p: p.block("extra2", _model()),
    lambda p: p.param(ConstParam("gamma", 1.4)),
    lambda p: p.aux("B_z"),
    lambda p: p.output(pops.output.OutputPolicy()),
    lambda p: p.time(pops.time.Program("t")),
])
def test_every_mutating_setter_raises_after_freeze(mutate):
    p = _problem()
    p.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        mutate(p)


def test_freeze_error_names_the_problem_and_recompile():
    p = pops.Problem(name="plasma").block("ne", physics=_model())
    p.freeze()
    with pytest.raises(RuntimeError) as exc:
        p.block("x", physics=_model())
    msg = str(exc.value)
    assert "plasma" in msg and "pops.compile" in msg and "recompile" in msg


def test_mutation_after_freeze_does_not_change_the_snapshot_hash():
    # No shallow-copy escape: the snapshot was captured deep + inert at freeze; a blocked mutation
    # cannot alter it, and the hash is stable.
    p = _problem()
    h = p.freeze().hash
    with pytest.raises(RuntimeError):
        p.block("late", physics=_model())
    assert p.snapshot.hash == h


# ---------------------------------------------------------------------------
# Descriptor / BrickDescriptor freeze.
# ---------------------------------------------------------------------------

def test_descriptor_freeze_raises_on_mutation():
    h = HLL()  # a BrickDescriptor (riemann)
    h.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        h.scheme = "other"


def test_brick_descriptor_freeze_raises():
    b = BrickDescriptor("x", "native", native_id="pops::X")
    b.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        b.native_id = "pops::Y"


def test_fluent_builder_still_mutates_before_freeze():
    # A descriptor is mutable while authored (no freeze on validate); a fluent builder is unaffected.
    from pops.mesh.amr import Refine
    from pops.model import Handle, OwnerPath
    rho = Handle("rho", kind="state", owner=OwnerPath.shared("freeze-test"))
    r = Refine.on(rho).above(0.05)  # mutates during build
    assert r.validate() is True


def test_problem_freeze_seals_member_descriptors():
    # Freezing the Problem cascades freeze to the typed member descriptors it holds (a field
    # problem's typed solver). The block's runtime spatial brick is not a typed Descriptor, so the
    # cascade seals what it can: the field registry's FieldProblem descriptors.
    from pops.math import unknown, laplacian
    from pops.ir import ValueExpr
    from pops.fields import PoissonProblem
    model = _model("field-freeze")
    state = _state_handle(model)
    field = PoissonProblem(unknown=unknown("phi"),
                           equation=(-laplacian(unknown("phi")) == ValueExpr(state)),
                           inputs=(state,))
    p = pops.Problem(name="plasma").block("ne", physics=model).field(field)
    p.freeze()
    # The FieldProblem descriptor is sealed: a post-freeze attribute mutation raises.
    with pytest.raises(RuntimeError, match="frozen"):
        field.solver = "changed"


# ---------------------------------------------------------------------------
# Program freeze (via compile) + snapshot authentication.
# ---------------------------------------------------------------------------

def test_program_freeze_raises_on_new_node():
    model = _model("program-freeze")
    problem = pops.Problem(name="program-freeze")
    block = problem.block("ne", model)
    state = _state_handle(model)
    prog = pops.time.Program("t")
    prog.state(block, state)
    prog.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        prog.state(block, state)


def test_snapshot_validator_authenticates_type_hash_and_payload():
    from pops.problem._snapshot import validate_problem_snapshot

    snapshot = AuthoringSnapshot({"problem": "cache-identity"})
    assert validate_problem_snapshot(snapshot) == snapshot.hash
    with pytest.raises(TypeError, match="AuthoringSnapshot"):
        validate_problem_snapshot(snapshot.hash)

    object.__setattr__(snapshot, "_hash", "a" * 64)
    with pytest.raises(ValueError, match="canonical payload"):
        validate_problem_snapshot(snapshot)

    artifact_snapshot = AuthoringSnapshot({"problem": "artifact-identity"})
    object.__setattr__(artifact_snapshot, "_artifact_hash", "b" * 64)
    with pytest.raises(ValueError, match="canonical artifact projection"):
        validate_problem_snapshot(artifact_snapshot)


def test_compiled_handle_is_sealed_after_public_compile():
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.model.bind_schema import BindSchema

    handle = CompiledProblem("x.so", None, None, "abi", "c++", "c++23")
    handle._advanced_attach = "ok"  # the advanced compile_problem route stays attachable
    handle._seal()
    with pytest.raises(AttributeError, match="immutable after pops.compile"):
        handle.so_path = "y.so"
    with pytest.raises(AttributeError, match="ADC-563"):
        handle.install_plan = object()

    block = CompiledModel(
        "block.so", "production", "add_native_block", (), (), (), 0,
        None, 0, {}, {}, "abi", "model-hash", "c++", "c++23",
    )
    block.bind_schema = BindSchema()  # advanced Model.compile remains attachable with typed metadata
    block._seal()
    with pytest.raises(AttributeError, match="immutable after pops.compile"):
        block.bind_schema = None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
