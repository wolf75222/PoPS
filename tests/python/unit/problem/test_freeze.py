"""ADC-563: the real freeze lifecycle -- Descriptor / Problem / Program / snapshot.

A Problem is MUTABLE while authored and FROZEN by pops.compile. After freeze, every mutating setter
RAISES (naming the frozen object), the member descriptors are sealed, and Problem.freeze() returns a
stable ProblemSnapshot whose .hash the compile cache key folds in. There is NO warning, NO
shallow-copy escape. Pure Python; needs only ``import pops``.
"""
import pytest
from decimal import Decimal
from fractions import Fraction

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.problem._snapshot import ProblemSnapshot, build_problem_snapshot  # noqa: E402
from pops.numerics.riemann import HLL  # noqa: E402
from pops.descriptors import BrickDescriptor  # noqa: E402


def _model():
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5), transport=pops.IsothermalFlux(),
                      source=pops.PotentialForce(charge=1.0), elliptic=pops.ChargeDensity(charge=1.0))


def _problem(name="plasma"):
    return pops.Problem(name=name).block("ne", physics=_model(), spatial=pops.FiniteVolume())


# ---------------------------------------------------------------------------
# ProblemSnapshot: stable hash, mutation-sensitive, JSON-ready.
# ---------------------------------------------------------------------------

def test_snapshot_hash_is_a_stable_sha256():
    snap = build_problem_snapshot(_problem())
    assert isinstance(snap, ProblemSnapshot)
    assert isinstance(snap.hash, str) and len(snap.hash) == 64
    assert all(c in "0123456789abcdef" for c in snap.hash)


def test_snapshot_hash_stable_across_identical_assemblies():
    assert build_problem_snapshot(_problem()).hash == build_problem_snapshot(_problem()).hash


def test_snapshot_hash_changes_on_a_different_assembly():
    p2 = _problem().param(pops.physics.ConstParam("gamma", 1.4))
    assert build_problem_snapshot(_problem()).hash != build_problem_snapshot(p2).hash


def test_snapshot_is_json_ready():
    import json
    d = build_problem_snapshot(_problem()).to_dict()
    assert d["schema_version"] == 3
    assert json.loads(json.dumps(d, sort_keys=True)) == d  # no runtime object, no numpy array


def test_snapshot_to_dict_has_no_mutable_escape_from_cached_identity():
    snap = ProblemSnapshot({"blocks": [{"name": "a"}]})
    before = snap.hash
    external = snap.to_dict()
    external["blocks"][0]["name"] = "mutated"

    assert snap.to_dict()["blocks"][0]["name"] == "a"
    assert snap.hash == before
    with pytest.raises(AttributeError, match="immutable"):
        snap._hash = "changed"


def test_snapshot_preserves_exact_parameter_literals():
    p = _problem().param(pops.physics.ConstParam("third", Fraction(1, 3)))
    payload = build_problem_snapshot(p).to_dict()

    assert payload["params"]["third"]["default"] == {
        "$scalar": {
            "kind": "rational",
            "numerator": "1",
            "denominator": "3",
        }
    }


def test_structural_objects_with_different_state_never_collide():
    class C:
        def __init__(self, value):
            self.value = value

    one = ProblemSnapshot({"descriptor": C(1)})
    two = ProblemSnapshot({"descriptor": C(2)})

    assert one.hash != two.hash
    assert one.to_dict()["descriptor"]["$object"]["projections"]["fields"]["value"] != \
        two.to_dict()["descriptor"]["$object"]["projections"]["fields"]["value"]


def test_private_slotted_state_is_structural_not_a_type_only_token():
    class C:
        __slots__ = ("__value",)

        def __init__(self, value):
            self.__value = value

    assert ProblemSnapshot({"value": C(1)}).hash != ProblemSnapshot({"value": C(2)}).hash


def test_problem_cache_identity_does_not_collapse_same_named_models():
    class C:
        name = "same-display-name"

        def __init__(self, coefficient):
            self.coefficient = coefficient

    first = pops.Problem(name="first").block("u", physics=C(1))
    second = pops.Problem(name="first").block("u", physics=C(2))

    assert build_problem_snapshot(first).hash != build_problem_snapshot(second).hash


def test_problem_cache_identity_covers_composed_model_formulas():
    from pops.physics.facade import Model

    def scalar_problem(speed):
        model = Model("same-model-name")
        (u,) = model.conservative_vars("u")
        model.flux(x=[speed * u], y=[u])
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
    snapshot = ProblemSnapshot({"descriptor": descriptor})
    descriptor.settings["levels"].append(3)

    captured = snapshot.to_dict()["descriptor"]["$object"]["projections"]
    assert len(captured["options"]["levels"]) == 2
    assert len(captured["fields"]["settings"]["levels"]) == 2


def test_mapping_valued_descriptor_options_participate_in_hash():
    first = BrickDescriptor("scheme", "native", options={"order": 1})
    second = BrickDescriptor("scheme", "native", options={"order": 2})

    assert ProblemSnapshot({"brick": first}).hash != ProblemSnapshot({"brick": second}).hash


def test_snapshot_envelope_schema_version_cannot_be_shadowed_by_payload():
    with pytest.raises(ValueError, match="reserved key 'schema_version'"):
        ProblemSnapshot({"schema_version": 999})


@pytest.mark.parametrize("accessor", ["options", "to_dict"])
def test_failing_structural_projection_is_never_swallowed(accessor):
    class Broken:
        pass

    def fail(_self):
        raise RuntimeError("projection exploded")

    setattr(Broken, accessor, fail)
    with pytest.raises(RuntimeError, match="projection exploded") as exc:
        ProblemSnapshot({"broken": Broken()})
    assert any("ProblemSnapshot" in note and "%s()" % accessor in note
               for note in getattr(exc.value, "__notes__", ()))


def test_non_finite_decimal_is_refused_instead_of_entering_json_hash():
    with pytest.raises(ValueError, match="non-finite Decimal"):
        ProblemSnapshot({"bad": Decimal("NaN")})


# ---------------------------------------------------------------------------
# Problem.freeze(): idempotent, mutation-after-freeze RAISES.
# ---------------------------------------------------------------------------

def test_problem_freeze_returns_snapshot_and_is_idempotent():
    p = _problem()
    snap = p.freeze()
    assert p.frozen and p.snapshot is snap
    assert p.freeze() is snap  # idempotent: the same snapshot


def test_failed_freeze_is_atomic_and_problem_can_be_edited_then_retried():
    class Freezable:
        def __init__(self, name):
            self.name = name
            self.frozen = False

        def options(self):
            return {"name": self.name}

        def freeze(self):
            self.frozen = True

        def _pops_freeze_snapshot(self, capability):
            from pops.problem._freeze_transaction import _require_freeze_capability
            _require_freeze_capability(capability)
            return (self.name, self.frozen)

        def _pops_freeze_restore(self, capability, state):
            from pops.problem._freeze_transaction import _require_freeze_capability
            _require_freeze_capability(capability)
            self.name, self.frozen = state

    class BrokenOptions:
        def options(self):
            raise RuntimeError("not serializable yet")

    layout = Freezable("layout")
    spatial = Freezable("spatial")
    p = (pops.Problem(layout=layout, name="atomic")
         .block("ne", physics=_model(), spatial=spatial)
         .param("bad", BrokenOptions()))

    with pytest.raises(RuntimeError, match="not serializable yet"):
        p.freeze()

    assert not p.frozen
    assert p.snapshot is None
    assert not layout.frozen
    assert not spatial.frozen
    # Neither the Problem facade, its registries nor its member descriptors/layout were sealed.
    p.block("ion", physics=_model())
    spatial.name = "repaired-spatial"
    layout.name = "repaired-layout"
    p.param("bad", 2)
    snapshot = p.freeze()
    assert p.frozen
    assert layout.frozen
    assert spatial.frozen
    assert snapshot is p.snapshot
    assert "ion" in p.blocks()


def test_freeze_rolls_back_every_prior_member_when_a_later_member_raises():
    class FailingBrick(BrickDescriptor):
        def __init__(self, name, *, fail=False):
            super().__init__(name, "native")
            self.fail = fail

        def freeze(self):
            super().freeze()
            if self.fail:
                raise RuntimeError("second member failed")
            return self

    first = FailingBrick("first")
    second = FailingBrick("second", fail=True)
    program = pops.time.Program("rollback-program")
    p = pops.Problem(name="rollback")
    p.add_block("u", _model(), spatial=first, time=program, diagnostics=[second])

    with pytest.raises(RuntimeError, match="second member failed"):
        p.freeze()

    assert not p.frozen and p.snapshot is None
    assert not getattr(first, "_frozen", False)
    assert not getattr(second, "_frozen", False)
    assert not program._frozen
    assert not p._block_registry._frozen
    p.block("v", physics=_model())  # the registry is editable after exact rollback
    first.scheme = "edited-after-rollback"
    program.state("editable-after-rollback")
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

    first = BrickDescriptor("first", "native")
    opaque = Opaque()
    p = pops.Problem(name="preflight")
    p.add_block("u", _model(), spatial=first, time=opaque)

    with pytest.raises(TypeError, match="opaque non-transactional"):
        p.freeze()

    assert not getattr(first, "_frozen", False)
    assert not opaque.frozen
    assert not p.frozen and p.snapshot is None
    p.block("still-editable", physics=_model())


def test_deep_freeze_covers_block_members_and_global_program_without_stale_snapshot():
    model = BrickDescriptor("model", "native")
    spatial = BrickDescriptor("spatial", "native")
    diagnostic = BrickDescriptor("diagnostic", "native")
    block_time = pops.time.Program("block-time")
    global_time = pops.time.Program("global-time")
    p = pops.Problem(name="deep-freeze")
    p.add_block("u", model, spatial=spatial, time=block_time,
                diagnostics=[diagnostic])
    p.time(global_time)

    snapshot = p.freeze()

    for descriptor in (model, spatial, diagnostic):
        with pytest.raises(RuntimeError, match="frozen"):
            descriptor.scheme = "mutated"
    for program in (block_time, global_time):
        with pytest.raises(RuntimeError, match="frozen"):
            program.state("late")
    assert build_problem_snapshot(p).hash == snapshot.hash


def test_python_physics_model_is_deeply_frozen_without_changing_snapshot_identity():
    from pops.physics.facade import Model

    model = Model("frozen-scalar")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[u])
    p = pops.Problem(name="physics-freeze").block("u", physics=model)

    snapshot = p.freeze()

    assert model.frozen and model._m.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        model.flux(x=[2 * u], y=[u])
    with pytest.raises(RuntimeError, match="frozen"):
        model._m.gamma = 1.2
    assert build_problem_snapshot(p).hash == snapshot.hash


def test_later_freeze_failure_restores_python_physics_cascade_exactly():
    from pops.physics.facade import Model

    class FailingSpatial(BrickDescriptor):
        def freeze(self):
            super().freeze()
            raise RuntimeError("spatial freeze failed")

    model = Model("rollback-scalar")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[u])
    p = pops.Problem(name="physics-rollback").block(
        "u", physics=model, spatial=FailingSpatial("bad", "native"))
    before = build_problem_snapshot(p).hash

    with pytest.raises(RuntimeError, match="spatial freeze failed"):
        p.freeze()

    assert not model.frozen and not model._m.frozen
    assert build_problem_snapshot(p).hash == before
    model.flux(x=[2 * u], y=[u])  # complex child/container state was restored, not merely the facade bit


@pytest.mark.parametrize("mutate", [
    lambda p: p.block("extra", physics=_model()),
    lambda p: p.add_block("extra2", _model()),
    lambda p: p.param(pops.physics.ConstParam("gamma", 1.4)),
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
    r = Refine.on("rho").above(0.05)  # mutates during build
    assert r.validate() is True


def test_problem_freeze_seals_member_descriptors():
    # Freezing the Problem cascades freeze to the typed member descriptors it holds (a field
    # problem's typed solver). The block's runtime spatial brick is not a typed Descriptor, so the
    # cascade seals what it can: the field registry's FieldProblem descriptors.
    from pops.math import unknown, laplacian
    from pops.ir.expr import Var
    from pops.fields import PoissonProblem
    field = PoissonProblem(unknown=unknown("phi"),
                           equation=(-laplacian(unknown("phi")) == Var("rho", "cons")))
    p = pops.Problem(name="plasma").block("ne", physics=_model()).field(field)
    p.freeze()
    # The FieldProblem descriptor is sealed: a post-freeze attribute mutation raises.
    with pytest.raises(RuntimeError, match="frozen"):
        field.solver = "changed"


# ---------------------------------------------------------------------------
# Program freeze (via compile) + cache-key fold.
# ---------------------------------------------------------------------------

def test_program_freeze_raises_on_new_node():
    prog = pops.time.Program("t")
    prog.state("ne")
    prog.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        prog.state("ne2")


def test_cache_key_fold_composes_not_replaces():
    from pops.problem._snapshot import fold_snapshot_hash

    class _Handle:
        _cache_key = "model=abc|kokkos=1|mpi=0|precision=double"

    h = _Handle()
    fold_snapshot_hash(h, "a" * 64)
    # The base key (with the compile stream's tokens) is PRESERVED; the snapshot hash is appended.
    assert h._cache_key.startswith("model=abc|kokkos=1|mpi=0|precision=double|")
    assert "problem_snapshot=" + "a" * 64 in h._cache_key


def test_compiled_handle_is_sealed_after_public_compile():
    from pops.codegen.loader import CompiledProblem

    handle = CompiledProblem("x.so", None, None, "abi", "c++", "c++23")
    handle._advanced_attach = "ok"  # the advanced compile_problem route stays attachable
    handle._seal()
    with pytest.raises(AttributeError, match="immutable after pops.compile"):
        handle.so_path = "y.so"
    with pytest.raises(AttributeError, match="ADC-563"):
        handle._layout = object()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
