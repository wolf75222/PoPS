"""ADC-563: the real freeze lifecycle -- Descriptor / Problem / Program / snapshot.

A Problem is MUTABLE while authored and FROZEN by pops.compile. After freeze, every mutating setter
RAISES (naming the frozen object), the member descriptors are sealed, and Problem.freeze() returns a
stable ProblemSnapshot whose .hash the compile cache key folds in. There is NO warning, NO
shallow-copy escape. Pure Python; needs only ``import pops``.
"""
import pytest

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
    assert d["schema_version"] == 1
    assert json.loads(json.dumps(d, sort_keys=True)) == d  # no runtime object, no numpy array


# ---------------------------------------------------------------------------
# Problem.freeze(): idempotent, mutation-after-freeze RAISES.
# ---------------------------------------------------------------------------

def test_problem_freeze_returns_snapshot_and_is_idempotent():
    p = _problem()
    snap = p.freeze()
    assert p.frozen and p.snapshot is snap
    assert p.freeze() is snap  # idempotent: the same snapshot


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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
