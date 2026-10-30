"""N7 lot 1 (spec 0.3 IR8, IR9, IR13, IR14, IR17, IR20, IR21): typed index maps and their analyses.

The closed list of IR9 lives in ``pops._ir.index_maps`` and is public through ``pops.regions``.
These tests check the normal forms against the definitions of the constructors: sound (equal
normal forms are equal maps) everywhere, and canonical (equal maps have equal normal forms) on the
fragments where the module states it. Each map carries its type, the entity kinds it reads and
gives, whatever its spelling, and compositions are checked on it. The analyses answer three
values, the halo is exact on every kind of entity read, and only a proved fact enables a
property: the others are refused with a typed refusal that names the region and the first
blocking map. Level E of K6: source, no compilation.
"""
import itertools
import random
from collections import defaultdict
from fractions import Fraction

import pytest

import pops.regions as regions
from pops._ir.index_analysis import (
    Read, Region, Truth, check_expected_class, classify, conservation, halo, injective, require)
from pops._ir.index_maps import Adjacency, ArithmeticMap, ComputedMap, DataMap, IndexMap, Translate
from pops._ir.refusals import ClassNotReached, ConservationNotStructural, PropertyNotDerived
from pops.regions import (
    Child, Compose, Embed, Incidence, Inverse, Parent, Project, Shift, Trace, boundary, cell, face)


# ---------------------------------------------------------------------------------------------
# The definitions of the constructors, written independently of the normal forms.
# ---------------------------------------------------------------------------------------------
def reference(map_, index):
    kind = type(map_).__name__
    if kind == "Shift":
        return tuple(i + o for i, o in zip(index, map_.vector, strict=True))
    if kind == "Incidence":
        return tuple(i - (axis == map_.src.axis and map_.side == "left")
                     for axis, i in enumerate(index))
    if kind == "Trace":
        axis, side = map_.piece.axis, map_.piece.piece
        step = map_.depth if side == "low" else -(1 + map_.depth)
        return tuple(i + step*(d == axis) for d, i in enumerate(index))
    if kind == "Parent":
        return tuple(i // map_.ratio for i in index)
    if kind == "Child":
        return tuple(map_.ratio*i + o for i, o in zip(index, map_.child_offset, strict=True))
    if kind == "Project":
        return index[:map_.base_rank]
    if kind == "Embed":
        return index + map_.fibre
    if kind == "Compose":
        return reference(map_.outer, reference(map_.inner, index))
    if kind == "Inverse" and type(map_.of).__name__ == "Incidence":
        # The face whose ``side`` incident cell is the given cell.
        return tuple(i + (axis == map_.of.src.axis and map_.of.side == "left")
                     for axis, i in enumerate(index))
    raise TypeError(kind)


def _cells(rng, depth, ratios, rank):
    if depth == 0 or rng.random() < .3:
        choice = rng.randrange(3)
        if choice == 0:
            return Shift(tuple(rng.randint(-3, 3) for _ in range(rank)))
        if choice == 1:
            return Parent(rng.choice(ratios))
        ratio = rng.choice(ratios)
        return Child(ratio, tuple(rng.randrange(ratio) for _ in range(rank)))
    return Compose(_cells(rng, depth - 1, ratios, rank), _cells(rng, depth - 1, ratios, rank))


def _product(rng, depth, source, target):
    # Maps between X (rank 1) and ProductSet(X, V) (rank 2): "x" and "p".
    if depth == 0 or rng.random() < .3:
        if source == target:
            return Shift(tuple(rng.randint(-3, 3) for _ in range(1 if source == "x" else 2)))
        if source == "p":
            return Project(1)
        return Embed((rng.randint(-2, 2),), 1)
    middle = rng.choice("xp")
    return Compose(_product(rng, depth - 1, middle, target), _product(rng, depth - 1, source, middle))


def _faces(rng, depth):
    # Maps from the faces normal to x of a rank-2 lattice to its cells.
    if depth == 0 or rng.random() < .4:
        return Incidence(face(0), cell, rng.choice(("left", "right")))
    return Compose(Shift((rng.randint(-2, 2), rng.randint(-2, 2))), _faces(rng, depth - 1))


def _check(maps, grid, *, canonical):
    groups = defaultdict(set)
    for map_ in maps:
        signature = tuple(reference(map_, index) for index in grid)
        assert tuple(map_(index) for index in grid) == signature, map_
        groups[signature].add(map_.normal_form())
    if canonical:
        assert all(len(forms) == 1 for forms in groups.values())
    return len(groups)


def test_normal_forms_are_sound_on_every_fragment():
    rng = random.Random(20261030)
    grid2 = list(itertools.product(range(-7, 8), repeat=2))
    _check([_cells(rng, 4, (2, 3, 4), 2) for _ in range(1500)], grid2, canonical=False)
    _check([_faces(rng, 4) for _ in range(300)], grid2, canonical=False)


@pytest.mark.parametrize("ratios", ((2,), (2, 4)))
def test_normal_forms_are_canonical_on_levels_whose_ratios_divide(ratios):
    rng = random.Random(len(ratios))
    grid = [(i,) for i in range(-600, 601)]
    distinct = _check([_cells(rng, 4, ratios, 1) for _ in range(600)], grid, canonical=True)
    assert distinct > 150   # the sample spans many distinct maps, not a few


@pytest.mark.parametrize("source,target", (("x", "x"), ("p", "p"), ("x", "p"), ("p", "x")))
def test_normal_forms_are_canonical_on_product_lattices(source, target):
    rng = random.Random(hash((source, target)) % 1000)
    grid = ([(i,) for i in range(-20, 21)] if source == "x"
            else list(itertools.product(range(-9, 10), repeat=2)))
    _check([_product(rng, 4, source, target) for _ in range(800)], grid, canonical=True)


def test_face_maps_compare_by_normal_form():
    left, right = Incidence(face(0), cell, "left"), Incidence(face(0), cell, "right")
    # The left cell of a face shifted by +e_x is its right cell: one entity, two paths.
    assert Compose(Shift((1, 0)), left) == right
    assert hash(Compose(Shift((1, 0)), left)) == hash(right)
    assert Compose(Shift((0, 1)), left) != right
    assert left.offset(2) == (-1, 0) and right.offset(3) == (0, 0, 0)
    assert Compose(Shift((0, 2)), Compose(Shift((0, -1)), right)).offset(2) == (0, 1)
    assert Compose(Shift((0, 0)), right) == right
    # The kinds: faces in, cells out; a shift carries none of its own.
    assert (left.source, left.target) == (face(0), cell)
    assert (Shift((1,)).source, Shift((1,)).target) == (None, None)
    with pytest.raises(TypeError, match="not a translation"):
        Compose(Parent(2), Shift((1, 0))).offset(2)


def test_maps_carry_their_type():
    left0, right0 = Incidence(face(0), cell, "left"), Incidence(face(0), cell, "right")
    left1 = Incidence(face(1), cell, "left")
    # The next face along x goes from the faces normal to x to themselves; a shift acts on any
    # kind, so the two maps share a normal form and differ.
    next_face = Compose(Inverse(left0), right0)
    assert (next_face.source, next_face.target) == (face(0), face(0))
    assert next_face((3, 2)) == reference(next_face, (3, 2)) == (4, 2)
    assert next_face.normal_form() == Shift((1, 0)).normal_form() and next_face != Shift((1, 0))
    # The zero round trip is the identity of the faces normal to y, not of every kind.
    round_trip = Compose(Inverse(left1), left1)
    assert round_trip.normal_form() == () and (round_trip.source, round_trip.target) == (
        face(1), face(1))
    assert round_trip != Shift((0, 0))
    # Compositions are checked on the types, written with or without the round trip.
    for outer, inner in ((left1, next_face), (Parent(2), next_face), (left0, round_trip),
                         (left1, right0), (Parent(2), Inverse(right0))):
        with pytest.raises(TypeError, match="cannot compose"):
            Compose(outer, inner)
    # A map of type (None, None) takes the type of its neighbour.
    shifted = Compose(Shift((0, 1)), next_face)
    assert (shifted.source, shifted.target) == (face(0), face(0))
    assert Compose(right0, Compose(Shift((1, 0)), next_face)) == Compose(Shift((2, 0)), right0)
    assert (Inverse(next_face).source, Inverse(next_face).target) == (face(0), face(0))
    # A shift composed with a typed identity of a boundary piece keeps to the piece.
    low = boundary(0, "low")
    on_piece = Compose(Inverse(Trace(low, 0)), Trace(low, 0))
    assert on_piece.normal_form() == () and (on_piece.source, on_piece.target) == (low, low)
    assert (Compose(Shift((0, 1)), on_piece).source, Compose(Shift((0, 1)), on_piece).target) == (
        low, low)
    with pytest.raises(TypeError, match="off their boundary piece"):
        Compose(Shift((1, 0)), on_piece)
    # The same holds inward: the type of the outer map fixes what the inner one acts on.
    with pytest.raises(TypeError, match="off their boundary piece"):
        Compose(on_piece, Shift((1, 0)))
    with pytest.raises(TypeError, match="not an axis of its base"):
        Compose(round_trip, Project(1))


def test_the_two_product_rules_of_ir9():
    # Project composed with the rank-(r+s) Incidence is the rank-r Incidence composed with Project.
    assert (Compose(Project(1), Incidence(face(0), cell, "left"))
            == Compose(Incidence(face(0), cell, "left"), Project(1)))
    assert (Compose(Project(2), Compose(Shift((1, 0, 5)), Incidence(face(0), cell, "left")))
            == Compose(Incidence(face(0), cell, "right"), Project(2)))
    # Shift((o, 0)) composed with Embed(v) is Embed(v) composed with Shift(o).
    assert Compose(Shift((2, 0)), Embed((3,), 1)) == Compose(Embed((3,), 1), Shift((2,)))
    assert Compose(Shift((2, 1)), Embed((3,), 1)) == Compose(Embed((4,), 1), Shift((2,)))
    assert Compose(Project(1), Embed((3,), 1)).normal_form() == ()
    # A face normal to the fibre axis has no base face: the map keeps Project outermost.
    fibre_face = Compose(Project(1), Incidence(face(1), cell, "left"))
    assert fibre_face((3, 5)) == (3,) and type(fibre_face.normal_form()[0]).__name__ == "Drop"
    with pytest.raises(TypeError, match="not an axis of its base"):
        Compose(Project(1), Inverse(Incidence(face(1), cell, "left"))).normal_form()


def test_level_rules():
    assert Compose(Parent(2), Child(2, (1, 0))).normal_form() == ()
    assert Compose(Parent(2), Parent(2)) == Parent(4)
    assert Compose(Child(2, (1, 0)), Child(2, (0, 1))) == Child(4, (1, 2))
    assert Compose(Shift((1, 0)), Parent(2)) == Compose(Parent(2), Shift((2, 0)))
    assert Compose(Parent(2), Child(4, (3, 1))) == Child(2, (1, 0))
    assert Compose(Parent(4), Child(2, (1, 1))) == Parent(2)
    with pytest.raises(TypeError, match="cannot compose"):
        Compose(Incidence(face(0), cell, "left"), Parent(2)).normal_form()
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        Child(2, (2, 0))
    with pytest.raises(ValueError, match=">= 2"):
        Parent(1)


def test_trace_reads_the_cells_behind_a_boundary_piece():
    low, high = boundary(0, "low"), boundary(0, "high")
    assert Trace(low, 0)((0, 3)) == (0, 3)            # the right cell of a low face
    assert Trace(low, 2)((0, 3)) == (2, 3)
    assert Trace(high, 0)((8, 3)) == (7, 3)           # the left cell of a high face
    assert Trace(high, 1)((8, 3)) == (6, 3)
    assert Trace(low).normal == ((0, -1),) and Trace(high).normal == ((0, 1),)
    # Tangent shifts keep the faces on their piece; a normal shift does not.
    assert Compose(Trace(low, 1), Shift((0, 1)))((0, 3)) == (1, 4)
    with pytest.raises(TypeError, match="off their boundary piece"):
        Compose(Trace(low, 1), Shift((1, 0))).normal_form()
    with pytest.raises(TypeError, match="boundary piece"):
        Trace(face(0))


def test_inverses():
    left = Incidence(face(1), cell, "left")
    assert Inverse(left)((4, 5)) == (4, 6)            # the face above a cell along y
    assert Inverse(Inverse(left)) == left
    assert Compose(Inverse(left), left).normal_form() == ()
    child = Child(2, (1, 0))
    assert Inverse(child)((5, 4)) == (2, 2) and Inverse(child)((4, 4)) is None
    assert Compose(Inverse(child), child).normal_form() == ()
    assert Inverse(Embed((3,), 1))((7, 3)) == (7,) and Inverse(Embed((3,), 1))((7, 2)) is None
    # Parent and Project are not injective: their inverse is an adjacency, whose fan-out is the
    # product of one factor per blocking atom. |V| comes from the descriptor of the product set
    # (IR2): the factor names it without a value, whatever the rank of the base.
    adjacency = Inverse(Parent(2))
    assert type(adjacency) is Adjacency and adjacency.fan_out == (("ratio^rank", 2),)
    assert (adjacency.source, adjacency.target, adjacency.provenance) == (cell, cell, "typed")
    assert Inverse(Project(1)).fan_out == Inverse(Project(3)).fan_out == (
        ("fibre cardinality", None),)
    assert Inverse(Compose(Project(1), Parent(2))).fan_out == (
        ("fibre cardinality", None), ("ratio^rank", 2))
    coarse_face = Inverse(Compose(Parent(2), Incidence(face(0), cell, "left")))
    assert (coarse_face.source, coarse_face.target) == (cell, face(0))
    assert type(Inverse(Compose(Parent(2), Child(2, (0, 0))))) is not Adjacency


def test_offsets_are_integer_constants():
    with pytest.raises(TypeError, match="StructuredOffsetNotConstant"):
        Shift((1.5, 0))
    with pytest.raises(TypeError, match="tuple of integers"):
        Shift([1, 0])
    with pytest.raises(TypeError, match="faces normal to one axis"):
        Incidence(cell, cell, "left")
    with pytest.raises(ValueError, match="'left' or 'right'"):
        Incidence(face(0), cell, "up")


def test_maps_of_other_provenances_are_compared_by_key_and_never_rewritten():
    data = DataMap("graph", source=face(0), target=cell)
    assert data == DataMap("graph", source=face(0), target=cell)
    assert data != DataMap("other", source=face(0), target=cell)
    assert data != ComputedMap("graph", source=face(0), target=cell)
    assert data.provenance == "data" and ArithmeticMap("j", source=cell, target=cell).provenance \
        == "arithmetic"
    assert Compose(Shift((1, 0)), data).normal_form()[0] == Translate(None, None, ((0, 1),))
    assert injective(data).truth is Truth.UNKNOWN
    assert Inverse(data) != data and Inverse(Inverse(data)) == data
    # The provenance of a map is that of its normal form, the weakest of its atoms (IR8).
    computed = ComputedMap("departure", source=cell, target=cell)
    assert Compose(Shift((1, 0)), data).provenance == Inverse(data).provenance == "data"
    assert Compose(ArithmeticMap("j", source=cell, target=cell), data).provenance == "data"
    assert Compose(computed, data).provenance == "computed"
    assert Inverse(Compose(Parent(2), data)).provenance == "data"
    assert Compose(Shift((1, 0)), Incidence(face(0), cell, "left")).provenance == "typed"


def test_the_public_home_of_the_typed_maps():
    assert set(regions.__all__) == {"Child", "Compose", "Embed", "Incidence", "Inverse", "Parent",
                                    "Project", "Shift", "Trace", "boundary", "cell", "face"}
    assert all(issubclass(getattr(regions, name), IndexMap) for name in regions.__all__
               if name[0].isupper())


# ---------------------------------------------------------------------------------------------
# Analyses.
# ---------------------------------------------------------------------------------------------
def _face_region(offsets, *, axis=0, rank=2, targets=None, expected="face_incidence"):
    right = Incidence(face(axis), cell, "right")
    reads = tuple(Read(("state", 0), Compose(Shift(o), right)) for o in offsets)
    if targets is None:
        targets = ((Incidence(face(axis), cell, "left"), -1), (right, 1))
    return Region("probe", face(axis), rank, reads, targets, expected=expected)


def _brute_force_halo(offsets, axis, extents):
    # Every face of the owned cells [0, n) reads the cells i + o; count those outside.
    low, high = [0]*len(extents), [0]*len(extents)
    faces = [range(n + 1) if d == axis else range(n) for d, n in enumerate(extents)]
    for index in itertools.product(*faces):
        for offset in offsets:
            for d, n in enumerate(extents):
                position = index[d] + offset[d]
                low[d] = max(low[d], -position)
                high[d] = max(high[d], position - (n - 1))
    return tuple(low), tuple(high)


def test_halo_is_exact_box_arithmetic():
    rng = random.Random(3)
    for _ in range(60):
        axis = rng.randrange(2)
        offsets = [tuple(rng.randint(-3, 3) for _ in range(2)) for _ in range(rng.randint(1, 5))]
        fact = halo(_face_region(offsets, axis=axis))
        assert fact.truth is Truth.PROVED
        for extents in ((7, 8), (9, 7)):
            assert fact.value == _brute_force_halo(offsets, axis, extents)
    # The M05 reads: one cell on each side, where the b6 face formula stores two on y.
    m05 = [(-1, 0), (0, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)]
    assert halo(_face_region(m05)).value == ((1, 1), (1, 1))


def _owned(kind, extents):
    # A rank owns the cells [0, n) of each axis and the faces normal to a of its cells.
    return [range(n + 1) if kind.kind == "face" and kind.axis == d else range(n)
            for d, n in enumerate(extents)]


def _typed_read(rng, source, target):
    # A translation from ``source`` to ``target`` entities, written through the cells.
    offset = (rng.randint(-2, 2), rng.randint(-2, 2))
    inward = (Shift(offset) if source == cell
              else Compose(Shift(offset), Incidence(source, cell, rng.choice(("left", "right")))))
    if target == cell:
        return inward
    return Compose(Inverse(Incidence(target, cell, rng.choice(("left", "right")))), inward)


def _brute_force_typed_halo(region, extents):
    low, high = [0]*len(extents), [0]*len(extents)
    for index in itertools.product(*_owned(region.domain, extents)):
        for read in region.reads:
            target = region.domain if read.map.target is None else read.map.target
            entity = reference(read.map, index)
            assert read.map(index) == entity
            for d, (position, owned) in enumerate(zip(entity, _owned(target, extents),
                                                      strict=True)):
                low[d] = max(low[d], -position)
                high[d] = max(high[d], position - (len(owned) - 1))
    return tuple(low), tuple(high)


def test_halo_is_exact_on_every_kind_of_entity_read():
    rng = random.Random(30)
    kinds = (cell, face(0), face(1))
    for _ in range(120):
        domain = rng.choice(kinds)
        region = Region("probe", domain, 2, tuple(
            Read("u", _typed_read(rng, domain, rng.choice(kinds)))
            for _ in range(rng.randint(1, 4))))
        fact = halo(region)
        assert fact.truth is Truth.PROVED
        for extents in ((6, 7), (8, 5)):
            assert fact.value == _brute_force_typed_halo(region, extents)
    # A face read in faces of its own kind is one face beyond the owned [0, n], not two.
    left0, right0 = Incidence(face(0), cell, "left"), Incidence(face(0), cell, "right")
    next_face = Compose(Inverse(left0), right0)
    assert halo(Region("faces", face(0), 2, (Read("J", next_face),))).value == ((0, 0), (1, 0))
    assert halo(Region("faces", face(0), 2, (Read("J", Shift((1, 0))),))).value == (
        (0, 0), (1, 0))
    assert halo(Region("cells", cell, 2, (Read("J", Inverse(left0)),))).value == ((0, 0), (0, 0))


def test_halo_of_a_cell_region_and_a_filtered_read():
    reads = (Read("u", Shift((1, 0))), Read("u", Shift((-2, 0))), Read("v", Shift((0, 3))))
    region = Region("stencil", cell, 2, reads)
    assert halo(region).value == ((2, 0), (1, 3))
    assert region.halo(lambda array: array == "u").value == ((2, 0), (1, 0))


def test_unknown_halo_names_the_blocking_map_and_require_refuses_it():
    graph = DataMap("graph", source=face(0), target=cell)
    region = Region("probe", face(0), 2, (Read("u", Incidence(face(0), cell, "left")),
                                          Read("u", graph)), expected="csr")
    fact = halo(region)
    assert fact.truth is Truth.UNKNOWN and fact.blocking == graph
    with pytest.raises(PropertyNotDerived, match="halo") as refusal:
        require(fact, region)
    assert (refusal.value.region, refusal.value.property, refusal.value.blocking) == (
        "probe", "halo", graph)
    assert refusal.value.phase == "resolve" and refusal.value.rule == "IR13"
    assert halo(Region("edge", boundary(0, "low"), 2, ())).truth is Truth.UNKNOWN
    behind = Inverse(Trace(boundary(0, "low"), 0))
    fact = halo(Region("trace", cell, 2, (Read("g", behind),)))
    assert fact.truth is Truth.UNKNOWN and fact.blocking == behind


def test_conservation_is_three_valued():
    assert conservation(_face_region([(0, 0)])).truth is Truth.PROVED
    unbalanced = _face_region([(0, 0)], targets=((Incidence(face(0), cell, "left"), -1),
                                                 (Incidence(face(0), cell, "right"), 2)))
    assert conservation(unbalanced).truth is Truth.DISPROVED
    assert conservation(unbalanced).value == 1
    halves = ((Incidence(face(0), cell, "left"), Fraction(-1, 2)),
              (Incidence(face(0), cell, "right"), Fraction(1, 2)))
    assert conservation(_face_region([(0, 0)], targets=halves)).truth is Truth.PROVED
    symbolic = ((Incidence(face(0), cell, "left"), "w"), (Incidence(face(0), cell, "right"), 1))
    fact = conservation(_face_region([(0, 0)], targets=symbolic))
    assert fact.truth is Truth.UNKNOWN and fact.blocking == "w"
    assert conservation(Region("map", cell, 2, ())).truth is Truth.UNKNOWN


def test_injectivity():
    assert injective(Incidence(face(0), cell, "left")).truth is Truth.PROVED
    assert injective(Child(2, (0, 1))).truth is Truth.PROVED
    assert injective(Parent(2)).truth is Truth.DISPROVED
    assert injective(Project(1)).truth is Truth.UNKNOWN
    assert injective(Compose(Shift((1, 0)), Parent(2))).truth is Truth.UNKNOWN


def test_classification_reads_only_normal_forms():
    right = Incidence(face(0), cell, "right")
    assert classify(_face_region([(0, 0), (-1, 0)])) == ("face_incidence", None)
    assert classify(Region("s", cell, 2, (Read("u", Shift((1, 0))),))) == ("box_stencil", None)
    assert classify(Region("p", cell, 2, (Read("u", Shift((0, 0))),))) == ("pointwise", None)
    # Pointwise: every read and every delivery is the identity of the domain, a scatter to its
    # own entities included; a delivery elsewhere makes a stencil.
    assert classify(Region("p", cell, 2, (Read("u", Shift((0, 0))),),
                           ((Shift((0, 0)), 1),))) == ("pointwise", None)
    assert classify(Region("q", cell, 2, (Read("u", Shift((0, 0))),),
                           ((Shift((1, 0)), 1),))) == ("box_stencil", None)
    left1 = Incidence(face(1), cell, "left")
    assert classify(Region("f", face(1), 2, (Read("J", Compose(Inverse(left1), left1)),))) == (
        "pointwise", None)
    next_face = Compose(Inverse(Incidence(face(0), cell, "left")), right)
    assert classify(Region("n", face(0), 2, (Read("J", next_face),))) == ("face_incidence", None)
    graph = DataMap("graph", source=face(0), target=cell)
    region = Region("g", face(0), 2, (Read("u", right), Read("u", graph)))
    assert classify(region) == ("csr", graph)
    coarse = Compose(Parent(2), right)
    assert classify(Region("l", face(0), 2, (Read("u", coarse),))) == ("level_transfer", coarse)
    fibre = Compose(Project(1), Shift((0, 1)))
    assert classify(Region("r", cell, 2, (Read("u", fibre),))) == ("axis_reduction", fibre)
    moving = ComputedMap("departure", source=cell, target=cell)
    assert classify(Region("c", cell, 2, (Read("u", moving),))) == ("generic", moving)


def test_class_not_reached_names_the_region_and_the_first_blocking_map():
    graph = DataMap("graph", source=face(0), target=cell)
    region = Region("block fluid/face flux/axis 0", face(0), 2,
                    (Read("u", Incidence(face(0), cell, "left")), Read("u", graph)),
                    expected="face_incidence")
    with pytest.raises(ClassNotReached, match="block fluid/face flux/axis 0") as refusal:
        check_expected_class(region)
    assert (refusal.value.expected, refusal.value.reached, refusal.value.blocking) == (
        "face_incidence", "csr", graph)
    assert refusal.value.phase == "compile" and refusal.value.rule == "IR21"
    assert check_expected_class(_face_region([(0, 0)])) == "face_incidence"
    with pytest.raises(ValueError, match="unknown lowering class"):
        Region("x", cell, 2, (), expected="stencil")


def test_productions_deliver_side_values_without_an_identity():
    left, right = Incidence(face(0), cell, "left"), Incidence(face(0), cell, "right")
    region = Region("path", face(0), 2, (Read("u", left), Read("u", right)),
                    ((left, -1), (right, 1)), expected="face_incidence",
                    productions=((left, 1), (right, 1)))
    # Conservation reads the shared value only (IR17); a production is never an exchange (IR25).
    assert (conservation(region).truth, conservation(region).value) == (Truth.PROVED, 0)
    assert classify(region) == ("face_incidence", None)
    assert halo(region).value == ((1, 0), (1, 0))
    graph = DataMap("graph", source=face(0), target=cell)
    blocked = Region("path", face(0), 2, (Read("u", left),), ((left, -1), (right, 1)),
                     productions=((graph, 1),))
    assert classify(blocked) == ("csr", graph)
    # Every map of a region starts from its domain.
    with pytest.raises(TypeError, match="maps over"):
        Region("bad", face(1), 2, (), productions=((left, 1),))
    with pytest.raises(TypeError, match="maps over"):
        Region("bad", face(1), 2, (Read("u", left),))
    with pytest.raises(TypeError, match="pair"):
        Region("bad", face(0), 2, (), productions=(left,))


def test_require_enables_only_proved_facts():
    left, right = Incidence(face(0), cell, "left"), Incidence(face(0), cell, "right")
    balanced = _face_region([(0, 0)])
    assert require(conservation(balanced), balanced) == 0
    unbalanced = _face_region([(0, 0)], targets=((left, -1), (right, 2)))
    with pytest.raises(ConservationNotStructural, match="disproved: the analysis disproves") as \
            refusal:
        require(conservation(unbalanced), unbalanced)
    assert (refusal.value.rule, refusal.value.phase, refusal.value.property) == (
        "IR17", "resolve", "conservation")
    assert isinstance(refusal.value, PropertyNotDerived)
    symbolic = _face_region([(0, 0)], targets=((left, "w"), (right, 1)))
    with pytest.raises(ConservationNotStructural, match="unknown") as refusal:
        require(conservation(symbolic), symbolic)
    assert refusal.value.blocking == "w"
    # Another property keeps the refusal of IR13, Disproved as Unknown.
    with pytest.raises(PropertyNotDerived, match="disproves") as refusal:
        require(injective(Parent(2)), balanced)
    assert type(refusal.value) is PropertyNotDerived and refusal.value.rule == "IR13"
    assert require(injective(left), balanced) is None
