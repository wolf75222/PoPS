from collections import Counter

import pytest

from pops._ir.index_analysis import (
    FibreFoldRead, LocalControl, ProductRead, Read, Region, Truth, classify, halo,
    product_read_footprints, region_reads,
)
from pops._ir.index_maps import Compose, Embed, Incidence, Inverse, Parent, Project, Shift, cell, face
from pops._ir.index_sets import Configuration, box, cells, faces, product


def one_footprint(read, *, rank=1, entity=cell, extents=None):
    _, fact = product_read_footprints(Region("product read", entity, rank, (read,)),
                                      extents=extents)[0]
    return fact


def test_product_maps_keep_base_displacement_and_incidence():
    configuration = Configuration("base", (13,), (False,))
    base, boundary_faces = cells(configuration), faces(configuration, 0)
    fibre = box((5,))
    pinned = ProductRead("A", Compose(Shift((1, 1)), Embed((3,), 1)),
                         base, product(base, fibre), fibre)
    assert halo(Region("pin", cell, 1, (pinned,))).value == ((0,), (1,))
    assert list(one_footprint(pinned).value.indices((7,))) == [(8, 4)]

    displaced_projection = ProductRead("U", Compose(Project(1), Shift((1, 0))),
                                       product(base, fibre), base, fibre)
    assert halo(Region("drop", cell, 2, (displaced_projection,))).value == ((0, 0), (1, 0))
    assert list(one_footprint(displaced_projection, rank=2).value.indices((7, 3))) == [(8,)]
    needs_boundary = ProductRead("U", Compose(Project(1), Shift((0, 1))),
                                 product(base, fibre), base, fibre)
    fact = one_footprint(needs_boundary, rank=2)
    assert fact.truth is Truth.UNKNOWN and "translated variable fibre" in fact.detail

    incident = ProductRead("F", Compose(Embed((4,), 1), Incidence(face(0), cell, "left")),
                           boundary_faces, product(base, fibre), fibre)
    assert halo(Region("face", face(0), 1, (incident,))).value == ((1,), (0,))
    assert list(one_footprint(incident, entity=face(0)).value.indices((7,))) == [(6, 4)]


def test_typed_checks_survive_erased_atoms_and_symbolic_extents():
    base, fibre = box((13,)), box(("nv",))
    projection = Project(1)
    valid = ProductRead("A", Compose(projection, Embed((4,), 1)), base, base, fibre)
    invalid = ProductRead("A", Compose(projection, Embed((5,), 1)), base, base, fibre)
    assert valid.map == invalid.map and valid.map.normal_form() == ()
    assert one_footprint(valid).truth is Truth.UNKNOWN
    assert "nv" in one_footprint(valid).detail
    assert list(one_footprint(valid, extents={"nv": 5}).value.indices((7,))) == [(7,)]
    region = Region("bound even when equal maps", cell, 1, (valid, invalid))
    assert region_reads(region.reads) == (valid, invalid)
    with pytest.raises(ValueError, match="fibre index 5 is outside"):
        product_read_footprints(region, extents={"nv": 5})
    with pytest.raises(ValueError, match="positive integer"):
        one_footprint(valid, extents={"nv": 0})
    with pytest.raises(ValueError, match="outside"):
        bad = ProductRead("A", Compose(projection, Embed((4, 2), 1)),
                          base, base, box(("nv", 2)))
        one_footprint(bad)


def test_repin_reads_a_constant_and_unpin_keeps_its_partial_domain():
    base, fibre = box((13,)), box((5,))
    domain = product(base, fibre)
    repin = ProductRead("A", Compose(Embed((4,), 1), Project(1)), domain, domain, fibre)
    footprint = one_footprint(repin, rank=2).value
    assert list(footprint.indices((7, 1))) == [(7, 4)]
    with pytest.raises(ValueError, match="consumer index"):
        list(footprint.indices((7,)))
    unpin = Inverse(Embed((4,), 1))
    partial = ProductRead("U", unpin, domain, base, fibre)
    assert one_footprint(partial, rank=2).truth is Truth.UNKNOWN
    good = ProductRead("U", Compose(unpin, Embed((4,), 1)), base, base, fibre)
    assert list(one_footprint(good).value.indices((7,))) == [(7,)]
    bad = ProductRead("U", Compose(unpin, Embed((3,), 1)), base, base, fibre)
    with pytest.raises(ValueError, match="Unpin is undefined"):
        one_footprint(bad)


def test_control_body_retains_array_and_consumer_authority():
    base, fibre = box((13,)), box((5,))
    domain = product(base, fibre)
    point = ProductRead("field", Embed((4,), 1), base, domain, fibre)
    other_array = ProductRead("other", point.map, base, domain, fibre)
    other_pin = ProductRead("field", Embed((1,), 1), base, domain, fibre)
    adjacency = Inverse(Project(1))
    first_fold = FibreFoldRead("field", adjacency, domain, "ordered sum")
    second_fold = FibreFoldRead("field", adjacency, domain, "first maximum")
    predicate = Read("predicate", Shift((0,)))
    control = LocalControl("cond", (
        predicate, (point, first_fold), LocalControl("match", (
            other_array, (point, other_pin), second_fold,
        )),
    ))
    region = Region("all arms", cell, 1, (control,))
    assert region_reads(region.reads) == (
        predicate, point, first_fold, other_array, other_pin, second_fold,
    )
    assert tuple(read for read, _ in product_read_footprints(region)) == (
        point, first_fold, other_array, other_pin, second_fold,
    )
    assert tuple(read for read, _ in product_read_footprints(
        region, where=lambda array: array == "other",
    )) == (other_array,)
    assert classify(region) == ("axis_reduction", point.map)
    assert halo(region).value == ((0,), (0,))


def test_logical_partition_changes_transfers_not_the_footprints():
    base, fibre = box((13,)), box((5,))
    domain = product(base, fibre)
    point = ProductRead("A", Embed((4,), 1), base, domain, fibre)
    fold = FibreFoldRead("A", Inverse(Project(1)), domain, "ordered fold", Shift((1,)))
    projection = ProductRead("B", Project(1), domain, base, fibre)
    pin_path = one_footprint(point).value
    fold_path = one_footprint(fold).value
    project_path = one_footprint(projection, rank=2).value

    for sections in ((range(0, 5),), (range(0, 2), range(2, 5))):
        owners = {slot: part for part, slots in enumerate(sections) for slot in slots}

        for cell_index in (7, 8):
            pinned, = pin_path.indices((cell_index,))
            assert pinned == (cell_index, 4)
            assert owners[pinned[-1]] == len(sections) - 1
            contributions = list(fold_path.indices((cell_index,)))
            expected = [(cell_index + 1, slot) for slot in range(5)]
            assert contributions == expected
            assert Counter(owners[point_index[-1]] for point_index in contributions) == {
                part: len(slots) for part, slots in enumerate(sections)
            }
            values = {point_index: 100 * point_index[0] + point_index[1]
                      for point_index in expected}
            assert [values[index] for index in contributions] == [
                100 * (cell_index + 1) + slot for slot in range(5)
            ]
            requested_by = Counter(
                (owners[slot], key)
                for slot in range(5)
                for key in project_path.indices((cell_index, slot))
            )
            assert requested_by == {(part, (cell_index,)): len(slots)
                                    for part, slots in enumerate(sections)}
            # Owner zero stores the base value. Deduplication is by recipient and source,
            # independently of how often it is consumed on that recipient.
            assert {item for item in requested_by if item[0] != 0} == (
                {(1, (cell_index,))} if len(sections) == 2 else set()
            )


def test_unsupported_inverse_and_symbolic_folds_keep_the_blocker():
    domain = product(box((13,)), box(("nv", 2)))
    read = FibreFoldRead("A", Inverse(Project(1)), domain, "ordered fold")
    fact = one_footprint(read)
    assert fact.truth is Truth.UNKNOWN and fact.blocking is read and "nv" in fact.detail
    bound = one_footprint(read, extents={"nv": 2}).value
    assert list(bound.indices((7,))) == [(7, 0, 0), (7, 1, 0), (7, 0, 1), (7, 1, 1)]
    other = FibreFoldRead("A", Inverse(Parent(2)), domain, "ordered fold")
    fact = one_footprint(other)
    assert fact.truth is Truth.UNKNOWN and fact.blocking is other and "inverse" in fact.detail
    unsupported_base = FibreFoldRead("A", Inverse(Project(1)), domain, "ordered fold", Parent(2))
    fact = one_footprint(unsupported_base)
    assert fact.truth is Truth.UNKNOWN and fact.blocking is unsupported_base
    assert "no fibre fold composition" in fact.detail
