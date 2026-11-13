"""The reads walker of the region analyses (G02 lot L, design note §5.4; spec IR4, IR20, IR21, B4).

``region_reads`` returns the reads of a region body with local control: the union over the arms of
``cond`` and ``match`` and over the bodies of ``fold``, ``scan`` and ``iterate``, each read once.
``classify``, ``check_expected_class`` and ``halo`` read a region through it, so a body with local
control has the class of its reads: a read in an arm that does not run counts, and an ``iterate``
reads what its step reads.
"""
import pytest

from pops._ir.index_analysis import (
    LocalControl, Read, Region, Truth, check_expected_class, classify, halo, region_reads)
from pops._ir.index_maps import DataMap
from pops._ir.refusals import ClassNotReached
from pops.regions import Incidence, Parent, Shift, cell, face

LEFT, RIGHT, CENTRE = Read("u", Shift((-1, 0))), Read("u", Shift((1, 0))), Read("u", Shift((0, 0)))
COARSE = Read("u", Parent(2))


def _region(body, expected=None, domain=cell):
    return Region("probe", domain, 2, body, expected=expected)


def test_the_union_over_the_arms_of_cond_and_match():
    body = (CENTRE, LocalControl("cond", ((LEFT,), (RIGHT,), (CENTRE,))),
            LocalControl("match", ((RIGHT,), (COARSE,), (LEFT,))))
    assert region_reads(body) == (CENTRE, LEFT, RIGHT, COARSE)


def test_a_loop_body_is_read_once_whatever_its_trip_count():
    step = (LEFT, RIGHT)
    for node in ("fold", "scan", "iterate"):
        body = (LocalControl(node, (step, (CENTRE,))),)
        assert region_reads(body) == (LEFT, RIGHT, CENTRE)
    nested = LocalControl("iterate", (LocalControl("fold", (LocalControl("iterate", (step,)),)),))
    assert region_reads((nested, nested)) == (LEFT, RIGHT)


def test_local_control_never_changes_a_class():
    flat = _region((LEFT, CENTRE, RIGHT))
    looped = _region((LocalControl("iterate", ((LEFT, CENTRE), (LocalControl("cond", (
        (RIGHT,), (CENTRE,))),))),))
    assert classify(flat) == classify(looped) == ("box_stencil", None)
    assert halo(flat) == halo(looped)
    assert halo(looped).truth is Truth.PROVED and halo(looped).value == ((1, 0), (1, 0))
    pointwise = _region((LocalControl("match", ((CENTRE,), (CENTRE,))),))
    assert classify(pointwise) == ("pointwise", None)


def test_a_read_in_an_arm_that_does_not_run_still_counts():
    region = _region((CENTRE, LocalControl("cond", ((CENTRE,), (LEFT,), (COARSE,)))),
                     expected="box_stencil")
    assert classify(region) == ("level_transfer", COARSE.map)
    with pytest.raises(ClassNotReached, match="first map that blocks it is") as refusal:
        check_expected_class(region)
    assert (refusal.value.reached, refusal.value.blocking) == ("level_transfer", COARSE.map)


def test_a_face_kernel_with_a_local_newton_reaches_face_incidence():
    # The local Newton of an exact Riemann solver: an iterate over the two incident cells.
    left = Read(("state", 0), Incidence(face(0), cell, "left"))
    right = Read(("state", 0), Incidence(face(0), cell, "right"))
    graph = Read(("state", 0), DataMap("graph", source=face(0), target=cell))
    deliveries = ((Incidence(face(0), cell, "left"), -1), (Incidence(face(0), cell, "right"), 1))
    newton = LocalControl("iterate", ((left, right), (left,)))
    region = Region("face/axis 0", face(0), 2, (newton,), deliveries, expected="face_incidence")
    assert check_expected_class(region) == "face_incidence"
    blocked = Region("face/axis 0", face(0), 2, (LocalControl("cond", ((newton,), (graph,))),),
                     deliveries, expected="face_incidence")
    with pytest.raises(ClassNotReached) as refusal:
        check_expected_class(blocked)
    assert (refusal.value.reached, refusal.value.blocking) == ("csr", graph.map)


def test_the_walker_does_not_recurse():
    body = (LEFT,)
    for depth in range(5000):
        body = (LocalControl(("cond", "match", "fold", "scan", "iterate")[depth % 5], (body,)),)
    assert region_reads(body) == (LEFT,)
    assert classify(_region(body)) == ("box_stencil", None)


def test_malformed_bodies_are_refused():
    with pytest.raises(ValueError, match="unknown local control 'while'"):
        LocalControl("while", ((LEFT,),))
    with pytest.raises(TypeError, match="non-empty tuple of bodies"):
        LocalControl("cond", ())
    with pytest.raises(TypeError, match="Read records and local control"):
        region_reads((LEFT, "u"))
    with pytest.raises(TypeError, match="Read records and local control"):
        _region((LocalControl("fold", ((Shift((1, 0)),),)),))
