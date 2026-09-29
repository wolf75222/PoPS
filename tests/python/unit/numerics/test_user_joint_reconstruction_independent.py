"""Adversarial source contracts for a joint, vector-valued User reconstruction.

These tests deliberately use only the public authoring surface. Native face and
transactional tests are specified separately in the independent T2 report.
"""

import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import reconstruction
from pops.params import RuntimeParam


def _group():
    frame = Rectangle("joint_user_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("joint_user_independent", frame=frame)
    left = model.species("left", state=("a", "b"))
    right = model.species("right", state=("c", "d", "e"))
    foreign = pops.Model("joint_user_independent", frame=frame)
    alien = foreign.species("right", state=("c", "d", "e"))
    return model, left, right, alien


def test_vector_body_reads_declared_other_row_and_unions_the_stencil():
    _, left, right, _ = _group()
    policy = reconstruction.User(
        lambda sample: (
            sample(0)[0] + .13 * (sample(3, right)[2] - sample(-1, right)[2]),
            sample(0)[1] - .07 * (sample(1, right)[0] - sample(-2)[0]),
        ), state=left, sampling=(right,), formal_order=1)
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    assert authenticated_user_reconstruction(policy) is policy
    assert policy.options["ghost_depth"] == 4
    assert policy.options["stencil_min_offset"] == -2
    assert policy.options["stencil_max_offset"] == 3


def test_vector_output_width_and_declared_sampling_are_exact():
    _, left, right, alien = _group()
    with pytest.raises((TypeError, ValueError), match="component|width|length|arity"):
        reconstruction.User(lambda sample: (sample(0)[0],),
                            state=left, sampling=(right,), formal_order=1)
    with pytest.raises((TypeError, ValueError), match="sampling|state|owner|model"):
        reconstruction.User(lambda sample: (sample(0, alien)[0], sample(0)[1]),
                            state=left, sampling=(right,), formal_order=1)


def test_live_parameter_capture_changes_identity_without_baking_a_value():
    model, left, right, _ = _group()
    alpha = model.value(model.param(RuntimeParam("alpha", default=.125)))
    beta = model.value(model.param(RuntimeParam("beta", default=.375)))
    calls = []

    def authored(sample):
        calls.append(1)
        return (sample(0)[0] + alpha * sample(1, right)[2],
                sample(0)[1] + beta * sample(-1, right)[0])

    policy = reconstruction.User(authored, state=left, sampling=(right,), formal_order=1)
    identity = policy.options["source_identity"]
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    authenticated_user_reconstruction(policy)
    assert calls == [1]
    swapped = reconstruction.User(
        lambda sample: (sample(0)[0] + beta * sample(1, right)[2],
                        sample(0)[1] + alpha * sample(-1, right)[0]),
        state=left, sampling=(right,), formal_order=1)
    assert swapped.options["source_identity"] != identity


def test_existing_scalar_user_contract_remains_available():
    scalar = reconstruction.User(lambda sample: sample(0) + .1 * (sample(1) - sample(-1)),
                                 formal_order=1)
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    assert authenticated_user_reconstruction(scalar) is scalar
    assert scalar.options["ghost_depth"] == 2


def test_frozen_sampling_owner_cannot_be_swapped_for_a_homonymous_foreign_model():
    _, left, right, alien = _group()
    policy = reconstruction.User(
        lambda sample: (sample(0)[0] + sample(1, right)[2], sample(0)[1]),
        state=left, sampling=(right,), formal_order=1)
    policy.options["sampling"] = (alien,)
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    with pytest.raises(ValueError, match="source|identity|state|sampling|owner"):
        authenticated_user_reconstruction(policy)
