"""Independent authoring contracts for source-compiled scalar reconstruction."""
from __future__ import annotations

import pytest

from pops.math import where
from pops.numerics import reconstruction


def test_stencil_support_includes_both_conditional_branches():
    recipe = reconstruction.User(
        lambda sample: where(sample(0) > 0, lambda: sample(-3), lambda: sample(2)),
        formal_order=1)
    assert recipe.options["stencil_min_offset"] == -3
    assert recipe.options["stencil_max_offset"] == 2
    assert recipe.options["ghost_depth"] == 4
    assert recipe.options["sample_offsets"] == (-3, 0, 2)
    assert recipe.capabilities["componentwise_scalar"] is True
    assert recipe.capabilities["oriented_sample"] is True


def test_discarded_python_intermediate_does_not_inflate_live_support():
    def body(sample):
        sample(12)
        return sample(0) + .25 * (sample(1) - sample(-1))

    recipe = reconstruction.User(body, formal_order=2)
    direct = reconstruction.User(
        lambda sample: sample(0) + .25 * (sample(1) - sample(-1)), formal_order=2)
    assert recipe.options["sample_offsets"] == (-1, 0, 1)
    assert recipe.options["ghost_depth"] == 2
    assert recipe.options["source_identity"] == direct.options["source_identity"]


@pytest.mark.parametrize("body", [
    lambda sample: sample(True), lambda sample: sample(1.),
    lambda sample: sample("1"), lambda sample: sample(sample(0)),
])
def test_offsets_require_exact_static_python_integers(body):
    with pytest.raises(TypeError, match="offset.*exact integer"):
        reconstruction.User(body, formal_order=1)


@pytest.mark.parametrize("body", [
    lambda sample: sample(-(1 << 31) - 1), lambda sample: sample(1 << 31),
])
def test_offsets_outside_native_integer_representation_are_rejected(body):
    with pytest.raises(ValueError, match="native int32"):
        reconstruction.User(body, formal_order=1)


@pytest.mark.parametrize("body", [
    lambda sample: sample(-(1 << 31)), lambda sample: sample((1 << 31) - 1),
    lambda sample: sample(-(1 << 30)) + sample(1 << 30),
])
def test_stencil_halo_and_span_must_fit_native_integer_representation(body):
    with pytest.raises(ValueError, match="native int32 ghost/count"):
        reconstruction.User(body, formal_order=1)


def test_stencil_has_no_arbitrary_small_radius_limit():
    recipe = reconstruction.User(lambda sample: sample(-17) + sample(17), formal_order=1)
    assert recipe.options["sample_offsets"] == (-17, 17)
    assert recipe.options["stencil_min_offset"] == -17
    assert recipe.options["stencil_max_offset"] == 17
    assert recipe.options["ghost_depth"] == 18


@pytest.mark.parametrize("order", [True, 0, -1, 1., "2"])
def test_order_is_an_explicit_positive_integer_assertion(order):
    with pytest.raises(ValueError, match="formal_order.*positive exact integer"):
        reconstruction.User(lambda sample: sample(0), formal_order=order)


def test_order_must_fit_the_native_static_integer_metadata():
    with pytest.raises(ValueError, match="formal_order.*native int32"):
        reconstruction.User(lambda sample: sample(0), formal_order=1 << 31)


@pytest.mark.parametrize("body", [
    lambda sample: (sample(0), sample(1)), lambda sample: [sample(0)],
])
def test_vector_return_does_not_silently_become_componentwise(body):
    with pytest.raises(TypeError, match="scalar per component.*vector"):
        reconstruction.User(body, formal_order=1)


def test_python_control_flow_does_not_choose_a_symbolic_branch():
    with pytest.raises(TypeError, match="truth value"):
        reconstruction.User(lambda sample: sample(1) if sample(0) > 0 else sample(-1),
                            formal_order=1)


def test_mutable_capture_is_materialized_once_and_cannot_change_authored_body():
    coefficients = [.25]
    calls = []

    def body(sample):
        calls.append(1)
        return sample(0) + coefficients[0] * (sample(1) - sample(-1))

    recipe = reconstruction.User(body, formal_order=2)
    before = recipe.options["source_identity"]
    coefficients[0] = .75
    assert calls == [1]
    same = reconstruction.User(
        lambda sample: sample(0) + .25 * (sample(1) - sample(-1)), formal_order=2)
    changed = reconstruction.User(body, formal_order=2)
    assert recipe.options["source_identity"] == before == same.options["source_identity"]
    assert changed.options["source_identity"] != before
    assert calls == [1, 1]


def test_source_identity_covers_math_support_and_declared_order():
    base = reconstruction.User(lambda sample: sample(0), formal_order=1, name="base")
    renamed = reconstruction.User(lambda sample: sample(0), formal_order=1, name="renamed")
    bodies = (
        reconstruction.User(lambda sample: 2 * sample(0), formal_order=1),
        reconstruction.User(lambda sample: sample(1), formal_order=1),
        reconstruction.User(lambda sample: sample(0), formal_order=2),
    )
    assert base.options["source_identity"] == renamed.options["source_identity"]
    assert len({base.options["source_identity"], *(r.options["source_identity"] for r in bodies)}) == 4


def test_foreign_physical_variable_is_not_a_stencil_capture():
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.representations import Conservative
    from pops.spaces import CellState

    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("foreign", frame=frame)
    state = model.state("U", components=("q",), representation=Conservative(),
                        space=CellState(frame=frame))
    with pytest.raises((ValueError, NotImplementedError), match="outside sample|leaf.*lowering"):
        reconstruction.User(lambda sample: sample(0) + state[0], formal_order=1)


def test_public_case_freeze_and_resolve_preserve_source_stencil():
    import pops
    from tests.python.integration.runtime.test_user_reconstruction_adversarial_runtime import make_case

    case, layout, _ = make_case(guarded=True)
    validated = pops.validate(case)
    before = case.snapshot.hash
    resolved = pops.resolve(validated, layout=layout)
    assert resolved.blocks
    assert case.snapshot.hash == before


def test_runtime_parameter_is_inferred_from_the_live_body_without_baking_its_value():
    import pops
    from pops.params import RuntimeParam

    model = pops.Model("coefficient_owner")
    coefficient = model.value(model.param(RuntimeParam("coefficient", default=.25)))
    recipe = reconstruction.User(lambda sample: coefficient * sample(0), formal_order=1)
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    assert authenticated_user_reconstruction(recipe) is recipe
    assert recipe.options["runtime_captures"] == (("coefficient", coefficient.handle.qualified_id),)
    changed = model.value(model.params["coefficient"])
    assert reconstruction.User(lambda sample: changed * sample(0), formal_order=1).options[
        "source_identity"] == recipe.options["source_identity"]


@pytest.mark.parametrize("mutation", ("expression", "source_identity", "ghost_depth"))
def test_modified_source_descriptor_cannot_route_as_authenticated_native_method(mutation):
    from pops.runtime._bricks_scheme import Spatial

    recipe = reconstruction.User(lambda sample: sample(0), formal_order=1)
    if mutation == "expression":
        recipe.expression = reconstruction.User(lambda sample: sample(1), formal_order=1).expression
    elif mutation == "source_identity":
        recipe.options["source_identity"] = "0" * 64
    else:
        recipe.options["ghost_depth"] = 0
    with pytest.raises(ValueError, match="body changed|source identity changed|ghost contract"):
        Spatial(reconstruction=recipe)
