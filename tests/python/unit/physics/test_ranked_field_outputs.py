"""Field gradients materialize exactly one component per inferred Cartesian axis."""
from __future__ import annotations

import pytest

from pops.domain import CartesianDomain
from pops.fields import FieldOutput, GradientOutput
from pops.math import laplacian
from pops.physics import Model


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_gradient_output_components_follow_model_frame(dimension: int) -> None:
    frame = CartesianDomain(
        "field-output-%d" % dimension,
        (0.0,) * dimension,
        (1.0,) * dimension,
    ).frame()
    model = Model("field_output_%d" % dimension, frame=frame)
    (rho,) = model.state("U", components=("rho",))
    potential = model.field("potential")

    model.field_operator(
        "poisson",
        unknown=potential,
        equation=(-laplacian(potential) == rho),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric", potential),
        ),
    )

    components = model.field_spaces()["potential"].components
    assert components == (
        "potential",
        *("electric_" + axis.name for axis in frame.axes),
    )


def test_gradient_output_requires_one_dimension_authority() -> None:
    model = Model("unranked_field_output")
    (rho,) = model.state("U", components=("rho",))
    potential = model.field("potential")

    with pytest.raises(ValueError, match="bounded Cartesian frame"):
        model.field_operator(
            "poisson",
            unknown=potential,
            equation=(-laplacian(potential) == rho),
            outputs=(
                FieldOutput("potential", potential),
                GradientOutput("electric", potential),
            ),
        )
