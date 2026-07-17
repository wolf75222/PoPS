"""Runtime-parameter identity for standalone named elliptic RHS bricks."""

from pops.fields import FieldOutput
from pops.frames import Cartesian2D
from pops.math import laplacian
from pops.params import RuntimeParam
from pops.physics import Model
from pops.domain import Rectangle


def test_named_field_only_runtime_param_uses_the_shared_bind_carrier() -> None:
    frame = Rectangle(
        "named-field-runtime-param-domain",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    model = Model("named_field_runtime_param", frame=frame)
    state = model.state("U", components=("density",))
    (density,) = state
    x_axis, y_axis = frame.axes
    model.flux(
        "stationary_density",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * density,), y_axis: (0.0 * density,)},
        waves={x_axis: (0.0 * density,), y_axis: (0.0 * density,)},
    )
    coefficient = model.value(
        model.param(RuntimeParam("named_rhs_scale", default=2.5)))
    potential = model.field("potential")
    model.field_operator(
        "potential",
        unknown=potential,
        equation=-laplacian(potential) == coefficient * density,
        outputs=(FieldOutput("potential", potential),),
    )

    emitted = model.__pops_compiler_lowering__().emit_model
    assert [node.name for node in emitted._m.runtime_param_nodes()] == [
        "named_rhs_scale"
    ]

    loader = emitted._m.emit_cpp_native_loader(
        name="NamedFieldRuntimeParamGen", target="system")
    named_start = loader.index("struct NamedFieldRuntimeParamGenEll_potential {")
    named_end = loader.index("}  // namespace pops_generated", named_start)
    named_brick = loader[named_start:named_end]
    assert "pops::RuntimeParams params{1," in named_brick
    assert "params.get(0)" in named_brick
    assert 'pops_compiled_param_names() { return "named_rhs_scale"; }' in loader
    assert "pops::compiled_model::bind_runtime_params(" in loader
