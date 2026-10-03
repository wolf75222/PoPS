"""Genuine Source Module admission; no compiler or Native execution is simulated."""
import pytest
from pops.model import Module, Rate
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.module_emit_brick import emit_cpp_brick
from pops.domain import CartesianDomain
from pops.frames import Cartesian


def model(dimension, reverse=False, framed=True):
    frame = CartesianDomain("declared physical storage", (0.,)*dimension, (1.,)*dimension).frame(Cartesian(dimension))
    module = Module("two independent source domains", frame=frame) if framed else Module("unframed source domains")
    components = ("last", "first", "middle")
    if reverse: components = tuple(reversed(components))
    module.state_space("reacting", components, sampling="cell_average")
    selected = module.state_spaces()["reacting"]
    @module.operator("local_kinetics", signature=selected >> Rate(selected), kind="local_source")
    def reaction(state):
        return (-state[2], state[0]+state[2], state[1]-state[0])
    return module, frame


@pytest.mark.parametrize("dimension", (1,2,3))
@pytest.mark.parametrize("reverse", (False,True))
def test_exact_nonfirst_source_state_uses_declared_frame_not_flux(dimension, reverse):
    module, frame = model(dimension, reverse)
    before = module.module_hash()
    emitter, canonical = lower_and_validate(module, state_space="reacting")
    body = emit_cpp_brick(emitter._m, name="ActualSourceOnly")
    assert emitter._m.n_vars == 3
    assert emitter._m._program_only_storage_axes == tuple(axis.name for axis in frame.axes)
    assert "static constexpr int dimension = %d;" % dimension in body
    assert "program_only_storage = true" in body
    assert "State flux(" not in body and "max_wave_speed(" not in body
    assert not emitter._m._flux
    assert "pops_install_native" in emitter._m.emit_cpp_native_loader(name="ActualSourceOnly", target="system")
    assert canonical.module_hash() == module.module_hash() == before


def test_absent_physical_frame_never_infers_native_or_other_state_geometry():
    module, _ = model(2, framed=False)
    emitter, _ = lower_and_validate(module, state_space="reacting")
    with pytest.raises(ValueError, match="set_flux|Cartesian axis"):
        emit_cpp_brick(emitter._m)


def test_exact_selection_among_independent_states_does_not_use_first_state():
    from pops.codegen.module_lowering import _module_to_model
    module, frame = model(2)
    module.state_space("other", ("aux1", "aux2"), sampling="cell_average")
    emitter = _module_to_model(module, state_space="other")
    body = emit_cpp_brick(emitter._m)
    assert emitter._m.n_vars == 2
    assert "static constexpr int dimension = 2;" in body
    assert not emitter._m._flux
