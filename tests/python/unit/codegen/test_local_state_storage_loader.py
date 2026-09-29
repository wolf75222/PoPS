"""Local States need real storage packages, without an invented transport law."""
import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.domain import CartesianDomain
from pops.frames import Cartesian


@pytest.mark.parametrize("dimension", (1,2,3))
@pytest.mark.parametrize("target", ("system","amr_system"))
def test_flux_free_state_emits_complete_native_loader(dimension, target):
    frame = CartesianDomain("local_domain", (0.,)*dimension, (1.,)*dimension).frame(Cartesian(dimension))
    model = pops.Model("local_unknown", frame=frame)
    state = model.state("U", components=("a","b","c"))
    before = model.module.module_hash()
    emitted, canonical = lower_and_validate(model)
    source = emitted._m.emit_cpp_native_loader(name="LocalUnknown", target=target)
    assert "program_only_storage = true" in source
    assert "static constexpr int dimension = %d;" % dimension in source
    entry = "pops_install_native" if target == "system" else "pops_install_native_amr"
    assert "void %s(" % entry in source
    assert "State flux(" not in source
    assert "max_wave_speed(" not in source
    assert not emitted._m._flux and not model._dsl._m._flux
    assert canonical.module_hash() == model.module.module_hash() == before
    assert state.space.frame == frame.canonical_id


@pytest.mark.parametrize("reverse", (False,True))
def test_resolved_local_product_emits_every_storage_package(reverse):
    from tests.python.support.local_residual_product_case import make_case
    case,layout,_ = make_case(reverse=reverse)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    for block in resolved.blocks:
        emitter,_ = lower_and_validate(block.model, state_space=block.state_spaces[0],
            resolved_operations=block.resolved_operations, numerics=block.numerics)
        source = emitter._m.emit_cpp_native_loader(name="LocalProduct"+block.name,
            target="system", consumer_owner_qid=block.instance_owner_qid)
        assert "program_only_storage = true" in source
        assert "void pops_install_native(" in source
        assert "State flux(" not in source


def test_unframed_local_state_does_not_invent_a_dimension():
    model = pops.Model("unframed_local")
    model.state("U", components=("u",))
    emitter,_ = lower_and_validate(model)
    with pytest.raises(ValueError,match="set_flux|Cartesian axis"):
        emitter._m.emit_cpp_native_loader(name="UnframedLocal",target="system")
