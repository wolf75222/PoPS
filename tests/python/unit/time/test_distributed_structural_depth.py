"""Distributed authoring retains the same finite contracts at structural depth."""
from types import SimpleNamespace

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.time import Program
from pops.time._program.distributed_arrays import (
    Distributed, local_effects, local_escaping_errors, transportable,
)
from typed_program_support import typed_state


def test_deep_distributed_product_reaches_real_cpp_lowering():
    program = Program("deep_distributed_product")
    state = typed_state(program, "q", state_name="U")
    distributed = Distributed(program, storage_bytes=64 << 20)
    domain = distributed.domain((0, 1), capacity=1)

    def local(builder, index):
        def body(p):
            value = p.float64(1.)
            for _ in range(1024):
                value = p.product(value=value)
            return value
        return builder.capture(body)

    mapped = distributed.local_map(domain, local, byte_capacity=16 << 20)
    program.require_success(mapped)
    program.commit(state.next, program.value("accepted", state.n, at=state.next.point))
    assert program.validate()
    source = emit_cpp_program(program)
    assert "ctx.distributed_registry()" in source
    assert "::map<pops::kNativeDimension>" in source
    assert "ContinuationKind::Product" in source
    assert "PyObject" not in source


def test_deep_transport_preserves_leaf_authority_and_rejects_cycles():
    leaf = {"vtype": "int64"}
    signature = leaf
    for index in range(1500):
        if index % 3 == 0:
            signature = {"vtype": "array", "ownership": "replicated",
                         "family": index, "capacity": 1, "element": signature}
        else:
            signature = {"vtype": "variant" if index % 3 == 1 else "product",
                         "members": (("value", signature),)}
    transportable(signature)
    leaf["vtype"] = "state"
    with pytest.raises(TypeError, match="legacy fields"):
        transportable(signature)
    leaf["vtype"] = "int64"
    transportable({"vtype": "product", "members": (("a", leaf), ("b", leaf))})
    cyclic = {"vtype": "product"}
    cyclic["members"] = (("self", cyclic),)
    with pytest.raises(TypeError, match="cyclic"):
        transportable(cyclic)


def test_deep_local_effects_preserve_capture_boundaries_and_fail_closed():
    from pops.codegen.program_emit_distributed_arrays import _raised_errors

    program = Program("local_effect_depth")
    result = program.success(program.float64(1.))
    leaf = SimpleNamespace(op="result_success", inputs=(result,), attrs={})
    node = leaf
    for _ in range(1500):
        node = SimpleNamespace(op="structured_fold", attrs={"body_block": (node,)})
    local_effects((node,))
    errors = local_escaping_errors((node,))
    assert len(errors) == 1
    assert list(_raised_errors((node,)).values()) == list(errors)
    assert local_escaping_errors((leaf, leaf)) == errors + errors
    captured = SimpleNamespace(op="capture_result", attrs={"body_block": (node,)})
    assert not local_escaping_errors((captured,))
    assert not _raised_errors((captured,))
    leaf.op = "distributed_import"
    with pytest.raises(TypeError, match="collective or unsupported"):
        local_effects((captured,))
    cyclic = SimpleNamespace(op="structured_fold", attrs={})
    cyclic.attrs["body_block"] = (cyclic,)
    with pytest.raises(TypeError, match="cyclic"):
        local_effects((cyclic,))
