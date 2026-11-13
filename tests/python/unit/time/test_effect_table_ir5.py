"""The effect table of the Program IR (IR5; G02 lot L, ROM-20) and the readers that use it.

The table replaced the private lists of ``graph_functions``, of the structured-control emitter and
of the finite Kokkos emitter. The identities below were computed at ``api03/integration`` 448c3d3b,
before the table: GraphFunction snapshots keep their two effect words, so no identity moves.
"""
import pytest

from pops._ir.effects import EFFECTS, Effect, _row, effect, graph_effect_words, region_safe
from pops.codegen.program_emit_finite_arrays import FINITE_OPS
from pops.codegen.program_emit_structured_control import STRUCTURED_OPS
from pops.graph_functions import graph
from pops.time._program.distributed_arrays import GLOBAL_OPS, LOCAL_OPS
from pops.time._program.field_representation import FIELD_REPRESENTATION_OPS
from pops.time._program.versioned_values import VERSIONED_OPS
from pops.time.variant_types import Bool, Float64, Int64, Product

IDENTITIES_448C3D3B = {
    "division": "976754c0f56133c94b4327fbfe645bbbdf8f874b3d1bfd87e782bf2d015272bd",
    "integer": "ec2bc576ef043b1f133e08590ff860d2f042d798ea4dc32d3b0b24877c1c789f",
    "branch": "52ee66c4a1f52d187afa1a255c1550fef137968db8e3cb619468155b44aee058",
    "iterate": "b93b4762f26e459d83fc4b086bb3499bc8563f1864a4551d30b0e503f94b3a2c",
    "finite_map_reduce": "0c28fa8d9d4b0f2317ee81656a9f81d1889c5154c5e47f2f43e34932f7673a00",
    "capture": "376dab386ba575bb698923c89ba67ebf0789662e5d14ece0ec6d22dc9fae55ec",
    "pure": "a75f9cbec67b6ce9d17402e12089caf5fb83cb91a43f3b02ec84f296fefdb99d",
}


def _functions():
    functions = {"division": graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: 1 / a["x"])}
    functions["integer"] = graph(inputs=Product(n=Int64), output=Product(m=Int64, odd=Bool))(
        lambda p, a: p.product(m=a["n"] * 3 + 1, odd=a["n"] % 2 == 1))
    functions["branch"] = graph(inputs=Product(x=Float64), output=Float64)(
        lambda p, a: p.branch(a["x"] > 0, lambda t: t.float64(2) * a["x"], lambda t: -a["x"]))

    def iterate(p, a):
        result = p.iterate(a["x"], lambda t, x: x * t.float64(.5), lambda t, x: x < 1, max_steps=8)
        return p.match(result, ok=lambda t, s: s["state"], err=lambda t, e: t.float64(-1))
    functions["iterate"] = graph(inputs=Product(x=Float64), output=Float64)(iterate)

    def finite(p, a):
        domain = p.range_set(4, capacity=4, ownership="replicated")
        values = p.map(domain, lambda b, i: b.float64(b.index_value(i)) * a["x"])
        return p.reduce(values, p.float64(0), lambda t, s, item: s + item, order="ascending")
    functions["finite_map_reduce"] = graph(inputs=Product(x=Float64), output=Float64)(finite)

    def captured(p, a):
        result = p.capture(lambda t: t.float64(1) / a["x"])
        return p.match(result, ok=lambda t, x: x, err=lambda t, e: t.float64(0))
    functions["capture"] = graph(inputs=Product(x=Float64), output=Float64)(captured)
    functions["pure"] = graph(inputs=Product(x=Float64), output=Product(y=Float64, z=Float64))(
        lambda p, a: p.product(y=a["x"], z=a["x"]))
    return functions


def test_graph_function_identities_and_effect_words_do_not_move():
    functions = _functions()
    assert {name: f.identity for name, f in functions.items()} == IDENTITIES_448C3D3B
    assert functions["pure"].effects == frozenset()
    assert functions["division"].effects == {"failure", "collective"}


def test_every_operation_of_the_dispatch_catalogs_has_a_row():
    scoped = VERSIONED_OPS - {"continuation_open", "continuation_publish"}
    for catalog in (STRUCTURED_OPS, FINITE_OPS, GLOBAL_OPS, LOCAL_OPS, FIELD_REPRESENTATION_OPS, scoped):
        assert set(catalog) <= set(EFFECTS), sorted(set(catalog) - set(EFFECTS))
    for op in STRUCTURED_OPS | FINITE_OPS | FIELD_REPRESENTATION_OPS | scoped:
        assert region_safe(op), op
    assert not any(region_safe(op) for op in GLOBAL_OPS)


def test_an_operation_without_a_row_has_no_contract():
    with pytest.raises(ValueError, match="no declared effect contract: no_such_op"):
        graph_effect_words("no_such_op")
    with pytest.raises(ValueError, match="no row in the effect table: no_such_op"):
        effect("no_such_op")
    assert not region_safe("no_such_op")
    with pytest.raises(ValueError, match="no declared effect contract: state"):
        graph_effect_words("state")


@pytest.mark.parametrize("words", ("graph graph", "graph unknown", "graph algorithm rendezvous",
                                   "device ordered", "fails", "rendezvous region"))
def test_a_malformed_row_is_refused(words):
    with pytest.raises(ValueError):
        _row(words)


def test_rows_read_as_their_words():
    assert effect("reduce") == Effect(graph=True, fails=False, collective="algorithm", region=True,
                                      device=None, host_region=False)
    assert effect("region_index").device == "ordered"
    assert effect("structured_iterate").host_region
    assert graph_effect_words("product") == frozenset()
