"""N16 (ROM-109), the lot that lifts the table for three words: the rows of b6 that serve
``cell_total(1)``, ``point_value(1)`` and ``cell_average(1, quadrature=Q)`` (N16 5.1, 5.5, A1;
review of #803, R1).

- ``plan.representations.add`` takes the four identities of one constant mode, matched whole, and
  stores each as the contract writes it, so two spellings of one identity give one artifact; any
  other definition is refused with N16 lot 2.
- A sample and a stated rule have the rate of an average: their C++ is the C++ of the case
  without a row (MG1). Their initial value is bound as data; the analytic route, which projects
  cell averages, is refused before any C++ with N16 lot 2 (the projection by encode), as for a
  total.
- The acceptance test ``finite()`` of the surface reads the same on a total as on its density, so
  the total's lowering serves exactly that test, and no other reduction of a total.

The native oracle tests are in ``tests/python/integration/runtime/test_n16_three_words_native.py``.
"""
from __future__ import annotations

import pytest

import pops
from pops.codegen._cell_total_consumers import (
    CellTotalNotYetElaborated, finite_guard_values,
)
from pops.lib.quadrature import GaussLegendre
from pops.numerics import DiscretizationPlan
from pops.numerics.plan import served_representation
from pops.representations.definition import (
    RepresentationDefinition, cell_average, cell_total, coefficients, point_value,
)
from pops.time._program.traced_method import NotYetElaborated

from examples.api_v02.models_N16_three_words import REPRESENTATIONS, build_words


class _Rule:
    """A rule written by hand, the data of GaussLegendre(3) (the core names no rule, D0)."""

    def __init__(self, rule):
        self.nodes, self.weights, self.interval = rule.nodes, rule.weights, rule.interval


def _resolved(**keywords):
    case, layout, _, _ = build_words(**keywords)
    return pops.resolve(pops.validate(case), layout=layout)


def _state():
    case, _, _, _ = build_words(represent=None)
    model = case._block_registry.spec("line")["model"]
    return next(iter(model.states.values()))


@pytest.mark.parametrize("definition", [
    cell_average(1), cell_total(1), point_value(1),
    cell_average(1, quadrature=GaussLegendre(3)), cell_average(1, quadrature=GaussLegendre(2)),
    cell_total(2), point_value(3)], ids=["average", "total", "sample", "rule3", "rule2",
                                         "total-rank2", "sample-rank3"])
def test_the_four_identities_of_one_constant_mode_are_served(definition):
    assert served_representation(definition) == definition
    if definition.rank == 1:
        DiscretizationPlan().representations.add(_state(), definition)


def test_two_spellings_of_one_identity_give_one_row():
    """The rule as a word of the library and the same written by hand have one identity; the
    row stores the definition as the contract writes it, so the artifacts are equal."""
    word = cell_average(1, quadrature=GaussLegendre(3))
    written = cell_average(1, quadrature=_Rule(GaussLegendre(3)))
    assert word.identity == written.identity
    assert served_representation(written).to_data() == served_representation(word).to_data()


@pytest.mark.parametrize("definition", [
    RepresentationDefinition(decode=coefficients[0], rank=1, kind="coefficient"),
    RepresentationDefinition(decode=2*coefficients[0], rank=1, kind="average"),
    RepresentationDefinition(decode=coefficients[0], rank=1, encode=None, kind="sample"),
], ids=["coefficient", "doubled", "sample-without-encode"])
def test_another_definition_of_one_constant_mode_names_n16_lot_2(definition):
    assert served_representation(definition) is None
    with pytest.raises(NotYetElaborated, match="N16 lot 2"):
        DiscretizationPlan().representations.add(_state(), definition)


@pytest.mark.parametrize("represent", ["sample", "rule"])
def test_a_sample_and_a_stated_rule_emit_the_cpp_of_the_case_without_a_row(represent):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    def cpp(plan):
        return emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(
            plan.blocks))

    plan = _resolved(represent=represent, layout="uniform", name="N16_words_mg1")
    bare = _resolved(represent=None, layout="uniform", name="N16_words_mg1")
    (row,) = plan.blocks[0].numerics.representations
    assert row.method == REPRESENTATIONS[represent]
    assert plan.blocks[0].numerics.cell_total_states() == ()
    assert cpp(plan) == cpp(bare)
    assert plan.snapshot.semantic_identity != bare.snapshot.semantic_identity


@pytest.mark.parametrize("represent, reason", [
    ("total", "cell_total_initial_projection"), ("sample", "represented_initial_projection"),
    ("rule", "represented_initial_projection")])
def test_an_analytic_initial_projection_names_n16_lot_2(represent, reason):
    """b6's analytic route projects cell averages; under a total, a sample or a stated rule the
    initial value is bound as data until N16 lot 2 gives the projection by encode."""
    with pytest.raises(CellTotalNotYetElaborated) as caught:
        _resolved(represent=represent, layout="uniform", initial="analytic")
    assert caught.value.reason == reason
    assert "N16 lot 2 (the projection of an initial value by the encode" in str(caught.value)
    assert "BindArray" in str(caught.value)
    _resolved(represent="average", layout="uniform", initial="analytic")


@pytest.mark.parametrize("layout", ["graded", "uniform"])
@pytest.mark.parametrize("represent", sorted(REPRESENTATIONS, key=str))
def test_each_row_resolves_bound_as_data_on_both_layouts(represent, layout):
    numerics = _resolved(represent=represent, layout=layout).blocks[0].numerics
    assert len(numerics.representations) == (0 if represent is None else 1)


def _surface_total_case(accept_finite=True):
    import pops.surface as surface
    from pops.lib.numerics import upwind
    from pops.lib.representations import CellTotal
    from pops.lib.time.methods import ssprk2
    from pops.surface.math import ddt, div

    domain = surface.Box(x=surface.Periodic(0, 3))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=1.0 * u)
    model.balance(ddt(u) == -div(advection))
    u0 = model.data("u0", surface.Scalar(domain))
    numerics = surface.Numerics(represent={u: CellTotal()}, schemes={advection: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=16)})


def test_the_acceptance_test_finite_is_served_on_a_total():
    """The surface's default finite() is acceptance_guard(finite) of scalar_isfinite of
    reduce(norm_inf_all): the three values are served, and the case resolves."""
    from pops.time._program.traversal import all_ops

    elaborated = _surface_total_case().elaborated
    plan = pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)
    values = list(all_ops(plan.time))
    guards = finite_guard_values(values)
    assert sorted(value.op for value in values if value.id in guards) == [
        "acceptance_guard", "reduce", "scalar_isfinite"]
    assert len(plan.blocks[0].numerics.cell_total_states()) == 1


def test_a_reduction_of_a_total_read_by_another_value_is_not_the_finite_test():
    """A norm read by the finite test and by anything else is a diagnostic of U: not served."""

    class Value:
        def __init__(self, ident, op, inputs=(), **attrs):
            self.id, self.op, self.inputs, self.attrs = ident, op, tuple(inputs), attrs

    state = Value(0, "state")
    norm = Value(1, "reduce", (state,), kind="norm_inf_all")
    test = Value(2, "scalar_isfinite", (norm,))
    guard = Value(3, "acceptance_guard", (state, test), guard="finite")
    assert finite_guard_values([state, norm, test, guard]) == {1, 2, 3}
    other = Value(4, "linear_combine", (norm,))
    assert finite_guard_values([state, norm, test, guard, other]) == frozenset()
    maximum = Value(1, "reduce", (state,), kind="max")
    test = Value(2, "scalar_isfinite", (maximum,))
    guard = Value(3, "acceptance_guard", (state, test), guard="finite")
    assert finite_guard_values([state, maximum, test, guard]) == frozenset()
