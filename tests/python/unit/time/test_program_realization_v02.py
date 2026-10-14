"""Exact balance/Use/commit coverage and C++ emission; not native execution."""
from dataclasses import replace

import pops
import pytest
from pops import math as pm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, ProgramRealization
from pops.numerics.stencil_flux import StencilFlux
from pops.realization import Realize, Use
from pops.time import FixedDt


def assembled(*, kind="valid", repeated=False, source_only=False, derived=False):
    frame = Rectangle("program_balance_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("declared_transport_and_exchange", frame=frame)
    U = model.state("U", components=("a", "b"), sampling="cell_average")
    flux = model.flux("transport", state=U, frame=frame,
        components={frame.axes[0]: (U[0], U[1]), frame.axes[1]: (0., 0.)})
    source = model.source("exchange", on=U, value=(U[1]-U[0], U[0]-U[1]))
    rhs = source if source_only else -pm.div(flux)+source
    if repeated:
        rhs = rhs+source
    balance = model.rate("balance", equation=pm.ddt(U) == rhs)
    uses = tuple(Use(row, "face" if row.kind == "flux" else "cell", "stage0", "rate")
                 for row in balance.occurrences)
    selected = uses
    if kind == "missing":
        selected = uses[:-1]
    elif kind == "altered":
        selected = (replace(uses[0], occurrence=replace(uses[0].occurrence, coefficient=-2)), *uses[1:])
    elif kind == "foreign":
        other = model.rate("other_balance", equation=pm.ddt(U) == source)
        model.select_balance(balance)
        selected = (replace(uses[0], occurrence=other.occurrences[0]), *uses[1:])
    plan = DiscretizationPlan()
    method_type = ProgramRealization
    if derived:
        class DerivedProgramRealization(ProgramRealization):
            pass
        method_type = DerivedProgramRealization
    plan.rates.add(balance, method_type(uses=selected, ghost_depth=3))
    case = pops.Case("program_realization_case")
    block = case.block("material", model)
    case.numerics(plan, block=block)
    p = pops.Program("explicit_balance_equations")
    q = p.state(block[U])
    target = q
    if kind == "foreign_block":
        other_block = case.block("other", model)
        other_plan = DiscretizationPlan()
        other_plan.rates.add(balance, ProgramRealization(uses=uses, ghost_depth=3))
        case.numerics(other_plan, block=other_block)
        target = p.state(other_block[U])
    stencil = StencilFlux(dimension=2, components=2, radius=1,
        body=lambda sample, spacing, axis: (
            ((sample((-1, 0), 0), sample((-1, 0), 1)), 1/spacing[0])
            if axis == 0 else ((0., 0.), 0.)))
    source_operator = model.module.operator_handle("exchange")
    def body(builder, old):
        if source_only:
            return tuple(builder.source(source_operator, old) for _ in uses)
        face = builder.stencil_flux_divergence("authored_upwind_flux", old, flux=stencil)
        return (face, *(builder.source(source_operator, old) for _ in uses[1:]))
    recipe = Realize("exact_occurrences", owns=uses, bindings={"old": target.n}, body=body,
                     outputs={use: (index,) for index, use in enumerate(uses)})
    realized = recipe.evaluate(p)
    if kind in {"duplicate", "discarded", "branch_selected"}:
        second = recipe.evaluate(p)
    if kind == "ancestral":
        predictor = p.value("provisional_predictor", target.n + p.dt * realized[uses[0]], at=target.n.point)
        second = Realize("corrected_occurrences", owns=uses,
                         bindings={"old": predictor}, body=body,
                         outputs={use: (index,) for index, use in enumerate(uses)}).evaluate(p)
    total = sum((realized[use] for use in uses[1:]), realized[uses[0]])
    if kind == "duplicate":
        total = total+sum((second[use] for use in uses[1:]), second[uses[0]])
    if kind in {"discarded", "ancestral"}:
        total = sum((second[use] for use in uses[1:]), second[uses[0]])
    if kind == "branch_selected":
        total = p.branch(p.requested_dt() > 0., lambda b: realized[uses[0]], lambda b: second[uses[0]])
    if kind == "branch_missing":
        other = p.value("unrealized", p.source(source_operator, target.n))
        total = p.branch(p.requested_dt() > 0., lambda b: realized[uses[0]], lambda b: other)
    if kind == "predicate_only":
        other = p.source(source_operator, target.n)
        total = p.branch(p.sum(realized[uses[0]]) > 0., lambda b: other, lambda b: other)
    if kind == "diagnostic_only":
        p.record_scalar("observed_exchange", p.sum(realized[uses[-1]]))
        total = target.n if source_only else realized[uses[0]]
    if kind == "support_only":
        from pops.time.expressions import ProgramExpression
        total = p.value("constant", ProgramExpression((0., 0.), realized[uses[0]]))
    if kind == "unrealized":
        # Same physical occurrence with another structural Use is not the one
        # requested by the numerical descriptor.
        extra = replace(uses[0], temporal_role="unrequested_stage")
        extra_value = Realize("extra", owns=(extra,), bindings={"old": q.n},
            body=lambda b, old: (old[0],old[1]), outputs={extra: ()}).evaluate(p)[extra]
        total = total+extra_value
    if kind == "hidden_rate":
        total = total+balance(target.n)
    p.commit(target.next, p.value("candidate", target.n+p.dt*total, at=target.next.point))
    if kind == "foreign_block":
        p.commit(q.next, p.value("unchanged_material", q.n, at=q.next.point))
    p.step_strategy(FixedDt(.001))
    case.program(p)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, model, balance, uses


@pytest.mark.parametrize("repeated", (False, True))
def test_actual_balance_resolves_emits_and_retains_equations_and_storage(repeated):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout, model, balance, uses = assembled(repeated=repeated)
    plan = pops.resolve(pops.validate(case), layout=layout)
    block = plan.blocks[0]
    assert len(balance.occurrences) == (3 if repeated else 2)
    assert len(block.numerics.rates[0].method.uses) == len(uses)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    native = graph.model_for_block("material")._m
    assert native._program_state_ghost_depth == 3
    assert native._program_only_storage_axes == ("x", "y")
    assert not native._flux and not native._eig
    assert model.module.operator_registry().get("balance").lowering["physical_balance"]
    cpp = emit_cpp_program(plan.time, model_graph=graph)
    assert "stencil_flux" in cpp
    assert "pops_shared_source_" in cpp
    assert "exact_occurrences" in cpp


@pytest.mark.parametrize("kind,match", (("missing", "missing or foreign"),
    ("foreign", "missing or foreign"), ("altered", "altered"),
    ("duplicate", "exactly one owned SSA"), ("diagnostic_only", "exact state commit"),
    ("foreign_block", "exactly one owned SSA"), ("unrealized", "undeclared owned Use"),
    ("hidden_rate", "cannot also use a native rate")))
def test_missing_forged_duplicate_or_misdirected_coverage_refuses(kind, match):
    with pytest.raises(ValueError, match=match):
        case, layout, *_ = assembled(kind=kind)
        pops.resolve(pops.validate(case), layout=layout)


def test_same_occurrence_can_have_explicit_distinct_temporal_uses_but_not_duplicate_uses():
    _, _, _, balance, uses = assembled()
    first = uses[0]
    with pytest.raises(ValueError, match="repeats"):
        ProgramRealization(uses=(first, first))
    stages = ProgramRealization(uses=(*uses, replace(first, temporal_role="stage1")))
    assert stages.validate_balance_view(balance.view)


def test_geometric_support_is_not_numerical_delivery_of_a_balance_source():
    case, layout, *_ = assembled(kind="support_only", source_only=True)
    with pytest.raises(ValueError, match="exact state commit"):
        pops.resolve(pops.validate(case), layout=layout)


def test_descriptor_subtype_cannot_skip_mandatory_balance_coverage():
    case, layout, *_ = assembled(kind="diagnostic_only", source_only=True, derived=True)
    with pytest.raises(ValueError, match="exact state commit"):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("kind", ("discarded", "ancestral", "branch_selected"))
def test_provisional_evaluations_are_not_duplicate_final_deliveries(kind):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout, *_ = assembled(kind=kind, source_only=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    source = emit_cpp_program(plan.time, model_graph=graph)
    assert "exact_occurrences" in source


@pytest.mark.parametrize("kind", ("branch_missing", "predicate_only"))
def test_each_selected_path_must_numerically_deliver_its_required_use(kind):
    case, layout, *_ = assembled(kind=kind, source_only=True)
    with pytest.raises(ValueError, match="exact state commit"):
        pops.resolve(pops.validate(case), layout=layout)
