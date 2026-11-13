"""A refinement region reads a runtime parameter of its model (N12 lot 3c-2a, spec 0.3 DM13b, Q4).

``model.value(p)`` in a region reads the RuntimeParam ``p`` of the model of a block whose state the
region reads, from that block's RuntimeParams table, as the body of a ``StencilFlux`` does (core
commit 2f44e80f). These tests cover the resolved rows, the identity, the C++ and the refusals; the
native run with a value bound away from the default is in
``tests/python/integration/amr/test_refinement_criterion_model_parameters_native.py``.
"""
from __future__ import annotations

import re

import pops
import pytest
from pops._ir.values import RuntimeParamRef
from pops.amr import Coarsen, Tag, largest_cells
from pops.codegen.program_emit_refinement_criterion import emit_refinement_criterion
from pops.mesh._amr import ResolvedRefinementCriterion
from pops.mesh._amr.refinement_criterion import (
    RefinementCriterionModelParameter,
    criterion_parameter_routes,
)
from pops.mesh._amr.regrid_decision import RegridBodyError, RegridWhen
from pops.params import RuntimeParam
from tests.python.support.refinement_criterion_cases import advection_case


def _model_rules(names):
    return (
        Tag(lambda cell: cell.value(names.state) > names.model_value("nu")),
        Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")),
    )


def _case_rules(names):
    return (
        Tag(lambda cell: cell.value(names.state) > names.value("refine")),
        Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")),
    )


def _criterion(rules, **options) -> ResolvedRefinementCriterion:
    case, layout, _ = advection_case(rules, **options)
    criterion = pops.resolve(pops.validate(case), layout=layout).bootstrap_plan.tagging
    assert type(criterion) is ResolvedRefinementCriterion
    return criterion


def test_a_region_reads_a_runtime_parameter_of_the_model_of_the_block_it_reads() -> None:
    criterion = _criterion(_model_rules, model_parameters={"nu": 0.5})

    (row,) = criterion.model_parameters
    assert type(row) is RefinementCriterionModelParameter
    assert (row.block, row.name) == ("tracer", "nu")
    assert row.parameter.is_instance and row.parameter.param_kind == "runtime"
    # The case parameters keep their place: the model rows come after them.
    assert [handle.local_id for handle in criterion.parameters] == ["coarsen_threshold"]
    assert criterion.inspect()["model_parameters"] == [row.canonical_identity()]


def test_the_identity_names_the_parameter_and_not_its_value() -> None:
    first = _criterion(_model_rules, model_parameters={"nu": 0.5})
    second = _criterion(_model_rules, model_parameters={"nu": 0.8})
    assert first.qualified_id == second.qualified_id
    by_case = _criterion(_case_rules, model_parameters={"nu": 0.5})
    assert by_case.qualified_id != first.qualified_id


def test_a_criterion_without_model_parameters_keeps_its_identity_payload() -> None:
    criterion = _criterion(_case_rules)
    assert criterion.model_parameters == ()
    assert "model_parameters" not in criterion.canonical_identity()
    assert "model_parameters" not in criterion.runtime_criterion_data(
        {handle: 0.1 for handle in criterion.parameters})


def test_the_model_parameter_follows_the_case_parameters_in_the_patch_table() -> None:
    criterion = _criterion(_model_rules, model_parameters={"nu": 0.5})
    source = emit_refinement_criterion(criterion, native_dimension=2)

    assert "pops::Real parameters[2];" in source
    assert re.search(r"= parameters\[1\];", source)
    assert re.search(r"= parameters\[0\];", source)
    assert "0.5" not in source


def test_the_program_routes_give_each_model_parameter_its_block_slot() -> None:
    criterion = _criterion(_model_rules, model_parameters={"nu": 0.5})

    routes = criterion_parameter_routes(
        criterion, ((3, "tracer"),), ((3, "kappa", 0, 0.1), (3, "nu", 1, 0.5)))
    assert routes == ((3, 1),)
    with pytest.raises(ValueError, match=r"'nu' of block 'tracer'.*case\.param"):
        criterion_parameter_routes(criterion, ((3, "tracer"),), ((3, "kappa", 0, 0.1),))
    with pytest.raises(ValueError, match="no slot"):
        criterion_parameter_routes(criterion, ((0, "other"),), ((0, "nu", 0, 0.5),))


def test_a_parameter_without_its_declaring_handle_is_refused() -> None:
    def rules(names):
        return (Tag(lambda cell: cell.value(names.state) > RuntimeParamRef("free", 0.5)),
                Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")))

    with pytest.raises(TypeError, match="'free' without its declaring handle"):
        _criterion(rules)


def test_a_parameter_of_a_model_that_no_block_instantiates_is_refused() -> None:
    other = pops.Model("unrelated_model")
    foreign = other.param(RuntimeParam("foreign", default=0.5))

    def rules(names):
        return (Tag(lambda cell: cell.value(names.state) > other.value(foreign)),
                Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")))

    # Resolution refuses it by its declaration before the criterion sees it.
    with pytest.raises(Exception, match=r"parameter::foreign::.*which no block in this case "
                                        r"instantiates"):
        _criterion(rules)


def test_a_when_body_refuses_a_model_parameter_by_its_name() -> None:
    _, _, names = advection_case(_case_rules, model_parameters={"nu": 0.5})

    with pytest.raises(RegridBodyError, match=r"a when body reads the runtime parameter 'nu'.*"
                                               r"case\.param"):
        RegridWhen.trace(
            lambda B: largest_cells(lambda cell: cell.value(names.state)) > names.model_value("nu"))
    with pytest.raises(RegridBodyError, match=r"largest_cells\(f\) reads the runtime parameter "
                                               r"'nu'"):
        RegridWhen.trace(lambda B: largest_cells(
            lambda cell: cell.value(names.state) - names.model_value("nu")) > 0.0)
