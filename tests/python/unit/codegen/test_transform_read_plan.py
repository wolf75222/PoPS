"""A read certificate closes used inputs, not the unrelated model environment."""
from __future__ import annotations

import json

import pytest

from pops._ir.primitive_expansion import expand_primitive_recipes
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_emit_kernels import ProgramProviderPlans, _model_impl
from pops.codegen.program_emit_model_kernels import _emit_local_transform_kernel
from pops.codegen.program_transform_read_plan import transform_read_plan
from pops.math import where
from pops.params import RuntimeParam
from pops.physics import Model


def prepared(*, default=2., extra=False, provider="coefficient", primitive=False):
    model = Model("closed_transform")
    q, = model.state("U", components=("q",))
    a = model.value(model.param(RuntimeParam("factor", default=default)))
    extra_value = model.aux("unused_provider")
    coefficient = model.aux(provider)
    expression = a*q*q + coefficient
    if primitive:
        expression = model.primitive("expanded_map", expression)
    model.local_transform("map", (where(q > 0, lambda: expression, lambda: q),))
    if extra:
        unused = model.value(model.param(RuntimeParam("aaa_unused", default=11.)))
        model.local_transform("unrelated", (unused*q + extra_value,))
    emit_model, _ = lower_and_validate(model, facade=model)
    impl = _model_impl(emit_model)
    declaration = impl._local_transforms["map"]
    roots = expand_primitive_recipes(
        (*declaration["expressions"], declaration["valid_if"]), impl.prim_defs)
    plans = ProgramProviderPlans()
    binding = plans.bind(impl, roots, "closed/map")
    return emit_model, impl, roots, plans, binding


def test_unused_parameter_and_provider_are_absent_from_closed_schema():
    a, b = prepared(), prepared(extra=True)
    first = transform_read_plan(a[1], a[2], a[4])
    second = transform_read_plan(b[1], b[2], b[4])
    assert first.parameter_indices == (0,)
    assert second.parameter_indices == (1,)
    a_data, b_data = json.loads(first.descriptor), json.loads(second.descriptor)
    assert a_data["parameters"] == b_data["parameters"]
    assert a_data["expressions"] == b_data["expressions"]
    # Adding an operator changes the owning module, so the provider authority
    # must differ even though the selected free-read schema is unchanged.
    assert a_data["providers"][0]["key"]["owner_qid"] != b_data["providers"][0]["key"]["owner_qid"]
    assert first.count == 2 and first.components == 5
    assert "unused_provider" not in first.descriptor
    assert "aaa_unused" not in first.descriptor


def test_defaults_are_not_the_executed_parameter_binding():
    a, b = prepared(default=2.), prepared(default=9.)
    assert transform_read_plan(a[1], a[2], a[4]).identity == transform_read_plan(
        b[1], b[2], b[4]).identity
    # Actual differing values must come from the native journal. A model default
    # in the static descriptor must never stand in for an executing get(index).


def test_transitive_primitive_has_the_same_exact_free_reads():
    a, b = prepared(), prepared(primitive=True)
    first, second = (transform_read_plan(row[1], row[2], row[4]) for row in (a, b))
    assert first.parameter_indices == second.parameter_indices
    assert first.provider_count == second.provider_count == 1
    assert json.loads(first.descriptor)["expressions"] == json.loads(second.descriptor)["expressions"]


def test_provider_identity_is_committed_even_when_values_could_match():
    a, b = prepared(provider="coefficient"), prepared(provider="other_coefficient")
    first = transform_read_plan(a[1], a[2], a[4])
    second = transform_read_plan(b[1], b[2], b[4])
    assert first.identity != second.identity
    rows = json.loads(first.descriptor)["providers"]
    assert len(rows) == 1 and rows[0]["key"]["component"] == "coefficient"
    assert rows[0]["contract"]["centering"] == "cell"


def test_missing_or_extra_provider_cannot_be_certified_as_the_emitted_kernel():
    _, impl, roots, _, binding = prepared()
    for slots in ({}, {"coefficient": 0, "unused_provider": 1}, {"coefficient": 1}):
        with pytest.raises(ValueError, match="exact provider kernel plan"):
            transform_read_plan(impl, roots, {**binding, "slots": slots})


def test_capture_wraps_actual_reads_inside_active_lazy_cell_evaluation():
    model, _, _, plans, _ = prepared()
    source = "\n".join(_emit_local_transform_kernel(
        model, "map", "input", "output", "status", "active", provider_plans=plans,
        consumer_qid="closed/map", capture_var="read_journal"))
    assert source.index("transform_readsA_ =") < source.index("pops::for_each_cell(")
    assert source.index("transform_reads_.initialize()") < source.index("if (transform_has_active_mask_")
    assert source.index("return;") < source.index("transform_reads_.activate()")
    assert source.index("CapturedParameterReads<") < source.index("params.get(0)")
    assert source.count("params.get(0)") == 1
    assert "CapturedProviderReads<" in source
    assert "const pops::Real coefficient =" not in source
    assert source.count("providers(index, 0)") == 1
    guarded = next(line for line in source.splitlines() if "providers(index, 0)" in line)
    assert "? ([&]()" in guarded and guarded.index("?") < guarded.index("providers(index, 0)")
    assert "if (!transform_reads_.valid()) transform_failed_ = pops::Real(1);" in source
    assert "std::function" not in source and "py::" not in source


def test_ordinary_transform_does_not_allocate_or_capture_unrequested_journal():
    model, _, _, plans, _ = prepared()
    source = "\n".join(_emit_local_transform_kernel(
        model, "map", "input", "output", "status", "active", provider_plans=plans,
        consumer_qid="closed/map"))
    assert "FieldReadCapture" not in source and "CapturedParameterReads" not in source
    assert "params.get(0)" in source


def test_coordinate_identity_is_from_the_same_kernel_input_before_mask_branch():
    model, _, _, plans, _ = prepared()
    source = "\n".join(_emit_local_transform_kernel(
        model, "map", "input", "output", "status", "active", provider_plans=plans,
        consumer_qid="closed/map", capture_var="free_reads",
        coordinate_capture_var="coordinate_reads"))
    assert source.index("transform_coordinateA_ =") < source.index("pops::for_each_cell(")
    read = "transform_coordinate_.record(0, inputA(index, 0));"
    assert source.count(read) == 1
    assert source.index(read) < source.index("if (transform_has_active_mask_")
    assert "if (!transform_coordinate_.valid()) transform_failed_ = pops::Real(1);" in source
    with pytest.raises(ValueError, match="same-evaluation"):
        _emit_local_transform_kernel(model, "map", "input", "output", "status", "active",
            provider_plans=plans, consumer_qid="closed/map", coordinate_capture_var="orphan")


@pytest.mark.parametrize("target", ("system", "amr_system"))
@pytest.mark.parametrize("issue", (False, True))
def test_closed_transform_uses_one_actual_kernel_before_owned_publication(target, issue):
    import pops
    from pops.time.value_signatures import value_signature
    from pops.codegen.program_emit_transform_obligation import emit_closed_transform

    model, _, _, plans, _ = prepared()
    typed_model = pops.Model("typed_source")
    state = typed_model.state("q", components=("q",))
    case = pops.Case("typed_source")
    block = case.block("gas", typed_model)
    q = pops.Program("typed_source").state(block[state])
    signature = value_signature(q.n)
    lines, owned = emit_closed_transform(
        model=model, name="map", source="(*raw_owner)", output="physical_output",
        status="status", mask="active", owner=0, signature=signature,
        block_indices={signature["block"]: 0}, provider_plans=plans, consumer_qid="closed/map",
        stem="closed_q", capacity=2**25, target=target, issue=issue,
        production_id=17 if issue else None)
    source = "\n".join(lines)
    assert owned == "closed_q_owner"
    assert source.count("pops::for_each_cell(") == 1
    assert "const auto& transform_input_ = (*raw_owner);" in source
    assert source.index("field_read_detail::phase(") < source.index("transform_type_owner_.emplace(")
    assert source.index("field_transform_support(") < source.index("CapturedTransformReads<")
    assert source.index("transform_capture_.close(") > source.index("pointwise_")
    assert source.index("ctx.field_provider_read_authority(") > source.index("prepare_provider_values(")
    assert source.index("ctx.field_provider_read_authority(") < source.index("transform_capture_.close(")
    assert "transform_authorities_[1] = transform_provider_authorities_[0]" in source
    assert source.index("transform_closed_.inputs.size()") < source.index("make_shared<const")
    call = "own_field_representation" if issue else "own_discharged_field_representation"
    assert source.index("transform_capture_.close(") < source.index("ctx." + call)
    other = "own_discharged_field_representation" if issue else "own_field_representation"
    assert "ctx." + other not in source
    assert "transform_capture_.inputs.journal" in source
    assert "transform_capture_.coordinates.journal" in source
    assert "newton" not in source and "evaluate_python" not in source


def test_closed_transform_requires_native_issuance_identity():
    from pops.codegen.program_emit_transform_obligation import emit_closed_transform
    import pytest
    with pytest.raises(ValueError, match="compiled producer SSA"):
        emit_closed_transform(model=None, name="q", source="q", output="u", status="s",
            mask="m", owner=0, signature={}, block_indices={}, provider_plans=None,
            consumer_qid="q", stem="issued", capacity=1024, target="system", issue=True)


def test_provider_point_dependencies_come_from_actual_equation_lowering():
    from pops.codegen._analytic_aux import emit_analytic_aux_launcher
    from tests.python.unit.codegen.test_analytic_aux_time import producer
    from pops.analytic import constant
    _, _, _, temporal = producer()
    source = emit_analytic_aux_launcher("temporal", temporal)
    assert "AuxiliaryPointReads::clock | AuxiliaryPointReads::physical_time" in source
    assert "context.point.require_physical_time(" in source
    _, _, _, stationary = producer(constant(3.))
    source = emit_analytic_aux_launcher("stationary", stationary)
    assert "AuxiliaryPointReads{0}" in source
    assert "context.point." not in source
