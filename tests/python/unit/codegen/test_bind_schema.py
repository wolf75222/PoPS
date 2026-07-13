"""ADC-654: qualified BindSchema is the artifact/bind parameter authority."""
from __future__ import annotations

import copy
import gc
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("pops")

from pops import model
from _typed_artifact_fixture import artifact_fixture
from pops.identity import make_identity
from pops.math import Bool, Integer, Real
from pops.params import (
    ConstParam,
    DerivedParam,
    ParamInvalidation,
    ParamPhase,
    ParamStorage,
    Positive,
    RuntimeParam,
)
from pops.model.bind_schema import BindSchema
from pops.problem import Case
from pops.runtime._bound_snapshot import BoundSnapshot, build_amr_snapshot


def _resolve(schema, values=None):
    return schema.resolve_bind(values or {}, compile_values=schema.resolve_compile())


def _two_instance_schema(*, default: float = 1.0):
    module = model.Module("transport")
    speed = module.param(RuntimeParam("speed", dtype=Real, default=default, domain=Positive()))
    order = module.param(ConstParam("order", 2, dtype=Integer))
    problem = Case(name="two-blocks")
    left = problem.block("left", module)
    right = problem.block("right", module)
    return BindSchema.from_problem(problem), left, right, speed, order


def test_same_module_in_two_blocks_has_distinct_qualified_slots_and_defaults():
    schema, left, right, speed, _ = _two_instance_schema(default=1.25)

    assert len(schema.slots) == 4
    assert len(schema.runtime_slots) == 2
    assert schema.runtime_slots[0].qid != schema.runtime_slots[1].qid
    assert schema.slot(left[speed]).handle.block_ref.local_id == "left"
    assert schema.slot(right[speed]).handle.block_ref.local_id == "right"

    resolved = _resolve(schema, {left[speed]: 2.0})
    assert resolved[schema.slot(left[speed]).handle] == 2.0
    assert resolved[schema.slot(right[speed]).handle] == 1.25
    assert resolved.schema is schema
    assert resolved.source(schema.slot(left[speed]).handle) == "override"
    assert resolved.source(schema.slot(right[speed]).handle) == "default"
    assert [row["qid"] for row in resolved.rows()] == [slot.qid for slot in schema.slots]
    with pytest.raises(TypeError):
        resolved.values[schema.slot(left[speed]).handle] = 9.0


def test_two_instances_route_distinct_values_to_native_block_vectors():
    from pops.runtime._install_param_routing import route_block_params

    schema, left, right, speed, _ = _two_instance_schema(default=1.25)
    resolved = _resolve(schema, {left[speed]: 2.0, right[speed]: 3.0})
    carriers = {
        "left": SimpleNamespace(runtime_param_names=("speed",)),
        "right": SimpleNamespace(runtime_param_names=("speed",)),
    }

    assert route_block_params(carriers, schema, resolved) == {
        "left": [2.0],
        "right": [3.0],
    }


def test_native_real_carrier_refuses_lossy_integer_lowering():
    from pops.runtime._install_param_routing import route_block_params

    module = model.Module("integer-carrier")
    count = module.param(RuntimeParam("count", dtype=Integer, default=1))
    problem = Case(name="integer-carrier-case")
    block = problem.block("fluid", module)
    schema = BindSchema.from_problem(problem)
    resolved = _resolve(schema, {block[count]: 2**53 + 1})
    carrier = {"fluid": SimpleNamespace(runtime_param_names=("count",))}

    with pytest.raises(ValueError, match="not exactly representable"):
        route_block_params(carrier, schema, resolved)


def test_schema_extraction_from_an_already_frozen_problem_is_read_only():
    module = model.Module("frozen-schema")
    speed = module.param(RuntimeParam("speed", default=1.0))
    problem = Case(name="frozen-schema-case")
    block = problem.block("fluid", module)

    problem.freeze()
    schema = BindSchema.from_problem(problem)

    assert schema.slot(block[speed]).handle.block_ref.local_id == "fluid"
    assert _resolve(schema)[schema.slot(block[speed]).handle] == 1.0


def test_schema_alias_authentication_does_not_retain_live_authoring_graph():
    def build():
        module = model.Module("detached-schema")
        speed = module.param(RuntimeParam("speed", default=1.0))
        problem = Case(name="detached-schema-case")
        block = problem.block("fluid", module)
        live_alias = block[speed]
        schema = BindSchema.from_problem(problem)
        assert schema.slot(live_alias).handle.is_resolved
        assert all(isinstance(key, str) for key in schema._aliases)
        return schema, weakref.ref(problem), weakref.ref(module)

    schema, problem_ref, module_ref = build()
    gc.collect()

    assert schema.hash
    assert problem_ref() is None
    assert module_ref() is None


def test_raw_module_cannot_add_parameters_after_problem_freeze():
    module = model.Module("frozen-raw-module")
    module.param(RuntimeParam("speed", default=1.0))
    problem = Case(name="frozen-raw-module-case").block("fluid", physics=module)

    snapshot = problem.freeze()
    with pytest.raises(RuntimeError, match="frozen.*parameter"):
        module.param(RuntimeParam("late", default=2.0))
    with pytest.raises(RuntimeError, match="frozen.*parameter"):
        module.param_registry().register(RuntimeParam("also_late", default=3.0))

    assert tuple(module.params()) == ("speed",)
    assert problem.freeze() is snapshot


def test_case_and_block_parameters_with_same_local_name_never_merge():
    module = model.Module("transport-case-scope")
    local = module.param(RuntimeParam("threshold", default=1.0))
    problem = Case(name="case-scope")
    block = problem.block("fluid", module)
    case = problem.param(RuntimeParam("threshold", default=2.0))

    schema = BindSchema.from_problem(problem)
    assert len(schema.runtime_slots) == 2
    assert schema.slot(block[local]).qid != schema.slot(case).qid
    assert schema.slot(block[local]).handle.is_instance
    assert not schema.slot(case).handle.is_instance

    resolved = _resolve(schema, {block[local]: 3.0, case: 4.0})
    assert resolved[schema.slot(block[local]).handle] == 3.0
    assert resolved[schema.slot(case).handle] == 4.0


def test_schema_roundtrip_is_strict_and_hashes_have_separate_lifetimes():
    first, left, right, speed, _ = _two_instance_schema(default=1.0)
    second, _, _, _, _ = _two_instance_schema(default=2.0)

    rebuilt = BindSchema.from_json(first.to_json())
    assert rebuilt.to_dict() == first.to_dict()
    assert rebuilt.hash == first.hash
    assert rebuilt.slot(left[speed]).qid == first.slot(left[speed]).qid
    assert rebuilt.slot(right[speed]).qid == first.slot(right[speed]).qid
    without_aliases = BindSchema(first.slots)
    assert without_aliases.hash != first.hash
    assert first.hash != second.hash
    assert first.artifact_hash == second.artifact_hash

    unknown = copy.deepcopy(first.to_dict())
    unknown["unknown"] = True
    with pytest.raises(TypeError, match="unknown"):
        BindSchema.from_dict(unknown)
    bad_qid = copy.deepcopy(first.to_dict())
    bad_qid["payload"]["slots"][0]["qid"] += "-forged"
    with pytest.raises(ValueError, match="qid"):
        BindSchema.from_dict(bad_qid)
    bad_ordinal = copy.deepcopy(first.to_dict())
    bad_ordinal["payload"]["slots"][0]["ordinal"] = 2
    with pytest.raises(ValueError, match="ordinal"):
        BindSchema.from_dict(bad_ordinal)
    noncanonical_default = copy.deepcopy(first.to_dict())
    noncanonical_default["payload"]["slots"][0]["declaration"]["default"]["value"]["target"] = "Integer"
    with pytest.raises(ValueError, match="not in canonical form"):
        BindSchema.from_dict(noncanonical_default)

    forged_alias = copy.deepcopy(first.to_dict())
    alias_qid = next(iter(forged_alias["payload"]["aliases"]))
    forged_alias["payload"]["aliases"][alias_qid] = "parameter:missing"
    with pytest.raises(ValueError, match="unknown parameter slot"):
        BindSchema.from_dict(forged_alias)

    duplicate = first.to_json().replace(
        '"protocol":"pops.manifest",',
        '"protocol":"pops.manifest","protocol":"forged",',
        1,
    )
    with pytest.raises(ValueError, match="duplicate object key"):
        BindSchema.from_json(duplicate)


def test_bind_mapping_requires_handles_and_validates_kind_dtype_domain_and_requiredness():
    module = model.Module("typed")
    required = module.param(RuntimeParam("required", dtype=Integer))
    enabled = module.param(RuntimeParam("enabled", dtype=Bool, default=True))
    positive = module.param(RuntimeParam("positive", dtype=Real, default=1.0, domain=Positive()))
    order = module.param(ConstParam("order", 2, dtype=Integer))
    problem = Case(name="typed-case")
    block = problem.block("fluid", module)
    schema = BindSchema.from_problem(problem)

    with pytest.raises(ValueError, match="missing required"):
        _resolve(schema)
    with pytest.raises(TypeError, match="ParamHandle"):
        _resolve(schema, {"required": 2})
    with pytest.raises(ValueError, match="block-qualified"):
        _resolve(schema, {required: 2})
    with pytest.raises(TypeError, match="requires an int"):
        _resolve(schema, {block[required]: 2.5})
    with pytest.raises(TypeError, match="requires a bool"):
        _resolve(schema, {block[required]: 2, block[enabled]: 1})
    with pytest.raises(ValueError, match="outside domain"):
        _resolve(schema, {block[required]: 2, block[positive]: -1.0})
    with pytest.raises(TypeError, match="only RuntimeParam"):
        _resolve(schema, {block[required]: 2, block[order]: 4})

    resolved = _resolve(schema, {block[required]: 3})
    assert resolved[schema.slot(block[required]).handle] == 3
    assert resolved[schema.slot(block[enabled]).handle] is True
    assert resolved[schema.slot(block[positive]).handle] == 1.0
    assert resolved[schema.slot(block[order]).handle] == 2
    assert resolved.source(schema.slot(block[order]).handle) == "const"


def _bind_derived_module(*, phase=ParamPhase.Bind, invalidation=ParamInvalidation.OnDependencies):
    module = model.Module("derived")
    alpha = module.param(RuntimeParam("alpha", default=2.0))
    beta = module.param(DerivedParam(
        "beta",
        module.value(alpha) * 2,
        depends_on=(alpha,),
        phase=phase,
        storage=ParamStorage.DerivedCache,
        invalidation=invalidation,
    ))
    gamma = module.param(DerivedParam(
        "gamma",
        module.value(beta) + 1,
        depends_on=(beta,),
        phase=phase,
        storage=ParamStorage.DerivedCache,
        invalidation=invalidation,
    ))
    return module, alpha, beta, gamma


def test_bind_derived_cache_is_topological_and_cannot_be_overridden():
    module, alpha, beta, gamma = _bind_derived_module()
    problem = Case(name="derived-case")
    block = problem.block("fluid", module)
    schema = BindSchema.from_problem(problem)

    resolved = _resolve(schema, {block[alpha]: 3.0})
    assert resolved[schema.slot(block[beta]).handle] == 6.0
    assert resolved[schema.slot(block[gamma]).handle] == 7.0
    assert resolved.source(schema.slot(block[beta]).handle) == "derived"
    with pytest.raises(TypeError, match="only RuntimeParam"):
        _resolve(schema, {block[beta]: 9.0})


def test_compile_inline_derived_is_materialized_for_python_consumers():
    module = model.Module("compile-derived")
    scale = module.param(ConstParam("scale", 2, dtype=Integer))
    doubled = module.param(DerivedParam(
        "doubled",
        module.value(scale) * 2,
        depends_on=(scale,),
        phase=ParamPhase.Compile,
        storage=ParamStorage.Inline,
        invalidation=ParamInvalidation.Never,
        dtype=Integer,
        domain=Positive(),
    ))
    problem = Case(name="compile-derived-case")
    block = problem.block("fluid", module)
    schema = BindSchema.from_problem(problem)

    compile_values = schema.resolve_compile()
    resolved = schema.resolve_bind({}, compile_values=compile_values)
    assert resolved[schema.slot(block[scale]).handle] == 2
    assert resolved[schema.slot(block[doubled]).handle] == 4


def test_derived_foreign_cycle_phase_and_invalidation_fail_loudly():
    module, _, _, _ = _bind_derived_module()
    problem = Case(name="derived-invalid")
    problem.block("fluid", module)
    schema = BindSchema.from_problem(problem)

    foreign = copy.deepcopy(schema.to_dict())
    foreign["payload"]["slots"][1]["declaration"]["depends_on"] = [
        {"name": "foreign", "param_kind": "runtime"}
    ]
    with pytest.raises(ValueError, match="cannot resolve dependency"):
        BindSchema.from_dict(foreign)

    cycle = copy.deepcopy(schema.to_dict())
    cycle["payload"]["slots"][1]["declaration"]["depends_on"] = [
        {"name": "gamma", "param_kind": "derived"}
    ]
    cycle["payload"]["slots"][1]["declaration"]["expression"]["value"] = ["rparam", "gamma"]
    with pytest.raises(ValueError, match="cycle"):
        BindSchema.from_dict(cycle)

    undeclared_read = copy.deepcopy(schema.to_dict())
    undeclared_read["payload"]["slots"][1]["declaration"]["expression"]["value"] = [
        "rparam", "foreign"
    ]
    with pytest.raises(ValueError, match="undeclared dependency"):
        BindSchema.from_dict(undeclared_read)

    late_dependency = copy.deepcopy(schema.to_dict())
    beta = late_dependency["payload"]["slots"][1]["declaration"]
    beta["phase"] = "compile"
    beta["storage"] = "inline"
    beta["invalidation"] = "never"
    with pytest.raises(ValueError, match="cannot depend on runtime"):
        BindSchema.from_dict(late_dependency)

    late = model.Module("late-derived")
    late_alpha = late.param(RuntimeParam("alpha", default=2.0))
    late.param(DerivedParam(
        "beta",
        late.value(late_alpha) * 2,
        depends_on=(late_alpha,),
        phase=ParamPhase.PerBlock,
        storage=ParamStorage.DerivedCache,
        invalidation=ParamInvalidation.OnDependencies,
    ))
    late_problem = Case(name="late")
    late_problem.block("fluid", late)
    with pytest.raises(NotImplementedError, match="no execution provider"):
        BindSchema.from_problem(late_problem)

    with pytest.raises(ValueError, match="invalidation"):
        _bind_derived_module(invalidation=ParamInvalidation.Never)


def test_compiled_arguments_and_manifest_are_derived_from_attached_schema():
    schema, _, _, _, _ = _two_instance_schema(default=1.5)
    compiled = artifact_fixture(
        target="amr_system", block_names=("left", "right"), bind_schema=schema,
    )

    arguments = compiled.arguments()
    assert set(arguments.params) == {slot.qid for slot in schema.slots}
    assert all(row["handle"]["handle_type"] == "parameter" for row in arguments.params.values())
    assert [row["required"] for row in arguments.params.values()].count(False) == 4

    manifest = compiled.manifest()
    assert manifest.to_dict()["payload"]["bind_schema"] == schema.to_dict()
    assert manifest.bind_schema_hash == schema.hash
    assert manifest.bind_schema_artifact_hash == schema.artifact_hash
    assert manifest.params_runtime == tuple(sorted(slot.qid for slot in schema.runtime_slots))
    assert manifest.params_const == tuple(sorted(slot.qid for slot in schema.const_slots))
    with pytest.raises(TypeError):
        manifest.bind_schema["payload"]["slots"][0]["declaration"]["default"]["value"] = 7.0
    assert type(manifest).from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    forged_hash = copy.deepcopy(manifest.to_dict())
    forged_hash["payload"]["bind_schema_hash"] = "0" * 64
    with pytest.raises(ValueError, match="bind_schema_hash"):
        type(manifest).from_dict(forged_hash)
    forged_summary = copy.deepcopy(manifest.to_dict())
    forged_summary["payload"]["params_runtime"] = []
    with pytest.raises(ValueError, match="not canonical"):
        type(manifest).from_dict(forged_summary)


def test_bound_snapshot_records_effective_values_sources_and_schema_identity():
    schema, left, _, speed, _ = _two_instance_schema(default=1.5)

    def snapshot(value):
        resolved = _resolve(schema, {left[speed]: value})
        return BoundSnapshot(
            semantic_identity=make_identity("semantic", {"problem": "bind-schema"}),
            artifact_identity=make_identity("artifact", {"binary": "bind-schema"}),
            layout={"kind": "uniform"}, blocks=[], solvers={},
            step_transaction=None,
            params=resolved.rows(),
            aux_evidence={}, initial_evidence={},
            bind_schema_identity=make_identity("bind-schema", schema.to_dict()),
        )

    first = snapshot(2.0)
    second = snapshot(3.0)
    assert first.bind_identity != second.bind_identity
    payload = first.to_dict()
    assert payload["bind_schema_identity"]["hexdigest"] == make_identity(
        "bind-schema", schema.to_dict()).hexdigest
    rows = {row["qid"]: row for row in payload["params"]}
    assert rows[schema.slot(left[speed]).qid]["source"] == "override"
    assert rows[schema.slot(left[speed]).qid]["value"]["kind"] == "binary64"
    with pytest.raises(AttributeError, match="immutable"):
        first.params = ()


def test_amr_bound_snapshot_retains_installed_program_identity_and_bindings():
    schema, left, _, speed, _ = _two_instance_schema(default=1.5)
    resolved = _resolve(schema, {left[speed]: 2.0})
    compiled = SimpleNamespace(
        semantic_identity=make_identity("semantic", {"problem": "amr-bind"}),
        artifact_identity=make_identity("artifact", {"binary": "amr-bind"}),
    )
    engine = SimpleNamespace(_output_policies=[], _diagnostic_measures=[])

    snapshot = build_amr_snapshot(
        engine, compiled, {}, {}, {}, resolved
    ).to_dict()
    assert snapshot["semantic_identity"]["domain"] == "semantic"
    assert snapshot["artifact_identity"]["domain"] == "artifact"
    assert snapshot["step_transaction"] is None
    assert snapshot["bind_schema_identity"]["domain"] == "bind-schema"
    assert len(snapshot["params"]) == len(schema.slots)
