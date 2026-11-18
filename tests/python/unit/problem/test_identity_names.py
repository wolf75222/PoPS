"""N11 lot 2b (spec 0.3 CC2, C47, G07 criterion (ii)): owner names leave the identities.

The owner names (the case, its blocks, the model definitions, the layouts) are presentation names.
A two-block, two-model case renamed so that the alphabetical order of its blocks and of its models
reverses keeps its semantic identity, ``snapshot.artifact_hash`` and ``plan_identity``; the full
authored snapshot keeps the names. Every other name is kept in the identity while gap C52 is open:
renaming a state, a term or a realization (review R1) changes the identities, which proves the
projection does not erase a kept name. The block registration order is the order of the native
block indices, so swapping it changes the identities too.
"""

import json

import pytest

import pops
from pops.identity.presentation import (
    PresentationMap, PresentationProjectionError, ReservedOwnerNameError,
)
from pops.problem._snapshot import AuthoringSnapshot
from tests.python.support.presentation_case import FIRST, RENAMED, build_named_case, resolve_named_case


def _identities(resolved):
    snapshot = resolved.snapshot
    return {
        "semantic_identity": snapshot.semantic_identity.token,
        "artifact_hash": snapshot.artifact_hash,
        "plan_identity": resolved.plan_identity.token,
    }


@pytest.fixture(scope="module")
def first():
    return resolve_named_case(FIRST)


def test_renamed_owners_keep_every_identity(first):
    renamed = resolve_named_case(RENAMED)

    assert _identities(renamed) == _identities(first)
    # The names are presentation: the full authored snapshot keeps each of them.
    assert renamed.snapshot.hash != first.snapshot.hash
    authored = json.dumps(renamed.snapshot.to_dict())
    for name in (RENAMED["case"], *RENAMED["blocks"], *RENAMED["models"]):
        assert name in authored


def test_parameterized_models_keep_identity_under_owner_rename():
    literal = "case:advection"
    first = resolve_named_case(FIRST, speed_parameter=literal)
    renamed = resolve_named_case(RENAMED, speed_parameter=literal)
    changed = resolve_named_case(FIRST, speed_parameter=literal, speeds=(1, 3))
    assert _identities(first) == _identities(renamed)
    assert _identities(first)["artifact_hash"] != _identities(changed)["artifact_hash"]
    assert first.plan_identity != changed.plan_identity
    assert first.bind_schema.artifact_hash != renamed.bind_schema.artifact_hash
    for plan in (first, renamed):
        plan.verify()
        assert [slot.handle.local_id for slot in plan.bind_schema] == [literal, literal]


def test_case_parameter_expression_projects_only_authenticated_handle_leaves():
    from pops.params import DerivedParam, ParamInvalidation, ParamPhase, ParamStorage, RuntimeParam

    literal = "case:advection"

    def resolve(names, *, factor=2, default=2.0):
        case, layout = build_named_case(names)
        alpha = case.param(RuntimeParam(literal, default=default))
        case.param(DerivedParam(
            "beta", case.value(alpha) * factor, depends_on=(alpha,), phase=ParamPhase.Bind,
            storage=ParamStorage.DerivedCache, invalidation=ParamInvalidation.OnDependencies))
        return pops.resolve(pops.validate(case), layout=layout)

    first = resolve(FIRST)
    renamed = resolve(RENAMED)
    changed_default = resolve(FIRST, default=4.0)
    changed_expression = resolve(FIRST, factor=3)
    assert _identities(first) == _identities(renamed) == _identities(changed_default)
    assert first.bind_schema.hash != changed_default.bind_schema.hash
    assert first.bind_schema.artifact_hash == changed_default.bind_schema.artifact_hash
    assert first.bind_schema.artifact_hash != renamed.bind_schema.artifact_hash
    assert first.snapshot.hash != renamed.snapshot.hash
    assert first.snapshot.artifact_hash != changed_expression.snapshot.artifact_hash
    assert first.plan_identity != changed_expression.plan_identity
    for plan in (first, renamed):
        plan.verify()
        params = plan.snapshot.artifact_to_dict()["payload"]["params"]
        assert params["beta"]["depends_on"] == [{"name": literal, "param_kind": "runtime"}]
        assert params[literal]["name"] == literal
        assert params["beta"]["expression"]["value"][1][0] == [
            "handle_value", params[literal]["qid"]]
        assert "#authoring=" not in params[literal]["qid"]
        values = plan.bind_schema.resolve_bind({}, compile_values=plan.bind_schema.resolve_compile())
        assert values[plan.bind_schema.slots[1].handle] == 4.


@pytest.mark.parametrize("block_name", (
    "blocks", "payload", "[]", "accepted Program exchange mailbox", "left/right:100%",
))
def test_block_names_may_match_schema_keys_and_continuation_labels(first, block_name):
    renamed = resolve_named_case(dict(FIRST, blocks=("alpha", block_name)))
    renamed.verify()
    assert _identities(renamed) == _identities(first)
    manifest_blocks = renamed.snapshot.artifact_to_dict()["payload"]["blocks"]
    assert [row["model"]["manifest"]["name"] for row in manifest_blocks.values()] == [
        "$model0", "$model1",
    ]
    rows = renamed._payload()["continuation_transitions"]["objects"]
    mailbox = next(row for row in rows if row["kind"] == "accepted_exchanges")
    assert mailbox["name"] == "accepted Program exchange mailbox"


@pytest.mark.parametrize("name", (
    "case:user", "case:advection", "pops.handle.v1::case:advection::state::U",
))
def test_literal_quantity_names_are_opaque_to_owner_projection(first, name):
    named = resolve_named_case(FIRST, state_name=name)
    renamed = resolve_named_case(RENAMED, state_name=name)
    assert _identities(named) == _identities(renamed)
    for key, identity in _identities(first).items():
        assert _identities(named)[key] != identity, key
    manifest_blocks = named.snapshot.artifact_to_dict()["payload"]["blocks"]
    for row in manifest_blocks.values():
        assert name in row["model"]["manifest"]["state_spaces"]


def test_retained_auxiliary_keeps_owner_renames_and_distinguishes_its_expression():
    first = resolve_named_case(FIRST, auxiliary_offset=2)
    renamed = resolve_named_case(RENAMED, auxiliary_offset=2)
    changed = resolve_named_case(FIRST, auxiliary_offset=3)
    renamed.verify()
    assert _identities(first) == _identities(renamed)
    for key, identity in _identities(first).items():
        assert _identities(changed)[key] != identity, key

    def retained(plan, *, projected):
        data = plan._payload() if projected else plan._named_payload()
        return {row["identity"] for row in data["continuation_transitions"]["objects"]
                if row["kind"] == "auxiliary"}

    assert len(retained(first, projected=True)) == 2
    assert retained(first, projected=True) == retained(renamed, projected=True)
    assert retained(first, projected=True).isdisjoint(retained(changed, projected=True))
    assert retained(first, projected=False).isdisjoint(retained(renamed, projected=False))


def test_explicit_layout_names_are_presentation():
    from pops.mesh import LayoutPlanBuilder

    def explicit_layouts(names, layout_names):
        case, layout = build_named_case(names)
        validated = pops.validate(case)
        subjects = validated.layout_subjects()
        builder = LayoutPlanBuilder(validated.owner_path.canonical())
        handles = dict(zip(names["blocks"], (builder.layout(name, layout) for name in layout_names),
                           strict=True))
        for block in subjects.blocks:
            builder.assign_block(block, handles[block.local_id])
        for state in subjects.states:
            builder.assign_state(state, handles[state.block_ref.local_id])
        plan = builder.resolve(**subjects.to_dict())
        return pops.resolve(validated, layout=plan, layout_providers=dict.fromkeys(
            handles.values(), layout))

    first = explicit_layouts(FIRST, ("one", "two"))
    layout_only = explicit_layouts(FIRST, ("three", "zero"))
    all_owners = explicit_layouts(RENAMED, ("Z/~?", "A:payload"))
    assert _identities(first) == _identities(layout_only) == _identities(all_owners)


def _resolve_m14(case_name, *, block_name="fluid", layout_name=None, inlet=None, brick_name=None):
    from types import FunctionType
    from examples.api_v02 import models_M14
    from pops.mesh import LayoutPlanBuilder

    class ModuleWithOverride:
        def __init__(self, module, **overrides):
            self.module, self.overrides = module, overrides

        def __getattr__(self, name):
            return self.overrides[name] if name in self.overrides else getattr(self.module, name)

    def make_case(_):
        case = pops.Case(case_name)
        block = case.block
        case.block = lambda name, model, **options: block(block_name, model, **options)
        return case

    base = models_M14.build_m14
    namespace = dict(base.__globals__)
    namespace["pops"] = ModuleWithOverride(pops, Case=make_case)
    if inlet is not None:
        namespace["RESERVOIR"] = inlet
    if brick_name is not None:
        riemann = namespace["riemann"]
        namespace["riemann"] = ModuleWithOverride(
            riemann, User=lambda **options: riemann.User(name=brick_name, **options))
    build = FunctionType(base.__code__, namespace, base.__name__, base.__defaults__, base.__closure__)
    build.__kwdefaults__ = base.__kwdefaults__
    spec = build(cells=(4, 1))
    case = pops.validate(spec.case)
    if layout_name is None:
        return pops.resolve(case, layout=spec.layout)
    subjects = case.layout_subjects()
    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layout = builder.layout(layout_name, spec.layout)
    for subject in subjects.blocks:
        builder.assign_block(subject, layout)
    for subject in subjects.states:
        builder.assign_state(subject, layout)
    for subject in subjects.fields:
        builder.assign_field(subject, layout)
    return pops.resolve(case, layout=builder.resolve(**subjects.to_dict()),
                        layout_providers={layout: spec.layout})


def test_m14_case_rename_keeps_derived_identities_in_parallel(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from pops.boundary.transport import ResolvedTransportCondition
    from pops.problem._snapshot_presentation import ghost_presentation_map

    nested = []
    original = ResolvedTransportCondition.presentation_handles

    def resolve_inside_projection(condition, names):
        if names.case == "sheath" and not nested:
            nested.append(_resolve_m14("nested_sheath"))
        return original(condition, names)

    monkeypatch.setattr(ResolvedTransportCondition, "presentation_handles", resolve_inside_projection)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, renamed = pool.map(_resolve_m14, ("sheath", "renamed_sheath"))
    assert _identities(first) == _identities(renamed) == _identities(nested[0])
    for plan in (renamed, first):
        plan.verify()
        ghost, = plan.blocks[0].numerics.boundaries
        producers = [row.producer for row in ghost.execution_order()]
        ordered = [producer.handle for producer in producers]
        assert any(producer.dependencies for producer in producers)
        for producer in producers:
            for dependency in producer.dependencies:
                if dependency in ordered:
                    assert ordered.index(dependency) < ordered.index(producer.handle)

    def named_authorities(plan):
        ghost, = plan.blocks[0].numerics.boundaries
        field, = plan.field_plans.values()
        return ghost.coverage.layout_manifest.qualified_id, field.native_install_data()["provider_slot"]

    first_ids, second_ids = named_authorities(first), named_authorities(renamed)
    assert all(left != right for left, right in zip(first_ids, second_ids, strict=True))
    block = first.blocks[0]
    ghost, = block.numerics.boundaries
    foreign, = renamed.blocks[0].numerics.boundaries
    origins = ghost._presentation_origins
    with pytest.raises(ValueError, match="composition evidence"):
        replace(origins.handles[0], evidence={}).named_evidence()
    altered = replace(ghost, _presentation_origins=replace(
        origins, handles=(foreign._presentation_origins.handles[0],)))
    altered_block = replace(block, numerics=replace(block.numerics, boundaries=(altered,)))
    with pytest.raises(ValueError, match="another boundary owner"):
        ghost_presentation_map(first.snapshot.presentation, (altered_block,), first.layout_plan)


def test_m14_tracks_layout_and_block_roles_and_keeps_boundary_data_and_brick_names():
    first = _resolve_m14("sheath", layout_name="old_layout")
    layout_renamed = _resolve_m14("sheath", layout_name="another/layout")
    block_renamed = _resolve_m14("sheath", block_name="new_fluid", layout_name="old_layout")
    assert _identities(first) == _identities(layout_renamed) == _identities(block_renamed)
    changed_inlet = _resolve_m14("sheath", layout_name="old_layout", inlet=(1.2, 2., 1., 3.))
    ghost, = first.blocks[0].numerics.boundaries
    literal = ghost.coverage.layout_manifest.qualified_id
    changed_brick = _resolve_m14("sheath", layout_name="old_layout", brick_name=literal)
    for changed in (changed_inlet, changed_brick):
        for key, identity in _identities(first).items():
            assert _identities(changed)[key] != identity, key
    assert literal in set(_strings(changed_brick.snapshot.artifact_to_dict()))


def test_field_provider_identity_keeps_the_order_of_two_rhs_contributions():
    from pops.fields._identity import field_identity
    from tests.python.unit.codegen.test_uniform_field_roles import uniform_multiphysics

    plan = uniform_multiphysics.__wrapped__()
    named, = plan.field_plans.values()
    projected, = plan._payload()["field_plans"].values()
    options = projected["value"]["native_options"]
    named_rows = named.native_install_data()["provider_identity"]["contributions"]
    evidence = options["provider_identity"]
    projected_rows = evidence["contributions"]
    assert len(named_rows) == len(projected_rows) == 2
    for before, after in zip(named_rows, projected_rows, strict=True):
        assert after["owner_block"] == plan.snapshot.presentation.block_label(before["owner_block"])
        assert after["coefficient"] == {"binary64": before["coefficient"].hex()}
    assert options["provider_slot"] == field_identity("qualified-field-provider", evidence).token
    assert options["provider_slot"] != field_identity("qualified-field-provider", {
        **evidence, "contributions": projected_rows[::-1],
    }).token


def _owner_names(text):
    """The owner names a string spells in a qualified id or an owner path, as ``kind:name``."""
    found = set()
    for segment in text.replace("::", "/").split("/"):
        kind, _, name = segment.partition(":")
        if kind in ("case", "block", "model_definition", "layout") and name:
            found.add("%s:%s" % (kind, name.partition("@")[0]))
    return found


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_identity_preimages_spell_no_owner_name(first):
    names = {FIRST["case"], *FIRST["blocks"], *FIRST["models"]}
    for preimage in (first.snapshot.semantic_to_dict(), first.snapshot.artifact_to_dict()):
        strings = list(_strings(preimage))
        assert not names.intersection(strings)
        spelled = set().union(*(_owner_names(text) for text in strings))
        assert {item.partition(":")[2] for item in spelled} <= {
            "%24case", "%24block0", "%24block1", "%24model0", "%24model1", "%24layout0",
            "$case", "$block0", "$block1", "$model0", "$model1", "$layout0"}, spelled


@pytest.mark.parametrize("options", [
    {"state_name": "V"},
    {"rate_name": "other_balance"},
], ids=["state", "term"])
def test_a_kept_name_changes_every_identity(first, options):
    changed = _identities(resolve_named_case(FIRST, **options))
    for key, value in _identities(first).items():
        assert changed[key] != value, key


def test_block_registration_order_is_identity(first):
    # The same two blocks (each name with its own content) registered in the other order.
    swapped = {"case": FIRST["case"], "blocks": FIRST["blocks"][::-1],
               "models": FIRST["models"][::-1]}
    assert _identities(resolve_named_case(swapped, speeds=(2, 1))) != _identities(first)


def test_two_models_with_one_name_stay_two_models(first):
    homonyms = {"case": "pair", "blocks": ("a", "b"), "models": ("m", "m")}
    assert _identities(resolve_named_case(homonyms)) == _identities(first)


def test_realization_names_are_kept_under_c52():
    """Review R1: the name= of reconstruction.User and a @pops.scheme function name are kept."""
    from pops import math as pm
    from pops.numerics import reconstruction
    from tests.python.unit.surface.test_su_t3_schemes_2d import _hand_case, _llf_face, _resolved

    def user(name):
        def make(u):
            def body(sample):
                minus, centre, plus = sample(-1)[0], sample(0)[0], sample(1)[0]
                slope = pm.where((centre - minus) * (plus - centre) > 0,
                                 lambda: (plus - minus) / 4, lambda: 0)
                return (centre + slope,)

            return reconstruction.User(body, state=u, formal_order=2, name=name)
        return make

    muscl = _identities(_resolved(*_hand_case(_llf_face, recon=user("muscl"))))
    other = _identities(_resolved(*_hand_case(_llf_face, recon=user("my_reconstruction"))))
    for key, value in muscl.items():
        assert other[key] != value, key


_FP = "pops.module:sha256:" + "0" * 64


def test_the_map_refuses_a_name_it_does_not_know():
    names = PresentationMap(case="c", blocks=("a",), models=(("m", _FP, _FP),),
                            block_models=(0,), layouts=("grid",))
    assert names.project_string("pops.handle.v1::case:c::block::a") \
        == "pops.handle.v1::case:%24case::block::%24block0"
    assert names.project_string("pops.handle.v1::case:c::layout::grid") \
        == "pops.handle.v1::case:%24case::layout::%24layout0"
    # A label suffix after the quoted local id stays (an AMR lowering coverage row).
    assert names.project_string("pops.handle.v1::case:c::layout::grid:amr-bootstrap:x") \
        == "pops.handle.v1::case:%24case::layout::%24layout0:amr-bootstrap:x"
    # A truly foreign name of a projected kind is refused, never passed through.
    for foreign in ("pops.handle.v1::case:c::block::b",
                    "pops.handle.v1::case:other::block::a",
                    "pops.handle.v1::case:c::layout::other",
                    "pops.handle.v1::model_definition:other@%s::state::U" % _FP,
                    "pops.handle.v1::model_definition:m@pops.module%%3Asha256%%3A%s::state::U"
                    % ("1" * 64)):
        with pytest.raises(PresentationProjectionError):
            names.project_string(foreign)
    with pytest.raises(PresentationProjectionError):
        names.project({"schema_version": 2, "definition_fingerprint": None,
                       "nodes": [{"kind": "case", "name": "other"}]})
    # A name of a kept kind passes unchanged.
    assert names.project_string("pops.handle.v1::case:c/descriptor:a::field::a") \
        == "pops.handle.v1::case:%24case/descriptor:a::field::a"


def test_an_owner_name_cannot_take_the_form_of_a_label():
    """Review A2 of #797: a name kept in a preimage can never equal a generated label.

    Without the refusal, ``pops.Case("$block0")`` was accepted, and its kept ``block:$block0``
    spelled the label of another case's first block: two preimages could collide.
    """
    from pops.mesh import LayoutPlanBuilder

    with pytest.raises(ReservedOwnerNameError, match="case name '\\$block0' is reserved"):
        pops.Case("$block0")
    with pytest.raises(ReservedOwnerNameError, match="model definition name"):
        pops.Model("$model0")
    case = pops.Case("pair")
    with pytest.raises(ReservedOwnerNameError, match="block name"):
        case.block("$block0", pops.Model("m"))
    with pytest.raises(ReservedOwnerNameError, match="layout name"):
        LayoutPlanBuilder(case.owner_path.canonical()).layout("$layout0", None)
    # Every label the projection writes starts with ``$``, every owner name it keeps does not.
    first = resolve_named_case(FIRST)
    strings = list(_strings(first.snapshot.semantic_to_dict()))
    assert any(text.startswith("$block") or "%24block" in text for text in strings)
    assert all(name[:1] != "$" for name in (*FIRST["blocks"], *FIRST["models"], FIRST["case"]))


def test_k1_a_layout_captured_before_resolve_keeps_its_name_until_resolve():
    validated = PresentationMap(case="c", blocks=("a",), models=(("m", _FP, _FP),),
                                block_models=(0,), layouts=("default",), layouts_open=True)
    assert validated.project_string("pops.handle.v1::case:c::layout::amr") \
        == "pops.handle.v1::case:%24case::layout::amr"
    # On the public route: the Program captures a LayoutPlanBuilder layout before resolve.
    from examples.api_v02.continuation_regrid_auxiliary import build_continuation_regrid_auxiliary

    spec, providers = build_continuation_regrid_auxiliary()
    validated_case = pops.validate(spec.case)
    assert "::layout::amr" in json.dumps(validated_case.snapshot.semantic_to_dict())
    resolved = pops.resolve(validated_case, layout=spec.layout, layout_providers=providers)
    assert "::layout::amr" not in json.dumps(resolved.snapshot.semantic_to_dict())


def test_k2_an_authoring_path_outside_the_block_models_keeps_its_name():
    # A runtime capture of a model parameter names its model by an authoring path, without a
    # definition fingerprint (pops.model.ownership.OwnerPath.__str__).
    names = PresentationMap(case="c", blocks=("a",), models=(("m", _FP, _FP),), block_models=(0,))
    assert names.project_string("pops.handle.v1::model_definition:m#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:%24model0#authoring=3::parameter::k"
    # A model outside the case: validation refuses it with its own reason.
    assert names.project_string("pops.handle.v1::model_definition:other#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:other#authoring=3::parameter::k"
    # Two models of one name are told apart only by their fingerprint.
    other = "pops.module:sha256:" + "1" * 64
    homonyms = PresentationMap(case="c", blocks=("a", "b"),
                               models=(("m", _FP, _FP), ("m", other, other)),
                               block_models=(0, 1))
    assert homonyms.project_string("pops.handle.v1::model_definition:m#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:m#authoring=3::parameter::k"


def test_k3_a_model_read_by_the_program_without_a_block_keeps_its_name():
    registry = "pops.operator-registry:sha256:" + "2" * 64
    names = PresentationMap(case="c", blocks=(), models=(), block_models=())
    record = {"schema_version": 2, "definition_fingerprint": registry,
              "nodes": [{"kind": "model_definition", "name": "equations"}]}
    assert names.project(record) == record
    # On the public route: a scientific graph case with no block (N9 spatial v14).
    from examples.api_v02.scientific_spatial_v14 import build_scientific_spatial_v14

    spec = build_scientific_spatial_v14(count=3)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert "periodic_diffusion_reaction_equations" in json.dumps(
        resolved.snapshot.semantic_to_dict())


_USER_MODULE = "{comment}half = {half}\n"


def _module_callback_snapshot(directory, name, *, comment="", half="0.5"):
    import importlib.util
    import sys

    directory.mkdir(parents=True)
    path = directory / ("%s.py" % name)
    path.write_text(_USER_MODULE.format(comment=comment, half=half))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        namespace = {name: module}
        exec("def callback(x):\n    return x * %s.half\n" % name, namespace)
        return AuthoringSnapshot({"callback": namespace["callback"]})
    finally:
        del sys.modules[name]


def test_a_user_module_read_by_a_callback_is_identified_by_its_values(tmp_path):
    """Criterion (i) on the user-module route that lot 2a left open (review rev 2)."""
    plain = _module_callback_snapshot(tmp_path / "a", "n11_user_module")
    commented = _module_callback_snapshot(
        tmp_path / "b", "n11_user_module", comment="# a comment\n\n# moves the line\n")
    changed = _module_callback_snapshot(tmp_path / "c", "n11_user_module", half="0.25")

    assert plain.artifact_hash == commented.artifact_hash
    assert plain.hash != commented.hash  # the source digest stays provenance
    assert plain.artifact_hash != changed.artifact_hash
