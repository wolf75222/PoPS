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
from pops.identity.presentation import PresentationMap, PresentationProjectionError
from pops.problem._snapshot import AuthoringSnapshot
from tests.python.support.presentation_case import FIRST, RENAMED, resolve_named_case


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


def test_the_map_never_gives_an_unknown_owner_another_label():
    names = PresentationMap(case="c", blocks=("a",), models=(("m", None, "x:sha256:" + "0" * 64),),
                            block_models=(0,), layouts=("grid",))
    assert names.project_string("pops.handle.v1::case:c::block::a") \
        == "pops.handle.v1::case:%24case::block::%24block0"
    assert names.project_string("pops.handle.v1::case:c::layout::grid") \
        == "pops.handle.v1::case:%24case::layout::%24layout0"
    # A name the map does not know keeps itself (residual), within this case or another one.
    assert names.project_string("pops.handle.v1::case:c::block::b") \
        == "pops.handle.v1::case:%24case::block::b"
    assert names.project_string("pops.handle.v1::case:c::layout::bootstrap") \
        == "pops.handle.v1::case:%24case::layout::bootstrap"
    # An owner of another case is kept whole: its block ``a`` is not this case's block ``a``.
    assert names.project_string("pops.handle.v1::case:other::block::a") \
        == "pops.handle.v1::case:other::block::a"
    assert names.project({"kind": "block", "local_id": "a",
                          "qualified_id": "pops.handle.v1::case:other::block::a"}) \
        == {"kind": "block", "local_id": "a", "qualified_id": "pops.handle.v1::case:other::block::a"}
    # A name of a kept kind passes unchanged.
    assert names.project_string("pops.handle.v1::case:c/descriptor:a::field::a") \
        == "pops.handle.v1::case:%24case/descriptor:a::field::a"
    # A malformed handle is refused.
    with pytest.raises(PresentationProjectionError):
        names.project_qualified_id("pops.handle.v1::case:c")


def test_an_authoring_path_is_projected_and_keeps_its_serial():
    # A runtime capture of a model parameter names its model by an authoring path, without a
    # definition fingerprint (pops.model.ownership.OwnerPath.__str__).
    fingerprint = "x:sha256:" + "0" * 64
    names = PresentationMap(case="c", blocks=("a",), models=(("m", fingerprint, fingerprint),),
                            block_models=(0,))
    assert names.project_string("pops.handle.v1::model_definition:m#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:%24model0#authoring=3::parameter::k"
    # A model outside the case keeps its name: validation refuses it with its own reason.
    assert names.project_string("pops.handle.v1::model_definition:other#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:other#authoring=3::parameter::k"
    # Two models of one name are told apart only by their fingerprint: the name is kept.
    homonyms = PresentationMap(case="c", blocks=("a", "b"),
                               models=(("m", fingerprint, fingerprint),
                                       ("m", "y:sha256:" + "1" * 64, "y:sha256:" + "1" * 64)),
                               block_models=(0, 1))
    assert homonyms.project_string("pops.handle.v1::model_definition:m#authoring=3::parameter::k") \
        == "pops.handle.v1::model_definition:m#authoring=3::parameter::k"


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
