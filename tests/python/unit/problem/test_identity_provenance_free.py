"""N11 lot 2a (spec 0.3 CC2, gap C48): source spans are provenance, never artifact identity.

One case module is loaded from two directories (two checkouts of one commit), and once more with
comment lines inserted before the case. The compile identities (``snapshot.artifact_hash`` and
``plan_identity``) and the semantic identity are equal across all three, while the full authored
snapshot keeps each file and line. A decoy that changes one constant changes the compile identity,
so the projection is not blind to the formula.
"""

import json
import sys
from pathlib import Path

import pytest

import pops
from tests.python.support.provenance_case import (
    COMMENT, compile_identities, resolve_written_case,
)


def test_two_checkout_paths_have_one_compile_identity(tmp_path):
    here, here_path = resolve_written_case(tmp_path / "checkout_a" / "examples")
    there, there_path = resolve_written_case(tmp_path / "elsewhere" / "checkout_b" / "examples")

    assert compile_identities(here) == compile_identities(there)
    # The full authored capture is provenance: it keeps the file of each declaration.
    assert here.snapshot.hash != there.snapshot.hash
    assert here_path in json.dumps(here.snapshot.to_dict())
    assert there_path in json.dumps(there.snapshot.to_dict())
    # No absolute path reaches the compile projection.
    artifact_text = json.dumps(here.snapshot.artifact_to_dict())
    assert here_path not in artifact_text
    assert str(tmp_path) not in artifact_text


def test_a_comment_does_not_change_the_compile_identity(tmp_path):
    plain, _ = resolve_written_case(tmp_path / "plain")
    commented, _ = resolve_written_case(tmp_path / "commented", comment=COMMENT)

    assert compile_identities(plain) == compile_identities(commented)
    assert plain.snapshot.hash != commented.snapshot.hash


def test_a_changed_constant_changes_the_compile_identity(tmp_path):
    reference, _ = resolve_written_case(tmp_path / "reference")
    decoy, _ = resolve_written_case(tmp_path / "decoy", loss="0.25")

    for key, value in compile_identities(reference).items():
        assert compile_identities(decoy)[key] != value, key


def _body(source, name="f"):
    namespace = {"__name__": "provenance_body_probe"}
    exec(compile(source, "/checkout/provenance_body_probe.py", "exec"), namespace)
    return namespace[name]


_BODIES = {
    "base": "def f(u):\n    a = 2*u\n    return a\n",
    "comment": "def f(u):\n    # a comment\n    a = 2*u  # another\n    return a\n",
    "shift": "\n\n\n# moved down\ndef f(u):\n    a = 2*u\n    return a\n",
    "docstring": 'def f(u):\n    """Twice u."""\n    a = 2*u\n    return a\n',
}


def test_a_comment_or_a_shift_keeps_the_body_identity():
    from pops.model.hash_data import body_identity

    reference = body_identity(_body(_BODIES["base"]))
    for variant in ("comment", "shift"):
        assert body_identity(_body(_BODIES[variant])) == reference, variant
    assert body_identity(_body("def f(u):\n    a = 3*u\n    return a\n")) != reference


# The docstring fold reads the CPython code layout (docstring in co_consts[0]), checked on 3.11 to 3.13.
@pytest.mark.skipif(not (3, 11) <= sys.version_info[:2] <= (3, 13),
                    reason="docstring code layout checked on CPython 3.11 to 3.13 only")
def test_a_docstring_keeps_the_body_identity():
    from pops.model.hash_data import body_identity

    reference = body_identity(_body(_BODIES["base"]))
    assert body_identity(_body(_BODIES["docstring"])) == reference
    # A body returning None implicitly: the docstring moves the None slot, which is folded back.
    plain = body_identity(_body("def f(u):\n    u.append(1)\n"))
    documented = body_identity(_body('def f(u):\n    """Append."""\n    u.append(1)\n'))
    assert documented == plain
    # A docstring the body also reads as a literal stays visible.
    reads_doc = body_identity(_body('def f(u):\n    "abc"\n    y = "abc"\n    return y\n'))
    returns_none = body_identity(_body("def f(u):\n    y = None\n    return y\n"))
    assert reads_doc != returns_none


def _spans_and_paths(value, roots, path="", found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        if {"file", "line"} <= set(value):
            found.append("%s: span %r" % (path, value))
        for key, item in value.items():
            if any(root in str(key) for root in roots):
                found.append("%s: path in key %r" % (path, key))
            _spans_and_paths(item, roots, "%s/%s" % (path, key), found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _spans_and_paths(item, roots, "%s[%d]" % (path, index), found)
    elif isinstance(value, str) and any(root in value for root in roots):
        found.append("%s: path %r" % (path, value[:120]))
    return found


_REPOSITORY = str(Path(__file__).resolve().parents[4])


def test_the_artifact_projection_holds_no_span_and_no_checkout_path(tmp_path):
    resolved, _ = resolve_written_case(tmp_path / "invariant", comment=COMMENT)
    roots = (str(tmp_path), _REPOSITORY)
    assert _spans_and_paths(resolved.snapshot.artifact_to_dict(), roots) == []
    # The walk sees spans: the full authored capture keeps them.
    assert _spans_and_paths(resolved.snapshot.to_dict(), roots) != []


@pytest.mark.parametrize("profile", ["M01", "M03"])
def test_a_profile_artifact_projection_holds_no_span_and_no_checkout_path(profile):
    from examples.api_v02.profile_dispatch import build_profile

    spec = build_profile(profile)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert _spans_and_paths(resolved.snapshot.artifact_to_dict(), (_REPOSITORY,)) == []
    assert _spans_and_paths(resolved.snapshot.to_dict(), (_REPOSITORY,)) != []
