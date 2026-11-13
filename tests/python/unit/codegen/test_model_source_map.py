"""The model source map (``pops.model-source-map@1``): the user bodies of a block's model C++.

The policies that PoPS generates from Python bodies, the face of ``riemann.User`` and the trace of
``reconstruction.User``, are found in the retained C++ by their struct and their source identity,
and each is named with the origin of its body: its file, its line and its function. The origin is
local provenance: it stays outside the identity of the descriptor. Nothing here compiles.
"""
from __future__ import annotations

from types import SimpleNamespace

from pops._source_origin import code_origin, origin_text
from pops.codegen.model_source_map import model_source_map
from pops.descriptors import BrickDescriptor

ORIGIN = {"written_in": "case", "path": "/cases/l1.py", "line": 12, "function": "upwind"}


def _brick(category, identity, origin=ORIGIN):
    return BrickDescriptor("user", "generated", category=category,
                           options={"source_identity": identity}, origin=origin)


def _policy(name, identity):
    return ["namespace pops_generated {", "struct %s {" % name,
            '  static constexpr const char* source_identity = "%s";' % identity,
            "  int body() const { return 0; }", "};", "}  // namespace pops_generated"]


SOURCE = "\n".join(["#include <pops/core.hpp>", *_policy("UserFacePolicy", "aa11"),
                    *_policy("UserReconstructionPolicy", "bb22"), ""])


def test_each_body_is_located_by_its_struct_and_its_identity_and_named_by_its_origin():
    spatial = SimpleNamespace(riemann=_brick("riemann", "aa11"),
                              reconstruction=_brick("reconstruction", "bb22", origin=dict(
                                  ORIGIN, written_in="pops.lib", function="muscl")))
    data = model_source_map(SOURCE, spatial)
    assert data["contract"] == "pops.model-source-map@1" and data["complete"]
    assert data["evidence"] == "source_correspondence_only" and data["unlocated"] == []
    lines = SOURCE.splitlines()
    face, trace = data["bodies"]
    assert (face["policy"], face["status"], face["origin"]) == ("UserFacePolicy", "located",
                                                                ORIGIN)
    assert trace["policy"] == "UserReconstructionPolicy" and trace["origin"]["function"] == "muscl"
    for row in data["bodies"]:
        ((first, last),) = row["ranges"]
        assert lines[first - 1] == "struct %s {" % row["policy"] and lines[last - 1] == "};"
    assert model_source_map(SOURCE, spatial) == data


def test_a_body_that_is_missing_unclaimed_or_without_origin_leaves_the_map_incomplete():
    other = model_source_map(SOURCE, SimpleNamespace(riemann=_brick("riemann", "cc33"),
                                                     reconstruction=None))
    assert not other["complete"]
    assert [row["status"] for row in other["bodies"]] == ["not_emitted"]
    assert {row["policy"] for row in other["unlocated"]} == {"UserFacePolicy",
                                                              "UserReconstructionPolicy"}
    bare = model_source_map(SOURCE, SimpleNamespace(
        riemann=_brick("riemann", "aa11", origin=None),
        reconstruction=_brick("reconstruction", "bb22")))
    assert not bare["complete"]
    assert [row["status"] for row in bare["bodies"]] == ["no_origin", "located"]


def test_the_origin_stays_outside_the_identity_of_the_descriptor():
    moved = _brick("riemann", "aa11", origin=dict(ORIGIN, path="/elsewhere/l1.py"))
    assert moved == _brick("riemann", "aa11") and hash(moved) == hash(_brick("riemann", "aa11"))


def test_the_origin_of_a_body_names_the_case_file_or_pops_lib():
    def face(uL, uR):
        return uL

    here = code_origin(face)
    assert here["written_in"] == "case" and here["path"] == __file__
    assert (here["line"], here["function"]) == (
        face.__code__.co_firstlineno,
        "test_the_origin_of_a_body_names_the_case_file_or_pops_lib.<locals>.face")
    from pops.lib.numerics import muscl

    library = code_origin(muscl.body)
    assert (library["written_in"], library["path"], library["function"]) == (
        "pops.lib", "pops/lib/numerics/limiters.py", "muscl")
    assert origin_text(library) == "pops.lib, pops/lib/numerics/limiters.py:%d" % library["line"]
    assert origin_text(here) == "the case file, %s:%d" % (__file__, here["line"])
