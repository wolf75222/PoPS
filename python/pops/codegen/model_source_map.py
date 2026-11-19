"""Where the generated C++ of a block's model comes from in Python: its user bodies.

The model C++ of a block holds the policies that PoPS generates from Python bodies: the face of a
``riemann.User`` (``UserFacePolicy``) and the trace of a ``reconstruction.User``
(``UserReconstructionPolicy``). This map names, for each such policy, its lines in the retained
C++ and the origin of its body: the file, the line and the function where it is written, the case
file or ``pops.lib`` (``pops._source_origin``). Like the Program map, it is evidence of source
correspondence only, never of execution; it reads the retained text and the descriptors, and
changes neither, so it stays outside the artifact identity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from pops._structural_data import canonical_json

CONTRACT = "pops.model-source-map@1"
# The policy that each category of generated brick lowers to (user_riemann_lowering.py,
# user_reconstruction_lowering.py).
_POLICIES = {"riemann": "UserFacePolicy", "reconstruction": "UserReconstructionPolicy"}
_IDENTITY = re.compile(r'^\s*static constexpr const char\* source_identity = "([0-9a-f]+)";$')


def user_bodies(spatial: Any) -> list[Any]:
    """The generated bricks of a spatial discretization that lower to a policy of the model."""
    from pops.descriptors import BrickDescriptor

    found = []
    for category in _POLICIES:
        value = getattr(spatial, category, None)
        if type(value) is BrickDescriptor and value.brick_type == "generated" and \
                value.category == category and "source_identity" in value.options:
            found.append(value)
    return found


def _policies(lines: list[str]) -> list[dict[str, Any]]:
    """Each generated policy struct of the text: its name, its lines and its source identity."""
    found = []
    names = set(_POLICIES.values())
    index = 0
    while index < len(lines):
        match = re.fullmatch(r"struct (\w+) \{", lines[index])
        if match is None or match.group(1) not in names:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end] != "};":
            end += 1
        if end == len(lines):
            raise ValueError("model source map: struct %s is not closed" % match.group(1))
        identities = [m.group(1) for m in map(_IDENTITY.match, lines[index:end]) if m]
        found.append({"policy": match.group(1), "ranges": [[index + 1, end + 1]],
                      "source_identity": identities[0] if len(identities) == 1 else None})
        index = end + 1
    return found


def model_source_map(source: str, spatial: Any) -> dict[str, Any]:
    """The map of the user bodies of one block: ``source`` is its retained model C++, and
    ``spatial`` the spatial discretization whose generated bricks it lowers.

    Each row of ``bodies`` is one generated brick: the policy it lowers to, its one-based
    inclusive line ranges in ``source``, its source identity and the ``origin`` of its body.
    A brick whose policy is not found in the text, a policy of the text that no brick claims,
    or a brick with no origin leaves the map incomplete; nothing is guessed."""
    if type(source) is not str:
        raise TypeError("model source map requires the exact generated C++ text")
    lines = source.splitlines()
    located = _policies(lines)
    bodies, claimed = [], set()
    for brick in user_bodies(spatial):
        policy = _POLICIES[brick.category]
        identity = brick.options["source_identity"]
        matches = [index for index, row in enumerate(located)
                   if row["policy"] == policy and row["source_identity"] == identity]
        origin = getattr(brick, "origin", None)
        row = {"policy": policy, "category": brick.category, "name": brick.name,
               "source_identity": identity,
               "ranges": [span for index in matches for span in located[index]["ranges"]],
               "origin": None if origin is None else dict(origin)}
        row["status"] = ("not_emitted" if not matches else
                         "no_origin" if origin is None else "located")
        claimed.update(matches)
        bodies.append(row)
    unclaimed = [dict(row, status="no_brick") for index, row in enumerate(located)
                 if index not in claimed]
    result = {"contract": CONTRACT, "evidence": "source_correspondence_only",
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "line_count": len(lines), "bodies": bodies, "unlocated": unclaimed,
              "complete": not unclaimed and all(row["status"] == "located" for row in bodies)}
    result["map_sha256"] = hashlib.sha256(
        canonical_json(result, allow_nan=False).encode()).hexdigest()
    return result


__all__ = ["CONTRACT", "model_source_map", "user_bodies"]
