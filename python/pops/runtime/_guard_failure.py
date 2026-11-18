"""Authenticate a native global-guard record against the bound artifact."""
from __future__ import annotations

from typing import Any

from pops._bootstrap import StepAttemptRejected, StepTerminalFailure
from pops.runtime._simulation import Failure, NotAdmissible
from pops.time._program.guard_failure_sites import SCHEMA


def attach_guard_failure(error: BaseException, artifact: Any) -> Failure | None:
    """Attach the typed T39 failure after rollback, preserving the native action."""
    if type(error) not in (StepAttemptRejected, StepTerminalFailure):
        return None
    if getattr(error, "typed_failure_schema", None) != SCHEMA:
        return None
    image = getattr(error, "typed_failure_payload", None)
    # 32-byte site digest and 64-bit predicate ordinal in lowercase hex, one Bool.
    if type(image) is not bytes or len(image) != 81 or image[-1:] != b"0":
        raise RuntimeError("native global guard returned an invalid T39 record") from error
    candidates = [
        site for row in artifact.layout_programs for site in row.program.guard_failure_sites
        if site.payload_prefix().encode("ascii") == image[:-1]
    ]
    action = "reject_attempt" if type(error) is StepAttemptRejected else "fail_run"
    if len(candidates) != 1 or candidates[0].action != action:
        raise RuntimeError("native global guard T39 record does not match the bound site") from error
    site = candidates[0]
    failure = Failure(NotAdmissible, "trial", {"predicate": site.predicate,
                                             "value": image[-1:] == b"1"}, site=site)
    error.__dict__["failure"] = failure
    return failure
