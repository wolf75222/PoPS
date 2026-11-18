"""Authenticate a native global-guard record against the bound artifact."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pops.provenance import SourceSpan
from pops.time._program.guard_failure_sites import SCHEMA
from pops.time._step.transaction import GuardFailureSite, validate_guard_failure_sites

if TYPE_CHECKING:
    from pops.runtime._simulation import Failure


def bound_guard_failure_catalog(artifact: Any) -> Mapping[str, GuardFailureSite]:
    """Retain only sites sealed by the already verified InstallPlan."""
    sites = tuple(site for row in artifact.layout_programs
                  for site in getattr(row.program, "guard_failure_sites", ()))
    validate_guard_failure_sites([site.to_data() for site in sites])
    return MappingProxyType({
        site.token: replace(site, python_origin=SourceSpan.from_data(site.python_origin.to_data()))
        for site in sites
    })


def decode_guard_failure(
    schema: Any, image: Any, catalog: Mapping[str, GuardFailureSite], *, action: str,
    compatibility: Mapping[str, Any] | None = None,
) -> Failure | None:
    if schema != SCHEMA:
        return None
    # 32-byte site digest and 64-bit predicate ordinal in lowercase hex, one Bool.
    if type(image) is not bytes or len(image) != 81 or image[-1:] != b"0":
        raise RuntimeError("native global guard returned an invalid T39 record")
    token = image[:64].decode("ascii", errors="replace")
    site = catalog.get(token)
    if site is None or site.action != action or site.payload_prefix().encode("ascii") != image[:-1]:
        raise RuntimeError("native global guard T39 record does not match the bound site")
    from pops.runtime._simulation import Failure, NotAdmissible

    payload = {} if compatibility is None else {
        **compatibility, "site": site.name, "reason": "NotAdmissible",
    }
    payload.update(predicate=site.predicate, value=image[-1:] == b"1")
    return Failure(NotAdmissible, "trial", payload, site=site)


def attach_guard_failure(
    error: BaseException, catalog: Mapping[str, GuardFailureSite],
) -> Failure | None:
    """Attach the typed T39 failure after rollback, preserving the native action."""
    from pops._bootstrap import StepAttemptRejected, StepTerminalFailure

    if type(error) not in (StepAttemptRejected, StepTerminalFailure):
        return None
    action = "reject_attempt" if type(error) is StepAttemptRejected else "fail_run"
    failure = decode_guard_failure(
        getattr(error, "typed_failure_schema", None), getattr(error, "typed_failure_payload", None),
        catalog, action=action,
    )
    if failure is not None:
        error.__dict__["failure"] = failure
    return failure
