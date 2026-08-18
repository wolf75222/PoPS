"""Official PF mapping. Absent benches are classified, not wrapped."""
from __future__ import annotations

from verification.pops_verify.official_benchmark import (
    OFFICIAL_MANIFEST,
    OfficialBenchmarkUnavailable,
    refuse_unofficial_pf,
    run_official_benchmark,
)

OFFICIAL_PF_MAP = {
    "PF-01": "arith_halo",
    "PF-02": "scalar_mg",
}


def official_authority(verification_id: str) -> dict:
    """Return the official bench for a PF id, or an honest absence."""
    mapped = OFFICIAL_PF_MAP.get(verification_id)
    if mapped is None:
        return {
            "verification_id": verification_id,
            "manifest": str(OFFICIAL_MANIFEST),
            "case_id": None,
            "status": "not-supported",
            "reason": refuse_unofficial_pf(verification_id),
        }
    return {
        "verification_id": verification_id,
        "manifest": str(OFFICIAL_MANIFEST),
        "case_id": mapped,
        "status": "official",
    }


def run_mapped_or_refuse(verification_id: str, request=None) -> dict:
    """Run an official case or raise with the official-absence reason."""
    del request
    authority = official_authority(verification_id)
    if authority["status"] != "official":
        raise OfficialBenchmarkUnavailable(authority["reason"])
    result = run_official_benchmark(authority["case_id"])
    result.update(authority)
    return result
