"""One fixed fixture/callsite for fresh parent and candidate source receipts.

Only the package search root changes between isolated interpreters. No
provenance is stripped, rewritten or normalized before comparison.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    package_root = Path(sys.argv[1]).resolve()
    gamma = .3 if len(sys.argv) == 2 else float(sys.argv[2])
    fixture_root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(package_root), str(fixture_root)]
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.identity import canonical_bytes
    from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback

    if Path(pops.__file__).resolve() != package_root / "pops" / "__init__.py":
        raise ValueError("control imported a different package than the named source")
    receipts = {}
    for physical_global in (False, True):
        for periodic in (False, True):
            # This authoring callsite is identical in BOTH interpreters.
            case, layout, program, _, _ = build_feedback(
                physical_global=physical_global, periodic=periodic, gamma=gamma)
            module = case._block_registry.spec("fluid")["model"].module
            manifest = module.manifest().to_dict()
            plan = pops.resolve(pops.validate(case), layout=layout)
            graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
            receipt = {
                "ir": program._ir_hash(),
                "module_hash": module.module_hash(),
                "module_manifest": manifest,
                "manifest_version": manifest["schema_version"],
                "plan": plan.plan_identity.token,
                "plan_payload_hex": canonical_bytes(plan._payload()).hex(),
                "snapshot_artifact_json": plan.snapshot._artifact_canonical_json,
            }
            for target in ("system", "amr_system"):
                cpp = emit_cpp_program(plan.time, model_graph=graph, target=target)
                receipt[target] = hashlib.sha256(cpp.encode()).hexdigest()
            receipts[str((physical_global, periodic))] = receipt
    print(json.dumps(receipts, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
