"""Rebuild the provisional API 0.4.0 contract and corpus inventories.

This script records source-level candidates. It does not run PoPS tests or infer
that a symbol satisfies the complete contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HANDOFF = ROOT / "PoPS_Codex_handoff_0.4.0"
OUT = Path(__file__).resolve().parent
HEAD = "3a93ba7f1b3fc46ee85a06d9d3482c9b4ece7feb"

# id | current mechanism | candidate source symbols | precise remaining gap
ROWS = """
C01|partial_gap|python/pops/_api.py:validate/resolve/compile/bind/run;python/pops/time/_step/transaction.py:StepTransactionPlan|No proof that every physical relation and publication authority stays distinct through the whole path
C02|existing_unverified|python/pops/physics/_facade.py:Model;python/pops/problem/problem.py:Problem;python/pops/time/_program/api.py:Program|Architecture correspondence needs an end-to-end example
C03|partial_gap|python/pops/_version.py;python/pops/codegen/_compiled_artifact.py|No evidence that reference specification 0.4.0 and native extension ABI 2 are accepted as distinct production versions
C04|gap|python/pops/problem/report.py|No central EXPR/IMPL/MATH/SCOPE diagnosis verified for the corpus
C05|partial_gap|python/pops/model/identity.py:StateSchema;python/pops/time/value_support.py;python/pops/spaces/__init__.py:StatePlacement|Full unit representation and heterogeneous support compatibility unverified
C06|new_code_pending|python/pops/_ir/balance.py:BalanceOccurrence;python/pops/physics/board_handles.py:RateHandle.occurrences|Repeated physical occurrences need a public S+S test through codegen
C07|new_code_pending|python/pops/_ir/expr.py:Expr/Equation;python/pops/time/_program/authoring.py:where|Cross-program vector/where and symbolic equality need independent negative tests
C08|partial_gap|python/pops/time/value_metadata.py;python/pops/model/identity.py:StateSchema|No general proof that nonlinear functions of FV coordinates preserve representation meaning
C09|partial_gap|python/pops/analytic/_functions.py:where;python/pops/time/_program/authoring.py:where|Lazy branch and binary64 rounded barrier at all lowering stages unverified
C10|partial_gap|python/pops/physics/_board_rate.py:rate/nonconservative_product;python/pops/numerics/nonconservative.py:PathConservativeFiniteVolume|Burgers weak-law and nonconservative path semantics need end-to-end cases
C11|partial_gap|python/pops/_ir/balance.py:BalanceView;python/pops/physics/_board_rate.py:rate|Principal-block equivalence across one vector versus scalar states unverified
C12|partial_gap|python/pops/physics/_authoring_flux.py:flux_term;python/pops/numerics/spatial.py:FiniteVolume|Open user face/path bodies through native lowering unverified
C13|partial_gap|python/pops/physics/_board_rate.py:nonconservative_product;python/pops/codegen/nonconservative_lowering.py|Full Fan-Li path and interior contribution for higher order unverified
C14|partial_gap|python/pops/numerics/diffusion.py:Diffusion;python/pops/numerics/reconstruction/__init__.py:_weno5;python/pops/codegen/program_emit_diffusion.py|Joint flux coverage and general typed stencil body unverified
C15|partial_gap|python/pops/linalg/problem.py:LinearProblem;python/pops/fields/problem.py:FieldProblem;python/pops/time/_program/local.py:solve|Mixed unknown products and fixed capture semantics unverified
C16|new_code_pending|python/pops/time/_program/local.py:solve;python/pops/fields/residual.py|Unknown-dependent named computations must stay inside residual region
C17|partial_gap|python/pops/solvers/tolerances.py;python/pops/linalg/problem.py:LinearProblem|Original residual and conditioning guarantees depend on actual solver route
C18|partial_gap|python/pops/fields/nullspace.py;python/pops/fields/gauges.py|Joint kernel compatibility and representative selection for target systems unverified
C19|partial_gap|python/pops/time/_step/transaction.py:StepTransactionPlan;python/pops/fields/solve.py:publish|Failed native solve must not partially publish every output
C20|new_code_pending|python/pops/time/_program/clocks.py:stage;python/pops/time/values.py:ProgramValue|Same-time distinct stage identity and immutable capture need tests
C21|partial_gap|python/pops/time/_program/authoring.py:step;python/pops/lib/time/rk.py;python/pops/lib/time/ssprk.py|Public arbitrary method construction and implicit reaction route unverified
C22|partial_gap|python/pops/time/_program/clocks.py:stage;python/pops/time/_program/solve.py:commit|General reached-time frontier and relaxed endpoint not established
C23|partial_gap|python/pops/time/_program/solve.py:commit_many;python/pops/_balance_contract.py:BalanceLedger|Candidate exchange replacement across AMR consumers not demonstrated
C24|partial_gap|python/pops/time/_schedule/ir.py;python/pops/time/_program/calls.py|Dependency-order equivalence for coupled stage solves unverified
C25|partial_gap|python/pops/codegen/_compile.py;python/pops/codegen/program_emit_solve.py|Open common bodies without model-name compiler branches need inspection and tests
C26|gap|python/pops/_pops.pyi|Reference pops_initial/pops_step/pops_run ABI is separate and not a production capability
C27|gap|include/pops;python/pops/native_calls.py|Experimental pops_contract.hpp ABI 2 is not integrated with production runtime
C28|partial_gap|python/pops/codegen/_compile.py;python/pops/runtime/_runtime_executor.py|No comparable performance campaign for API 0.4.0 paths
C29|partial_gap|python/pops/time/_step/transaction.py:StepTransactionPlan;python/pops/runtime/_consumer_transaction.py:ConsumerTransaction|No complete composition proof for values tasks resources attempts and migration
C30|partial_gap|python/pops/problem/_freeze_transaction.py;python/pops/time/_program/api.py:freeze|Deep immutable closure and invalidation after all structural changes unverified
C31|partial_gap|python/pops/time/value_metadata.py;python/pops/model/identity.py:StateSchema|Cross-stage version and scope validity across all values unverified
C32|partial_gap|python/pops/codegen/program_field_reuse.py;python/pops/runtime/_history_sample_identity.py|Query key completeness for ordered args versions and duration unverified
C33|partial_gap|python/pops/time/_step/transaction.py:StepTransactionPlan|Nested attempt state with parent/child isolation unverified
C34|partial_gap|python/pops/runtime/_consumer_transaction.py:ConsumerTransaction|Concurrent root publication and read-set conflict unverified
C35|partial_gap|python/pops/time/_step/transaction.py:AcceptanceGuard;python/pops/runtime/_consumer_transaction.py:ConsumerTransaction|Numerical acceptance must gate all side effects and remain separate from publication
C36|gap|python/pops/runtime/_consumer_transaction.py:ConsumerTransaction|Async cancellation acknowledgement and native resource lease not established
C37|partial_gap|python/pops/runtime/_continuation_transitions.py:ContinuationTransitionPlan|Progress and fairness guarantees for every proposed transition unverified
C38|partial_gap|python/pops/mesh/_amr/hierarchy_regrid.py:RegridTransactionGate;python/pops/runtime/_multi_layout_executor.py|Atomic migration of histories fields and all consumer epochs unverified
C39|partial_gap|python/pops/mesh/_amr/transfer.py:AMRTransfer|Conservation admissibility field re-solve and representation constraints need independent cases
C40|partial_gap|python/pops/runtime/_amr_checkpoint_v3.py;python/pops/output/_restart_provider.py|Crash durability and distributed consensus not established
""".strip().splitlines()

corpus_symbols = {
    "M01": "python/pops/numerics/spatial.py:FiniteVolume;python/pops/lib/time/ssprk.py",
    "M02": "python/pops/physics/_authoring_flux.py:flux_term;python/pops/numerics/spatial.py:FiniteVolume",
    "M03": "python/pops/physics/_facade.py:Model;include/pops/physics/fluids/euler.hpp",
    "M04": "python/pops/numerics/diffusion.py:Diffusion;python/pops/numerics/spatial.py:FiniteVolume",
    "M05": "python/pops/numerics/diffusion.py:Diffusion",
    "M06": "python/pops/physics/_board_rate.py:rate",
    "M07": "python/pops/physics/_authoring_flux.py:flux_term",
    "M08": "python/pops/fields/poisson.py;python/pops/time/_program/calls.py",
    "M09": "python/pops/time/_program/condensed.py;python/pops/fields/problem.py:FieldProblem",
    "M10": "python/pops/physics/drift_diffusion.py;python/pops/fields/poisson.py",
    "M11": "python/pops/physics/diffusion.py;python/pops/fields/problem.py:FieldProblem",
    "M12": "python/pops/physics/multispecies.py;python/pops/fields/poisson.py",
    "M13": "python/pops/physics/_authoring_sources.py:source_term;python/pops/time/_program/local.py:solve",
    "M14": "python/pops/time/_step/transaction.py:StepTransactionPlan",
    "M15": "python/pops/moments/hierarchy.py",
    "M16": "python/pops/moments/transport.py;python/pops/physics/_board_rate.py:rate",
    "M17": "python/pops/physics/_board_rate.py:nonconservative_product;python/pops/numerics/nonconservative.py:PathConservativeFiniteVolume",
    "M18": "python/pops/moments/closures/protocol.py;python/pops/time/_program/local.py:solve",
    "M19": "python/pops/moments/projection.py;python/pops/fields/poisson.py",
    "M20": "python/pops/physics/diffusion.py",
    "M21": "python/pops/physics/_facade.py:Model",
    "M22": "python/pops/physics/_board_rate.py:rate",
    "M23": "python/pops/physics/_board_rate.py:rate",
    "M24": "python/pops/physics/_board_rate.py:nonconservative_product",
    "M25": "python/pops/physics/_board_rate.py:rate",
    "M26": "python/pops/time/_program/local.py:solve",
    "M27": "python/pops/fields/problem.py:FieldProblem;python/pops/time/_program/local.py:solve",
    "M28": "python/pops/mesh/_amr/hierarchy_regrid.py:RegridTransactionGate",
}

witnesses = {
    "W01": ("Cosinus scalaire en FV", "M01", "python/pops/initial/__init__.py;python/pops/numerics/spatial.py:FiniteVolume"),
    "W02": ("Advection-diffusion avec borne combinée", "M04", "python/pops/numerics/diffusion.py:Diffusion;python/pops/time/_step/transaction.py:AcceptanceGuard"),
    "W03": ("Flux principal entre deux blocs", "M12", "python/pops/_ir/balance.py:BalanceView"),
    "W04": ("Poisson au vrai stage", "M08", "python/pops/time/_program/clocks.py:stage;python/pops/fields/poisson.py"),
    "W05": ("Rotation pure de Lorentz", "M12", "python/pops/physics/_authoring_sources.py:source_term"),
    "W06": ("Solve par blocs contre solve monolithique", "M09", "python/pops/time/_program/condensed.py;python/pops/linalg/problem.py:LinearProblem"),
    "W07": ("HyQMOM15 états réalisables normales et permutations", "M16", "python/pops/moments/hierarchy.py;python/pops/moments/transport.py"),
    "W08": ("Fan-Li réellement non conservatif", "M17", "python/pops/numerics/nonconservative.py:PathConservativeFiniteVolume"),
    "W09": ("Fermeture infaisable sur sa quadrature", "M18", "python/pops/moments/closures/protocol.py"),
    "W10": ("Contrainte collective de mélange", "M11", "python/pops/physics/diffusion.py"),
    "W11": ("Rejet après calcul d'un courant de paroi", "M14", "python/pops/time/_step/transaction.py:StepTransactionPlan"),
    "W12": ("Construction dépendant de la durée", "M17", "python/pops/runtime/_history_sample_identity.py"),
}


def main() -> None:
    all_symbol_sets = [line.split("|", 3)[2] for line in ROWS]
    all_symbol_sets += list(corpus_symbols.values())
    all_symbol_sets += [record[2] for record in witnesses.values()]
    for symbol_set in all_symbol_sets:
        for reference in symbol_set.split(";"):
            path = reference.split(":", 1)[0]
            assert (ROOT / path).exists(), path

    checklist_path = HANDOFF / "checklists/CONTRACTS_TO_IMPLEMENT.csv"
    with checklist_path.open(newline="") as source:
        checklist = {row["contract_id"]: row for row in csv.DictReader(source)}
    assert len(checklist) == 40 and len(ROWS) == 40
    fieldnames = [
        "contract_id", "title_as_in_spec", "spec_source", "spec_line",
        "reference_test_ids", "current_head", "upstream_status", "upstream_symbols",
        "remaining_gap", "upstream_test_status",
    ]
    out_rows = []
    for line in ROWS:
        cid, status, symbols, gap = line.split("|", 3)
        assert cid in checklist and status in {"existing_unverified", "partial_gap", "gap", "new_code_pending"}
        source = checklist[cid]
        out_rows.append({
            "contract_id": cid, "title_as_in_spec": source["title_as_in_spec"],
            "spec_source": source["spec_source"], "spec_line": source["spec_line"],
            "reference_test_ids": source["reference_test_ids"], "current_head": HEAD,
            "upstream_status": status, "upstream_symbols": symbols,
            "remaining_gap": gap, "upstream_test_status": "not_executed_for_this_mapping",
        })
    assert {r["contract_id"] for r in out_rows} == set(checklist)
    with (OUT / "contracts.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    matrix_path = HANDOFF / "reference/PoPS_API_v0.4.0/results/corpus_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    assert len(matrix["models"]) == 28 and set(corpus_symbols) == {m["id"] for m in matrix["models"]}
    models = []
    for model in matrix["models"]:
        models.append({
            "id": model["id"], "title": model["title"], "definition": model["definition"],
            "planned_test": model["planned_test"],
            "reference_profile": model["expressibility_profile"],
            "reference_limit": model["new_limit"],
            "reference_evidence": model["new_public_evidence"],
            "production_candidate_symbols": corpus_symbols[model["id"]],
            "production_status": "not_executed_for_this_mapping",
            "qualification": "candidate symbols only; complete native case not established",
        })
    payload = {
        "source": str(matrix_path.relative_to(ROOT)),
        "witness_source": "PoPS_Codex_handoff_0.4.0/context/CORPUS_ORIGINAL.md:278-289",
        "current_head": HEAD,
        "mapping_kind": "provisional source-level candidates; no production tests run by this inventory",
        "models": models,
        "witnesses": [
            {"id": wid, "title": title, "related_model": model,
             "production_candidate_symbols": symbols,
             "production_status": "not_executed_for_this_mapping"}
            for wid, (title, model, symbols) in witnesses.items()
        ],
    }
    (OUT / "corpus.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
