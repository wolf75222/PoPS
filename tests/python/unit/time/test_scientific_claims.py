"""Evidence categories and provenance, not native numerical receptions."""
import hashlib
import json

import pytest

from pops.claims import Claim, claims_manifest


def test_assumptions_and_unestablished_properties_never_become_proofs():
    assumed = Claim.assumption("solution is regular", scope="smooth initial data",
                               justification="analytical hypothesis")
    missing = Claim.unestablished("asymptotic preservation", reason="no limit proof supplied")
    exported = claims_manifest((assumed, missing))
    assert not exported["verified_graph_identity"]
    assert not exported["reported_numerical_execution"]
    assert len(exported["analytical_assumption"]) == len(exported["unestablished"]) == 1
    with pytest.raises(TypeError, match="Claim objects"):
        claims_manifest((True,))


def test_graph_identity_is_verified_from_finite_definition_not_a_claim_flag():
    from pops.graph_functions import graph, GraphFunction
    from pops.time.variant_types import Float64, Product
    function = graph(inputs=Product(value=Float64), output=Float64)(
        lambda p, x: p.value("doubled", x["value"]*2))
    restored = GraphFunction.from_data(function.to_data())
    claim = Claim.graph_identity("same finite algorithm", left=function, right=restored)
    assert claim.category == "verified_graph_identity"
    assert claim.to_data()["evidence"]["graph_sha256"]
    different = graph(inputs=Product(value=Float64), output=Float64)(
        lambda p, x: p.value("doubled", x["value"]*3))
    with pytest.raises(ValueError, match="not structurally identical"):
        Claim.graph_identity("invented theorem", left=function, right=different)
    with pytest.raises(TypeError, match="GraphFunction"):
        Claim.graph_identity("fake", left=True, right=True)


def test_numerical_claim_authenticates_receipt_bytes_artifact_and_selected_check(tmp_path):
    # Synthetic transport fixture tests provenance validation, not an executed PoPS run.
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"artifact": "sha256:fixture", "checks": {"mass": True, "entropy": False}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    claim = Claim.numerical_receipt("mass check reported", path=path, sha256=digest,
                                    check_path=("checks", "mass"), artifact_path=("artifact",))
    assert claim.category == "reported_numerical_execution"
    assert claim.to_data()["evidence"]["artifact_identity"] == "sha256:fixture"
    with pytest.raises(ValueError, match="passed"):
        Claim.numerical_receipt("entropy", path=path, sha256=digest,
                                check_path=("checks", "entropy"), artifact_path=("artifact",))
    path.write_text("{}")
    with pytest.raises(ValueError, match="bytes differ"):
        Claim.numerical_receipt("stale", path=path, sha256=digest,
                                check_path=("checks", "mass"), artifact_path=("artifact",))
