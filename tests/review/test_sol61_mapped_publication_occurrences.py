"""Independent points retain one physical producer, never competing same-point writes."""
from copy import deepcopy
import pytest
from pops.codegen.program_field_publication import _merge_publication_claim

def claim(point):
    return {"producer":"physical", "unknown":"phi", "key":"destination",
            "mapped_output": {"invocation":"invocation-"+point, "physical_map":"map",
                "source_port":"source", "target_port":"target", "source_point":point, "target_point":point}}

def test_distinct_points_keep_exact_occurrences():
    a,b=claim("initial"),claim("predictor")
    result=_merge_publication_claim(a,b)
    assert result["mapped_occurrences"] == [a["mapped_output"],b["mapped_output"]]
    assert result["occurrence_contract"] == "mapped-publication-occurrences@1"
    assert _merge_publication_claim(result,b)==result

@pytest.mark.parametrize("change", ["producer","unknown","key","physical_map","source_port","target_port"])
def test_foreign_physical_authority_refused(change):
    a,b=claim("initial"),claim("predictor")
    (b["mapped_output"] if change in b["mapped_output"] else b)[change]="foreign"
    with pytest.raises(ValueError,match="competing"):_merge_publication_claim(a,b)

def test_same_point_new_invocation_refused():
    a=claim("initial");b=deepcopy(a);b["mapped_output"]["invocation"]="other"
    with pytest.raises(ValueError,match="same point"):_merge_publication_claim(a,b)


def test_frozen_resolved_claims_keep_nested_exact_authorities():
    from types import MappingProxyType
    def freeze(value):
        if isinstance(value,dict): return MappingProxyType({k:freeze(v) for k,v in value.items()})
        if isinstance(value,list): return tuple(freeze(v) for v in value)
        return value
    a,b=claim("initial"),claim("predictor")
    a["key"]={"owner":"same", "component":"phi"}; b["key"]=dict(a["key"])
    result=_merge_publication_claim(freeze(a),freeze(b))
    assert result == _merge_publication_claim(a,b)
    assert _merge_publication_claim(freeze(result),freeze(b)) == result
    wrong=deepcopy(b);wrong["mapped_output"]["source_port"]="foreign"
    with pytest.raises(ValueError,match="competing"):
        _merge_publication_claim(freeze(result),freeze(wrong))
