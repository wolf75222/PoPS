"""Pure Source validation of @2 profile and public permutation; no Native data."""
import pytest
from tests.review import sol61_m17_saved_physics_offline_v2 as r

def test_public_full_basis_is_not_fixed_to_canonical_order():
    assert r.public_order(r.v1.I)==r.v1.I
    assert r.public_order(tuple(reversed(r.v1.I)))==tuple(reversed(r.v1.I))
    permuted=r.v1.I[3:]+r.v1.I[:3]
    assert r.public_order(permuted)==permuted
    for bad in (list(r.v1.I),r.v1.I[:-1],(r.v1.I[0],)*15,((False,False),)+r.v1.I[1:]):
        with pytest.raises(ValueError):r.public_order(bad)

@pytest.mark.parametrize('mutation',('source','native','header','count','bool-count','production-count'))
def test_unattested_profile_drift_refused(mutation):
    identity={'source_commit':r.SOURCE,'native_sha256':r.NATIVE,'abi_key':'headers='+r.HEADER+';','verified_source_files':1144}
    source={'head':r.SOURCE,'files':{str(k):'SOURCE-only' for k in range(1188)}}
    r.identity_profile(identity,source)
    if mutation=='source':source['head']='0'*40
    elif mutation=='native':identity['native_sha256']='0'*64
    elif mutation=='header':identity['abi_key']='headers='+'0'*64+';'
    elif mutation=='count':identity['verified_source_files']=1144.0
    elif mutation=='bool-count':identity['verified_source_files']=True
    else:source['files'].pop('0')
    with pytest.raises(ValueError):r.identity_profile(identity,source)

def test_missing_external_pin_refuses_before_any_data(tmp_path):
    path=tmp_path/'not-root.json';path.write_text('{}')
    with pytest.raises(ValueError,match='external ROOT reception pin differs'):r.receive_export(path,'0'*64,r.v1.I)
