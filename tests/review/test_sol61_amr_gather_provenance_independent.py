"""Independent synthetic provenance adversaries; no Native import."""
import hashlib,json
import pytest
from tests.review import sol61_amr_gather_engineering_saved_reader as r
from tests.review.test_sol61_amr_gather_engineering_saved_reader import test_explicit_provenance_join_synthetic_correspondence as prepare

@pytest.mark.parametrize('abi',('H|compiler|c++17|dim=2','H|compiler|c++20|dim=3','headers=H|compiler|c++20|dim=2','H||c++20|dim=2','H|compiler|c++20|dim=2|extra'))
def test_resealed_program_abi_mismatch_refused(tmp_path,abi):
    prepare(tmp_path,None)
    proof=json.loads((tmp_path/'provenance.json').read_text());proof['program']['abi_key']=abi
    manifest=json.loads((tmp_path/'compiled-manifest.json').read_text())
    manifest['payload']['abi_key']=abi
    (tmp_path/'compiled-manifest.json').write_text(json.dumps(manifest))
    proof['files']['compiled-manifest.json']=hashlib.sha256((tmp_path/'compiled-manifest.json').read_bytes()).hexdigest()
    (tmp_path/'provenance.json').write_text(json.dumps(proof))
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    exports={name:(tmp_path/(name+'.so'),pins[name+'.so']) for name in (*r.BLOCKS,'program')}
    with pytest.raises(ValueError,match='Program header signature'):
        r.join_provenance(tmp_path,pins,exports,native_sha256='N',header_signature='H')


@pytest.mark.parametrize('omitted',('program.cpp','program.ir.json','compiled-manifest.json'))
def test_resealed_missing_program_evidence_refused(tmp_path,omitted):
    prepare(tmp_path,None)
    proof=json.loads((tmp_path/'provenance.json').read_text());proof['files'].pop(omitted,None)
    (tmp_path/'provenance.json').write_text(json.dumps(proof))
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    exports={name:(tmp_path/(name+'.so'),pins[name+'.so']) for name in (*r.BLOCKS,'program')}
    with pytest.raises(ValueError):
        r.join_provenance(tmp_path,pins,exports,native_sha256='N',header_signature='H')


@pytest.mark.parametrize('mutation',('blocks','ghost-bool','schema-bool','gpu','headers'))
def test_resealed_manifest_authority_refused(tmp_path,mutation):
    prepare(tmp_path,None)
    manifest=json.loads((tmp_path/'compiled-manifest.json').read_text())
    if mutation=='blocks':manifest['payload']['blocks'].reverse()
    elif mutation=='ghost-bool':manifest['payload']['ghost_depth_by_block']['Q0']=True
    elif mutation=='schema-bool':manifest['schema_version']=True
    elif mutation=='gpu':manifest['payload']['supports_gpu']=True
    else:manifest['payload']['required_headers_sig']='foreign'
    (tmp_path/'compiled-manifest.json').write_text(json.dumps(manifest))
    proof=json.loads((tmp_path/'provenance.json').read_text());proof['files']['compiled-manifest.json']=hashlib.sha256((tmp_path/'compiled-manifest.json').read_bytes()).hexdigest()
    (tmp_path/'provenance.json').write_text(json.dumps(proof))
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    exports={name:(tmp_path/(name+'.so'),pins[name+'.so']) for name in (*r.BLOCKS,'program')}
    with pytest.raises(ValueError):r.join_provenance(tmp_path,pins,exports,native_sha256='N',header_signature='H')
