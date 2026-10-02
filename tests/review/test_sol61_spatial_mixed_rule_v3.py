"""SOURCE/offline admission adversaries; authentic failed campaign stays failed."""
import importlib.util
import math
from pathlib import Path
import struct
from copy import deepcopy
import pytest
import numpy as np
P=Path(__file__).with_name('sol61_evolved_stage_amr_spatial_reception_v3.py')
sp=importlib.util.spec_from_file_location('mixed_reader',P);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
RAW=Path('/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-nonconstant-n8-serial-dim2/pytest-tmp/test_public_evolved_stage_amr_0/evolved-stage-amr-spatial')
def triplet(norm=4.2112344066743426e-11,ref=.3630691185175441):
    return [('solve.'+name,value) for name,value in [('residual_norm',norm),('reference_residual_norm',ref),('rel_residual',norm/(ref if ref>0 else 1.))]]
def check(rows):return r._accept.selected_native_mixed_rule(rows,1e-10)
def test_actual_counterexample_preserves_raw_ratio():
    index=r.b.strict_json((RAW/'observation-index.json').read_bytes())
    rows=index['phases']['accepted']['metadata'][2]
    report=check(rows)
    assert report['rel_residual']==1.1598988159250038e-10>1e-10
    assert report['residual_norm']==4.2112344066743426e-11
    assert report['acceptance_ratio']<1e-10
@pytest.mark.parametrize('norm,ref',[(1.01e-10,.3630691185175441),(math.nextafter(2e-10,math.inf),2.),(math.nextafter(1e-10,math.inf),0.)])
def test_true_stopping_violations_refused(norm,ref):
    with pytest.raises(ValueError,match='stopping rule'):check(triplet(norm,ref))
@pytest.mark.parametrize('slot',range(3))
@pytest.mark.parametrize('bad',[True,1,float('nan'),float('inf'),-1.])
def test_invalid_diagnostic_scalars(slot,bad):
    rows=triplet();rows[slot]=(rows[slot][0],bad)
    with pytest.raises(ValueError):check(rows)
def test_inconsistent_one_ulp_ratio():
    rows=triplet();rows[-1]=(rows[-1][0],math.nextafter(rows[-1][1],math.inf))
    with pytest.raises(ValueError,match='inconsistent'):check(rows)
@pytest.mark.parametrize('mutation',['missing','duplicate','foreignstem'])
def test_closed_triplet(mutation):
    rows=triplet()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(rows[0])
    else:rows[-1]=('foreign.rel_residual',rows[-1][1])
    with pytest.raises(ValueError):check(rows)
def test_ref_zero_and_ref_large_at_boundary():
    assert check(triplet(1e-10,0.))['rel_residual']==1e-10
    assert check(triplet(2e-10,2.))['acceptance_ratio']==1e-10

def actual_images():
    index=r.b.strict_json((RAW/'observation-index.json').read_bytes())
    images={phase:[r.b.wire.archive(Path(row['path']).read_bytes()) for row in payload['levels']] for phase,payload in index['phases'].items()}
    for payload in index['phases'].values():
        for row in payload['levels']:assert r.b.digest(Path(row['path']).read_bytes())==row['sha256']
    cp=r.b.wire.archive((RAW/'accepted-checkpoint.npz').read_bytes())
    masks=r.b.topology(r.c.complete_carrier_geometry(cp),8,1,modes=[str(cp['distribution_mode_'+str(l)].item()) for l in range(2)],owners=[cp['dmap_'+str(l)] for l in range(2)])
    return images,masks

def test_authentic_original_F_and_nonlinear_adverse():
    images,masks=actual_images();metrics=r.science(images,masks,8)
    report=check(triplet())
    index=r.b.strict_json((RAW/'observation-index.json').read_bytes())
    for phase in ('accepted','continuous','reloaded','replay'):
        r.independent_original_norm(metrics[phase],check(index['phases'][phase]['metadata'][2]))
    assert metrics['accepted']['original_F_weighted_l2']==pytest.approx(4.211236792117065e-11,rel=1e-12)
    r.nonlinear_restriction_attacks(images,masks,8)

def test_original_F_recheck_refuses_below_old_Linf_guard():
    images,masks=actual_images()
    for phase in ('accepted','reloaded'):
        for row in images[phase]:row['z']=row['z']+2e-10
    for phase in ('continuous','replay'):
        for row in images[phase]:row['z-previous']=row['z-previous']+2e-10
    # Old 3e-8 science guards and exact reload equality still pass.
    metrics=r.science(images,masks,8)
    assert metrics['accepted']['original_F_weighted_l2']>1e-10
    with pytest.raises(ValueError,match='independent original-F'):r.independent_original_norm(metrics['accepted'],check(triplet()))

def test_historical_callback_and_ABI_profiles_immutable():
    assert r._historical_c.native_abi_receipt.__globals__['QUALIFICATION'].endswith('@3')
    assert r.c.checkpoint is not r._historical_c.checkpoint
    receipt=dict(schema='root.api040.native-abi@1',native={'path':'x','sha256':'y'},header_signature='real',module_abi_version=8,capability_abi_version=8,release_native_abi_version=8)
    r.native_abi_receipt(receipt,receipt['native'])
    for field in ('module_abi_version','capability_abi_version','release_native_abi_version'):
        for value in (6,7,True):
            changed=deepcopy(receipt);changed[field]=value
            with pytest.raises(ValueError):r.native_abi_receipt(changed,receipt['native'])

@pytest.mark.parametrize('phase',('accepted','continuous','replay'))
def test_authentic_CP12_body_through_versioned_callback(phase):
    images,_=actual_images();cp=r.b.wire.archive((RAW/(phase+'-checkpoint.npz')).read_bytes())
    ir=r.b.strict_json(next(RAW.glob('*.ir.json')).read_bytes())
    manifest=r.c.strict_json(str(cp['pops_checkpoint_manifest'].item()))
    identities=tuple(r.c.wire.identity_token(manifest[key+'_identity'],key) for key in ('artifact','bind','semantic'))
    _,_,parsed=r.checkpoint(cp,phase,images,8,2,1,identities,str(cp['abi_key'].item()),
        transfer_subjects=r.c.program_transfer_subjects(ir),history_registry=r.c.program_history_registry(ir))
    assert len(parsed)==1
    if phase=='accepted':
        with pytest.raises(ValueError,match='original-F diagnostic'):r._historical_c.checkpoint(cp,phase,images,8,2,1,identities,str(cp['abi_key'].item()),
            transfer_subjects=r.c.program_transfer_subjects(ir),history_registry=r.c.program_history_registry(ir))

def test_exact_fixture_support_pure_body_actual_saved_images():
    import ast
    import sys
    support=P.parents[1]/'python/support/evolved_stage_amr_spatial.py'
    tree=ast.parse(support.read_text())
    defs=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name!='build'],type_ignores=[])
    namespace=dict(np=np,DIFFUSION=r.D,accumulation=r.b.q_of,closed_data=r.b.declared_load,DT=.01,ACCEPTANCE=3e-8)
    exec(compile(defs,str(support),'exec'),namespace)
    images,masks=actual_images()
    for phase in ('accepted','continuous','reloaded','replay'):
        prev=None if phase in ('accepted','reloaded') else images['accepted']
        metrics,_=namespace['check_saved'](images[phase],8,2,prev,1 if prev is None else 2,images['initial'])
        assert metrics['original_F_weighted_l2']<1e-10
    assert 'pops' not in sys.modules

@pytest.mark.parametrize('header,release',[
 (b'inline constexpr int kReleaseNativeAbiVersion = 6;',b'NATIVE_ABI_VERSION = 8'),
 (b'inline constexpr int kReleaseNativeAbiVersion = 8;',b'NATIVE_ABI_VERSION = True'),
 (b'// inline constexpr int kReleaseNativeAbiVersion = 8;',b'NATIVE_ABI_VERSION = 8'),
 (b'inline constexpr int kReleaseNativeAbiVersion = 8;\ninline constexpr int kReleaseNativeAbiVersion = 8;',b'NATIVE_ABI_VERSION = 8'),
 (b'inline constexpr int kReleaseNativeAbiVersion = 8;',b'NATIVE_ABI_VERSION = 8\nNATIVE_ABI_VERSION = 6'),
])
def test_source_ABI_false_attestations_refused(header,release):
    with pytest.raises(ValueError):r.source_native_abi(header,release)

def test_source_ABI_positive_without_import():
    r.source_native_abi(b'inline constexpr int kReleaseNativeAbiVersion = 8;',b'NATIVE_ABI_VERSION = 8')

def test_private_checkpoint_changes_only_diagnostic_guard():
    import ast,inspect
    original=ast.parse(inspect.getsource(r._historical_c.checkpoint_current))
    loop=[node for node in ast.walk(original) if isinstance(node,ast.For) and isinstance(node.iter,ast.Name) and node.iter.id=='parsed'][0]
    loop.body=ast.parse('diagnostic_triplet(row)').body
    assert ast.dump(original)==ast.dump(r._tree)
    assert 'c' not in r.receive.__code__.co_varnames

@pytest.mark.parametrize('ref',(0.,.5,1.,10.))
def test_forged_consistent_reference_rejected_by_saved_seed(ref):
    images,masks=actual_images();metrics=r.science(images,masks,8)
    with pytest.raises(ValueError,match='zero-seed reference'):r.independent_original_norm(metrics['accepted'],check(triplet(ref=ref)))

def test_zero_seed_authority_actual_and_resealed_attacks():
    ir=r.b.strict_json(next(RAW.glob('*.ir.json')).read_bytes())
    identity=r.zero_seed_selection(ir)
    assert identity=='pops.solve-initialization.v1:sha256:b29f776ca5be8ebd1a543f97119f80783f49bb6be869ad4a407b1248ca74e1b1'
    for attack in ('index','seed','identity','missing'):
        bad=deepcopy(ir);attrs=next(n['attrs'] for n in bad['nodes'] if n['op']=='solve_spatial_field')
        if attack=='index':attrs['seed_index']=0
        elif attack=='seed':attrs['solve_request']['seed']={}
        elif attack=='identity':attrs['solve_request']['initialization_identity']=identity[:-1]+'2'
        else:del attrs['seed_index']
        with pytest.raises(ValueError):r.zero_seed_selection(bad)
