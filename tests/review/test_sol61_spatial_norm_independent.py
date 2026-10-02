"""Independent SOURCE-only adversaries; no receipt or approval creation."""
import importlib.util
import math
from pathlib import Path
import pytest
P=Path(__file__).with_name('sol61_evolved_stage_amr_spatial_reception_v3.py')
s=importlib.util.spec_from_file_location('norm_independent',P);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def report(norm,ref):
 return r._accept.selected_native_mixed_rule([('x.'+k,v) for k,v in
 [('residual_norm',norm),('reference_residual_norm',ref),('rel_residual',norm/(ref if ref>0 else 1.))]],1e-10)
@pytest.mark.parametrize('ref',[2.,1e100,float.fromhex('0x1.fffffffffffffp+1023')])
def test_spoof_large_native_reference_cannot_bypass_small_independent_seed(ref):
 native=report(2e-10,ref)
 with pytest.raises(ValueError):
  r.independent_original_norm({'original_F_weighted_l2':2e-10,'original_F_zero_seed_reference_l2':.363,'original_F_reference_terms':64,'original_F_reference_absolute_scale_l2':.4},native)
@pytest.mark.parametrize('ref',[0.,.5,1.])
def test_mixed_gate_keeps_absolute_boundary_when_reference_below_one(ref):
 assert report(1e-10,ref)['acceptance_ratio']==1e-10
 with pytest.raises(ValueError): report(math.nextafter(1e-10,math.inf),ref)
@pytest.mark.parametrize('version',[9,10,2**63])
def test_future_abi_not_implicitly_accepted(version):
 receipt=dict(schema='root.api040.native-abi@1',native={'path':'x','sha256':'y'},header_signature='s',module_abi_version=version,capability_abi_version=version,release_native_abi_version=version)
 with pytest.raises(ValueError):r.native_abi_receipt(receipt,receipt['native'])
def test_historical_namespace_gate_still_refuses_eight():
 receipt=dict(schema='root.api040.native-abi@1',native={'path':'x','sha256':'y'},header_signature='s',module_abi_version=8,capability_abi_version=8,release_native_abi_version=8)
 with pytest.raises(ValueError):r._historical_c.native_abi_receipt(receipt,receipt['native'])

def test_subnormal_reference_overflow_raw_ratio_is_refused():
 with pytest.raises(ValueError,match='invalid original-F diagnostic'):
  report(1e-10,math.nextafter(0.,1.))

def test_real_archived_small_reference_matches_derived_bound_and_zero_report_adversary():
 from copy import deepcopy
 import importlib.util
 spec=importlib.util.spec_from_file_location('real_norm_fixture',P.with_name('test_sol61_spatial_mixed_rule_v3.py'))
 t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 images,masks=t.actual_images();metrics=r.science(images,masks,8)['accepted']
 r.independent_original_norm(metrics,t.check(t.triplet()),t.arithmetic())
 # Diagnostic authenticity adversary: final zero cannot describe nonzero saved F.
 # Corrective author freeze must close this previously failing adversary.
 with pytest.raises(ValueError,match='original-F'):
  r.independent_original_norm(metrics,report(0.,metrics['original_F_zero_seed_reference_l2']),t.arithmetic())

@pytest.mark.parametrize('claimed',[0.,1e-10])
def test_in_range_false_final_norm_refused_with_real_arithmetic(claimed):
 spec=importlib.util.spec_from_file_location('auth_arithmetic',P.with_name('test_sol61_spatial_mixed_rule_v3.py'))
 t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 images,masks=t.actual_images();metrics=r.science(images,masks,8)['accepted']
 with pytest.raises(ValueError,match='native original-F norm differs'):
  r.independent_original_norm(metrics,report(claimed,metrics['original_F_zero_seed_reference_l2']),t.arithmetic())
def test_arithmetic_certificate_missing_refused_on_real_good_state():
 spec=importlib.util.spec_from_file_location('no_cert',P.with_name('test_sol61_spatial_mixed_rule_v3.py'))
 t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 images,masks=t.actual_images();metrics=r.science(images,masks,8)['accepted']
 with pytest.raises(ValueError,match='arithmetic proof'):
  r.independent_original_norm(metrics,t.check(t.triplet()))
