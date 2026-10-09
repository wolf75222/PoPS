"""Exact low-owner Program authentication and TemporalTau capability preservation."""
import copy
import pytest
from pops.time import Program,TimePoint,TemporalTau
from pops.time import _authoring
from pops.time._program import contract

def test_registry_reexports_are_exact_function_identity():
 assert contract.register_program_type is _authoring.register_program_type
 assert contract.require_program is _authoring.require_program
 assert _authoring.register_program_type(Program)is Program
 with pytest.raises(TypeError,match="Program authority must be a class"):_authoring.register_program_type(object())
 class Foreign:pass
 with pytest.raises(RuntimeError,match="already registered"):_authoring.register_program_type(Foreign)

def test_tau_issued_exact_program_and_foreign_point_refusal():
 p=Program("tau-owner");at=TimePoint(p.clock,0)
 tau=p.temporal_tau(at=at)
 assert type(tau)is TemporalTau and tau.prog is p and copy.deepcopy(tau)is tau
 tau.require_program(p,at=at)
 q=Program("tau-other")
 with pytest.raises(ValueError,match="another or relabelled"):tau.require_program(q)
 with pytest.raises(ValueError,match="evaluation point differs"):tau.require_program(p,at=TimePoint(p.clock,1))

def test_tau_rejects_duck_counterfeit_and_program_subclass_with_prior_error():
 p=Program("tau-canonical");at=TimePoint(p.clock,0)
 class Counterfeit:
  clock=p.clock
  _time_states=p._time_states
  owner_path=p.owner_path
 class Child(Program):pass
 for foreign in (Counterfeit(),Child("tau-subclass"),object()):
  with pytest.raises(TypeError)as caught:TemporalTau(foreign,p.dt,at=at)
  assert str(caught.value)=="TemporalTau requires an exact Program"
 assert _authoring.require_program(Child("tau-compatible"),exact=False,where="compat")is not None

def test_tau_rejects_foreign_clock_and_unissued_leaf():
 p=Program("tau-context");q=Program("tau-foreignclock")
 with pytest.raises(ValueError,match="no clock declared"):p.temporal_tau(at=TimePoint(q.clock,0))
 unissued=object.__new__(TemporalTau)
 with pytest.raises(KeyError):unissued.require_program(p)

def test_tau_single_live_low_registry_and_weakref_issuance():
 import gc,weakref
 from pops.time import evolved_field_stage
 assert not hasattr(evolved_field_stage,"_TAU_PROGRAMS")
 p=Program("tau-live-registry");tau=p.temporal_tau(at=TimePoint(p.clock))
 key=id(tau);leaf,owner=_authoring._TAU_PROGRAMS[key]
 assert leaf()is tau and owner()is p
 reference=weakref.ref(tau);del tau;gc.collect()
 assert reference()is None and key not in _authoring._TAU_PROGRAMS

def test_json_conversion_alias_identity_and_outputs_remain_exact():
 from pops.time import canonical_data
 from pops.time._program import serialization
 assert serialization._json_ready is canonical_data._json_ready
 samples=(None,1,{"nested":[1,2,{"x":"value"}]},("a",3))
 for value in samples:
  assert serialization._json_ready(value)==canonical_data._json_ready(value)
