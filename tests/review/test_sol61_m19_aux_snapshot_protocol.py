"""Genuine Source wrapper routing; recording sink is explicitly not Native."""
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from pops.runtime._system_aux_state import _SystemAuxState
from pops.model.provider_pack import ComponentKey, build_operator_provider_pack
from tests.python.integration.runtime.test_m19_freestreaming_runtime import _velocity_snapshot, MODULE

class RecordingSink:
    def __init__(self):self.calls=[];self.values=np.arange(256.,dtype=float).reshape(32,8)
    def auxiliary_component(self,owner_qid,space_kind,space_name,component):
        self.calls.append((owner_qid,space_kind,space_name,component));return self.values

class Wrapper(_SystemAuxState):
    def __init__(self):self._s=RecordingSink()

def test_actual_wrapper_signature_and_ranked_native_result_unmodified():
    assert tuple(inspect.signature(_SystemAuxState.auxiliary_component).parameters)==("self","key")
    wrapper=Wrapper();runtime=SimpleNamespace(_executor=wrapper)
    value=_velocity_snapshot(runtime)
    flux=MODULE.operator_registry().get("flux_default")
    declared=next(iter(build_operator_provider_pack(MODULE,flux)))
    assert declared == ComponentKey(*wrapper._s.calls[0])
    assert value is wrapper._s.values
    assert wrapper._s.calls==[(str(MODULE.owner_path.canonical()),"aux","velocity_coordinate","velocity_coordinate")]

@pytest.mark.parametrize("key",("velocity_coordinate",{"component":"velocity_coordinate"},None))
def test_real_wrapper_rejects_incomplete_key_before_sink(key):
    wrapper=Wrapper()
    with pytest.raises(TypeError):wrapper.auxiliary_component(key)
    assert wrapper._s.calls==[]

def test_failed_four_argument_python_call_never_reaches_sink():
    wrapper=Wrapper()
    with pytest.raises(TypeError):wrapper.auxiliary_component("owner","aux","velocity_coordinate","velocity_coordinate")
    assert wrapper._s.calls==[]
