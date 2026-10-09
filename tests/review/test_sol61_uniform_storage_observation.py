"""Source limits; native execution belongs to the installed integration fixture."""
import dataclasses, struct
import pytest
from pops.runtime._state_storage_observation import AcceptedStateStorageObservation as Observation

def record(dim=2):
    def wire(shard):return b"POPSCAR1"+struct.pack("<QQQqQQ",dim,64,2,shard,1,2)
    return dict(contract="accepted-state-storage-observation@1",dimension=dim,time=7.5,macro_step=3,rank_local=wire(1),complete=wire(-1))

@pytest.mark.parametrize("dim",[1,2,3])
def test_typed_immutable_ranked_result(dim):
    result=Observation.from_native(record(dim))
    assert result.time==7.5 and result.macro_step==3
    with pytest.raises(dataclasses.FrozenInstanceError):result.time=0.

@pytest.mark.parametrize("key,value",[("dimension",True),("time",0),("time",float("nan")),("macro_step",True),("macro_step",-1),("contract","accepted-state-storage-observation@2"),("rank_local",b"POPSCAR1"),("complete",bytearray(b"POPSCAR1"))])
def test_malformed_record_fails_closed(key,value):
    data=record();data[key]=value
    with pytest.raises((ValueError,TypeError)):Observation.from_native(data)

def test_shard_cannot_claim_complete_or_foreign_rank():
    for shard in (0,2):
        data=record();data["complete"]=b"POPSCAR1"+struct.pack("<QQQqQQ",2,64,2,shard,1,2)
        with pytest.raises(ValueError):Observation.from_native(data)

def test_public_facade_uses_explicit_executor_route_without_private_script_access():
    from pops.runtime._runtime_instance import RuntimeInstance
    class Executor:
        calls=0
        def observe_accepted_state_storage(self):self.calls+=1;return Observation.from_native(record())
    class Runtime: _executor=Executor()
    value=RuntimeInstance.observe_accepted_state_storage(Runtime())
    assert value.macro_step==3 and Runtime._executor.calls==1


def test_actual_source_origin_without_extension_native():
    import importlib.machinery,sys,pops
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve()==root/"python/pops/__init__.py"
    assert not any(isinstance(getattr(getattr(module,"__spec__",None),"loader",None),importlib.machinery.ExtensionFileLoader)
                   for name,module in sys.modules.items() if name.rsplit(".",1)[-1].startswith("_pops"))


@pytest.mark.parametrize("route",["direct","from_native"])
@pytest.mark.parametrize("kind",["custom_equality","str_subclass"])
def test_contract_cannot_retain_custom_equality_or_string_subclass(route,kind):
    class EqualContract:
        def __eq__(self,other):return True
        def __ne__(self,other):return False
    class DerivedString(str):pass
    data=record()
    data["contract"]=(EqualContract() if kind=="custom_equality" else DerivedString(data["contract"]))
    with pytest.raises(TypeError,match="exact str"):
        if route=="direct":Observation(**data)
        else:Observation.from_native(data)
