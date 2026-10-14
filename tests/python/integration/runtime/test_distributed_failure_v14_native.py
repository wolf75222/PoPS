"""Complete distributed terminal payload, inspected without adopting its authorities."""
import hashlib
import json
import numpy as np
import pops
import pytest

from examples.api_v02.distributed_algebra_v14 import build_distributed_algebra
from tests.python.integration.runtime.test_api_v02_distributed_program_native import _failure_shards
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _inspect_image(data):
    """Independent bounded parser for this scalar/alias POPSCONT2 fixture only."""
    cursor = 0
    def word():
        nonlocal cursor
        assert len(data)-cursor >= 8
        value=int.from_bytes(data[cursor:cursor+8],"little")
        cursor+=8
        return value
    def blob():
        nonlocal cursor
        size=word()
        assert size <= len(data)-cursor
        result=data[cursor:cursor+size]
        cursor+=size
        return result
    def text():
        return blob().decode("utf-8")
    assert text()=="POPSCONT2"
    schema=text()
    assert word()==2 and word()==0
    assert word()==0, "this fixture contains no replicated/local finite realms"
    assert word()==0, "this fixture contains no replicated/local finite domains"
    domain_count=word()
    assert domain_count==1
    issuer=(word(),word());storage=(word(),word())
    domains=[]
    for _ in range(domain_count):
        version,family,capacity=word(),word(),word()
        count=word()
        assert count in (3,4)
        domains.append((version,family,capacity,tuple(word() for _ in range(count))))
    def value():
        kind=word()
        if kind in (1,9,11):return (kind,word())
        if kind==10:return (kind,word(),word())
        if kind==4:
            arm=word();assert word()==1
            return (kind,arm,(value(),))
        assert kind==3, "fixture parser accepts only Int64, Product/Variant and exact distributed refs"
        size=word();assert size<=8
        return (kind,tuple(value() for _ in range(size)))
    array_count=word();assert array_count in (2,3)
    arrays=[]
    for _ in range(array_count):
        version=word();element_schema=blob();domain_slot=word();index_target=word();charge=word()
        # Transport identities contain integer words and may include non-UTF8 bytes.
        marker=b"pops.continuation-transport-type@2"
        assert element_schema[:8]==len(marker).to_bytes(8,"little")
        assert element_schema[8:8+len(marker)]==marker
        size=word();assert size<=4
        arrays.append(dict(version=version,schema=element_schema,domain=domain_slot,
                           target=index_target,charge=charge,values=tuple(value() for _ in range(size))))
    root=value()
    assert cursor==len(data)
    return dict(schema=schema,issuer=issuer,storage=storage,domains=domains,arrays=arrays,root=root)


@pytest.mark.parametrize("kind", ("terminal", "terminal_sparse"))
def test_distributed_terminal_failure_keeps_all_shards_and_aliases(kind, isolated_native_cache,
        native_cxx, kokkos_root, record_property, tmp_path):
    del isolated_native_cache, native_cxx, kokkos_root
    _,world=native_profile(dimension=2,scope="distributed terminal Failure v14")
    assert world is not None and int(world.size) in (2,3)
    spec=collective_call(world,lambda:build_distributed_algebra(kind,ranks=int(world.size)))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact=compile_profile_plan_once(world,plan,route="distributed-"+kind+"-v14-mpi"+str(world.size),
                                     compile_artifact=pops.compile)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=dict(spec.initial_values),
        resources={"execution_context":artifact_execution_context(artifact)}))
    before=collective_call(world,lambda:runtime.state_global("method"))
    error=None
    try:pops.run(runtime,t_end=spec.t_end,max_steps=1,console=False)
    except Exception as caught:error=caught
    after=collective_call(world,lambda:runtime.state_global("method"))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error,StepAttemptRejected),repr(error)
        shards=_failure_shards(error.typed_failure_payload)
        assert len(shards)==int(world.size)
        images=[_inspect_image(shard) for shard in shards]
        offsets=(0,1,4) if int(world.size)==2 else (0,1,1,4)
        for rank,item in enumerate(images):
            assert item["schema"]==error.typed_failure_schema
            assert item["domains"][0][2:]==(4,offsets)
            assert item["issuer"]==images[0]["issuer"]
            assert item["storage"]==(8*1024*1024,128)
            source,indices=item["arrays"][:2]
            assert (source["version"],indices["version"])==(1,2)
            assert source["domain"]==indices["domain"]==0
            assert source["target"]==0 and indices["target"]==1
            assert source["charge"]>0 and indices["charge"]>0
            gids=range(offsets[rank],offsets[rank+1])
            assert source["values"]==tuple((1,gid+1) for gid in gids)
            assert indices["values"]==tuple((10,0,(3*gid+1)%4) for gid in gids)
            if kind=="terminal":
                assert len(item["arrays"])==2
                assert item["root"]==(3,((3,((11,0),(11,0))),(9,0),(11,1),(11,0)))
            else:
                assert len(item["arrays"])==3
                container=item["arrays"][2]
                assert container["version"]==3 and container["domain"]==0 and container["target"]==0
                def expected(gid):
                    return ((4,1,((3,((11,1),(11,0),(11,0))),)) if gid==3
                            else (4,0,((1,gid),)))
                assert container["values"]==tuple(expected(gid) for gid in gids)
                assert item["root"]==(3,((11,2),))
        assert runtime.time()==0. and runtime.macro_step()==0
        if before is not None:np.testing.assert_array_equal(after,before)
        (tmp_path/"distributed-terminal-v14.bin").write_bytes(error.typed_failure_payload)
        record_property("mpi_ranks",int(world.size))
        record_property("artifact_identity",artifact.artifact_identity.hexdigest)
        record_property("typed_failure_sha256",hashlib.sha256(error.typed_failure_payload).hexdigest())
        record_property("typed_failure_shards",len(shards))
        record_property("alias_table",json.dumps([[(a["version"],len(a["values"])) for a in i["arrays"]] for i in images]))
