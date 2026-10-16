"""True Cartesian-face T12; NumPy is solely an independent reception oracle."""
import numpy as np
import pytest

from examples.api_v02.method_flux_correction import authored_flux, build_t12_meshface, reference_step
from pops.identity import make_identity
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile, profile_gather_value


pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize("cfl",(.25,.4))
def test_native_two_incident_face_replacement(cfl,isolated_native_cache,native_cxx,kokkos_root,
                                            record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="T12 native mesh-face correction")
    runtime,artifact,metrics=execute_case(lambda:build_t12_meshface(cfl=cfl),world,
                                  "T12 native mesh-face correction CFL="+str(cfl))
    local=collective_call(world,runtime._executor._program_exchange_records)
    shards=profile_gather_value(world,local)
    with collective_check(world):
        # A ledger may expose shards or collective replicas. Conflicting replicas
        # are errors; each oriented cell/face incidence counts exactly once.
        unique={}
        for shard in shards:
            for row in shard:
                key=tuple(row[name] for name in ("operation_identity","occurrence_identity",
                                                "evaluation_context","quadrature_identity"))
                assert key not in unique or unique[key]==row
                unique[key]=row
        assert len(unique)==32*4*4
        final_identity=make_identity("stencil-flux-v1",authored_flux(corrected=True).to_data()).token
        assert {row["operation_identity"] for row in unique.values()}=={final_identity}
        assert len({row["evaluation_context"] for row in unique.values()})==1
        initial=np.full((1,4,32),.15)
        initial[...,6:12]=[.15,.06,.005,1.,.65,.3]
        dt=cfl/32
        reference=reference_step(initial,dt=dt,dx=1/32)
        increments=np.zeros_like(initial)
        physical_faces={}
        for row in unique.values():
            cell,axis_token,side_token=row["quadrature_identity"].split("/")
            i,j=map(int,cell.split(":")[1:])
            axis,side=int(axis_token.split(":")[1]),int(side_token.split(":")[1])
            assert row["orientation"]==(1 if side==0 else -1)
            assert row["multiplicity"]==1 and row["temporal_weight"]==dt
            assert row["face_measure"]==(1/4 if axis==0 else 1/32)
            expected=reference["faces"][0,j,(i+side)%32] if axis==0 else 0.
            assert abs(row["numerical_flux"]-expected)<3.e-14
            assert row["integrated_amount"]==row["orientation"]*row["face_measure"]*row["numerical_flux"]*dt
            increments[0,j,i]+=row["integrated_amount"]
            face_key=(axis,(i+side)%32,j) if axis==0 else (axis,i,(j+side)%4)
            physical_faces.setdefault(face_key,[]).append(row["integrated_amount"])
        assert all(len(amounts)==2 and amounts[0]==-amounts[1]
                   for amounts in physical_faces.values())
        np.testing.assert_allclose(increments,(reference["corrected"]-initial)/(32*4),
                                   rtol=0,atol=3.e-17)
        record_property("accepted_oriented_incidences",len(unique))
        record_property("accepted_physical_faces",len(physical_faces))
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("metrics",str(metrics))
