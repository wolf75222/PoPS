"""Real Uniform1D ALE: original bodies, accepted nodes, saved inventory and restart.

Run against a freshly rebuilt native Dim1 SDK. No worker source test executes
this compiler/runtime fixture. POPS_ALE_RESOLUTIONS can select independent sizes.
"""
import json
import os
from pathlib import Path
import numpy as np
import pops
import pytest
from pops.output import NPZ, ParallelMode
from tests.python.unit.codegen.test_moving_interval_codegen import declared_case
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
RESOLUTIONS=tuple(int(value) for value in os.environ.get("POPS_ALE_RESOLUTIONS","16,32").split(","))
if len(set(RESOLUTIONS))<2 or any(value<4 for value in RESOLUTIONS):
    raise ValueError("ALE reception requires at least two distinct resolutions >=4")


def bind(artifact,initial):
    return pops.bind(artifact,initial_state={"fluid":np.ascontiguousarray(initial)},
                     resources={"execution_context":artifact_execution_context(artifact)})


@pytest.mark.parametrize("components",[("a",),("a","b","c"),("c","a","b")])
@pytest.mark.parametrize("with_source",[False,True])
def test_declared_moving_public_chain(isolated_native_cache,native_cxx,kokkos_root,tmp_path,
                                     components,with_source):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops.runtime_environment import runtime_environment_report
    mode=ParallelMode.SERIAL if runtime_environment_report()["communicator"]=="serial" else ParallelMode.ROOT
    evidence=[]
    for n in RESOLUTIONS:
        case,layout=declared_case(components=components,cells=n,with_source=with_source,output_mode=mode)
        artifact=pops.compile(pops.resolve(pops.validate(case),layout=layout)); artifact.verify()
        initial=np.stack([np.full(n,{"a":2.,"b":3.,"c":4.}[name]) for name in components])
        runtime=bind(artifact,initial)
        directory=tmp_path/(str(n)+"-"+"".join(components))
        pops.run(runtime,t_end=.002,max_steps=2,console=False,output_dir=directory)
        checkpoint=runtime.checkpoint(directory/"accepted-restart")
        accepted=np.asarray(runtime.state_global("fluid")).copy()
        mailbox=runtime._executor._checkpoint_program_exchanges()
        assert mailbox.startswith(b"POPSEX03")
        pops.run(runtime,t_end=.003,max_steps=1,console=False,output_dir=directory)
        final=np.asarray(runtime.state_global("fluid")).copy()
        final_mailbox=runtime._executor._checkpoint_program_exchanges()
        restored=bind(artifact,initial); restored.restart(checkpoint)
        np.testing.assert_array_equal(restored.state_global("fluid"),accepted)
        assert restored._executor._checkpoint_program_exchanges()==mailbox
        pops.run(restored,t_end=.003,max_steps=1,console=False,output_dir=directory/"replay")
        np.testing.assert_array_equal(restored.state_global("fluid"),final)
        assert restored._executor._checkpoint_program_exchanges()==final_mailbox
        # Read only scientific output, never substitute the coordinate law or
        # the runtime's own proposed volumes for its saved accepted geometry.
        saved=[]
        for path in sorted(directory.rglob("*.npz")):
            if "replay" in path.relative_to(directory).parts: continue
            with np.load(path,allow_pickle=False) as candidate:
                if "pops_output_manifest" not in candidate.files: continue
            reopened=NPZ(mode).reopen(path)
            datasets=reopened.manifest["datasets"]
            geometry=next(iter(datasets["geometries"].values()))
            nodes=reopened.arrays[geometry["node_coordinates"]][:,0]
            volumes=reopened.arrays[geometry["cell_volumes"]]
            np.testing.assert_array_equal(np.diff(nodes),volumes)
            field=next(iter(datasets["fields"].values()))
            values=np.concatenate([reopened.arrays[piece["name"]] for piece in field["pieces"]],axis=1)
            inventory=np.sum(values*volumes[None,:],axis=1)
            saved.append((path,inventory,nodes))
        from pops._native_collectives import rank
        if mode is ParallelMode.SERIAL or rank(artifact_execution_context(artifact).communicator.handle)==0:
            assert len(saved)==3,"each accepted interval must publish a real physical snapshot"
            for step,(_,inventory,nodes) in enumerate(saved,1):
                np.testing.assert_allclose(inventory,initial[:,0]*(1+.05*.001 if with_source else 1)**step,
                                           rtol=4e-14,atol=4e-14)
                assert np.max(np.abs(nodes-np.linspace(0,1,n+1)))>0
            if not with_source: np.testing.assert_allclose(final,initial,rtol=3e-14,atol=3e-14)
            evidence.append({"cells":n,"components":components,"source":with_source,
                "artifact":artifact.artifact_identity.token,"saved_inventory":saved[-1][1].tolist(),
                "checkpoint":str(checkpoint),"replay_exact":True})
    (tmp_path/"actual-moving-evidence.json").write_text(json.dumps(evidence,indent=2))
