"""Inert metadata counter-tests, not native execution or fabricated model blocks."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from pops._platform_contracts import CapabilityProof,artifact_platform_manifest
from pops.codegen._artifact_models import aggregate_capability,program_mpi_capability
from pops.codegen._compiled_artifact import CompiledSimulationArtifact
from pops.codegen.inspect_compiled import _build_arguments


@pytest.mark.parametrize('communicator,expected',[
    ('MPI_COMM_WORLD',True),('serial',False),(None,None),('unknown_backend_communicator',None)])
def test_zero_block_reports_its_exact_platform_without_native_queries(monkeypatch,communicator,expected):
    # The inert object supplies only the already-sealed metadata read by the
    # report; it is never bound, installed, dlopened or claimed as a real artifact.
    platform=artifact_platform_manifest(backend='production',target='system',
        component=SimpleNamespace(abi_key='source-only-fixture'),communicator='serial')
    fact=CapabilityProof.unknown() if communicator is None else CapabilityProof.proven(communicator,'source-only-fixture')
    platform=replace(platform,communicator=fact)
    artifact=object.__new__(CompiledSimulationArtifact)
    object.__setattr__(artifact,'blocks',())
    object.__setattr__(artifact,'platform_manifest',platform)
    object.__setattr__(artifact,'plan',SimpleNamespace(target='system'))
    def forbidden(*args,**kwargs):raise AssertionError('inert report queried a native module')
    monkeypatch.setattr('pops._native_selector.selected_native_module',forbidden)
    assert program_mpi_capability(artifact) is expected
    assert aggregate_capability(artifact,'mpi') is expected
    assert aggregate_capability(artifact,'gpu') is None
    import pops.codegen.inspect_compiled as module
    for name in ('_parameter_arguments','_field_output_components_by_block','_auxiliary_packs_by_block','_ghost_depth_by_block'):
        monkeypatch.setattr(module,name,lambda *args:{})
    arguments=_build_arguments(artifact,None,(),program_name='arbitrary_unrecognized_equation')
    assert arguments.layout_runtime['supports_mpi'] is expected
    assert arguments.layout_runtime['ghost_depth']==0 and arguments.instances=={}
