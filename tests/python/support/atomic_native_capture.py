"""Test-only capture helpers; authenticate the real compile-phase wrapper."""
import json
import numpy as np
from pops.codegen._compiled_artifact import CompiledSimulationArtifact


def select_layout_program(artifact,resolved):
    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError('exact CompiledSimulationArtifact required')
    artifact.verify()
    ids={row.layout.qualified_id for row in resolved.layout_plan.assignments
         if row.subject_kind=='block' and row.subject.local_id=='population'}
    if len(ids)!=1 or len(resolved.layout_plan.layouts)!=1:
        raise ValueError('fixture requires one exact population layout')
    matches=tuple(row for row in artifact.layout_programs if row.layout_id in ids)
    if len(matches)!=1 or len(artifact.layout_programs)!=1:
        raise ValueError('compiled layout program is absent or ambiguous')
    row,=matches
    row.verify()
    if row.target!='amr_system' or row.block_names!=('population',) or artifact.program is not row.program:
        raise ValueError('compiled layout target/partition/program differs')
    return row


def save_phase(directory,phase,image):
    values,carriers,clock=image
    np.save(directory/(phase+'.npy'),values,allow_pickle=False)
    (directory/(phase+'.carriers')).write_bytes(carriers)
    proof={'schema':'pops.atomic-native-captured-phase@1','phase':phase,'clock':clock,
           'capture_status':'captured','science_assertions':'not-yet-run'}
    (directory/(phase+'.capture.json')).write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')


def execute_captured_step(world,operation,capture,on_attempt,on_capture):
    """Save failure evidence and rollback image, preserving the actual exception.

    Callbacks are test orchestration only; this is not a solver/backend substitute.
    A capture may itself contain native collective calls and therefore must be
    entered by every rank in the same order.
    """
    from tests.python.support.collective_checks import collective_attempt
    original=None
    def invoke():
        nonlocal original
        try:return operation()
        except Exception as error:
            original=error
            raise
    report,failures=collective_attempt(world,invoke)
    _,persistence_failures=collective_attempt(world,lambda:on_attempt(failures))
    if any(failures):
        image,capture_failures=collective_attempt(world,capture)
        if not any(capture_failures):
            _,persistence_failures=collective_attempt(world,lambda:on_capture(image,True))
        if original is not None:
            original.add_note('collective failures: '+repr(failures))
            original.add_note('capture/persistence failures: '+repr((capture_failures,persistence_failures)))
            raise original
        raise RuntimeError('peer native run failed: '+repr(failures))
    assert not any(persistence_failures),persistence_failures
    image,capture_failures=collective_attempt(world,capture)
    assert not any(capture_failures),capture_failures
    _,persistence_failures=collective_attempt(world,lambda:on_capture(image,False))
    assert not any(persistence_failures),persistence_failures
    return report,image


def layout_program_json_identity(row):
    """Explicit JSON identity token; raw Identity.to_data contains digest bytes."""
    row.verify()
    return {'layout_id':row.layout_id,'target':row.target,'block_names':list(row.block_names),
            'identity_token':row.identity.token}
