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


def exception_evidence(error):
    """JSON-safe actual exception chain, retaining explicit cause/context choice."""
    if error is None:return None
    rows=[];seen=set()
    while error is not None:
        if id(error) in seen:
            rows.append({'cycle':True});break
        seen.add(id(error))
        cause=error.__cause__
        link='cause' if cause is not None else 'context' if not error.__suppress_context__ and error.__context__ is not None else None
        rows.append({'type':type(error).__name__,'message':str(error),'next':link})
        error=cause if cause is not None else error.__context__ if link=='context' else None
    return rows


def execute_captured_bind(world,operation,on_failure,on_recorded=lambda:None):
    """Journal genuine bind failures collectively, then rethrow the local object."""
    from tests.python.support.collective_checks import collective_attempt
    original=None
    def invoke():
        nonlocal original
        try:return operation()
        except Exception as error:
            original=error;raise
    runtime,failures=collective_attempt(world,invoke)
    if not any(failures):return runtime
    _,io_failures=collective_attempt(world,lambda:on_failure(exception_evidence(original),failures))
    _,receipt_failures=collective_attempt(world,on_recorded)
    if original is not None:
        original.add_note('collective bind failures: '+repr(failures))
        if any(io_failures):original.add_note('bind evidence persistence failures: '+repr(io_failures))
        if any(receipt_failures):original.add_note('bind receipt persistence failures: '+repr(receipt_failures))
        raise original
    raise RuntimeError('peer bind failed: '+repr(failures)+'; persistence failures: '+repr((io_failures,receipt_failures)))


def compile_with_model_tus(resolved,directory,compile_artifact):
    """Publisher only: actual compilation and exact model-binary TU correspondence."""
    from tests.python.support.actual_compile_capture import capture_actual_compiles,sha
    with capture_actual_compiles(directory) as capture:
        artifact=compile_artifact(resolved)
        if type(artifact) is not CompiledSimulationArtifact:
            raise TypeError('exact CompiledSimulationArtifact required')
        artifact.verify()
        rows=[]
        for block in artifact.blocks:
            binary=block.model.so_path
            rows.append({'block':block.name,'binary_file':str(binary),'binary_sha256':sha(binary),
                'actual_compile':capture.require_binary(binary)})
        if not rows:raise ValueError('no actual model binary to authenticate')
        proof={'schema':'pops.atomic-actual-model-compiles@1',
            'artifact_identity_token':artifact.artifact_identity.token,'models':rows}
        (capture.directory/'model-binaries.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
    return artifact
