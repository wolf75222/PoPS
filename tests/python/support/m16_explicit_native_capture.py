"""Test-only retained compile/capture evidence; no code generation or backend substitute."""
import hashlib
import json
from pathlib import Path
import pops
from tests.python.support.atomic_native_capture import select_layout_program
from tests.python.support.collective_checks import collective_attempt,collective_call


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_retained_program(program,directory):
    # The genuine public dump_cpp has an advanced-handle regeneration fallback.
    # Require the compiler-owned retained text first, rather than exercise it.
    if type(program._generated_cpp) is not str or not program._generated_cpp:
        raise ValueError('compiler-retained Program C++ required; regeneration forbidden')
    program.dump_cpp(directory/'program.cpp')
    program.dump_ir(directory/'program.ir.json')


def retain_provenance(artifact,resolved,native,directory):
    row=select_layout_program(artifact,resolved)
    program=row.program
    dump_retained_program(program,directory)
    (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
    binaries=[]
    for block in artifact.blocks:
        path=Path(block.model.so_path).resolve()
        binaries.append({'block':block.name,'states':list(block.state_spaces),'path':str(path),'sha256':digest(path)})
    binary=Path(program.so_path).resolve()
    proof={'schema':'pops.m16-explicit-retained-provenance@1',
        'package':str(Path(pops.__file__).resolve()),
        'native':{'path':str(Path(native.__file__).resolve()),'sha256':digest(native.__file__)},
        'artifact_identity':artifact.artifact_identity.token,
        'layout_program':{'layout_id':row.layout_id,'target':row.target,'blocks':list(row.block_names),
                          'identity':row.identity.token},
        'program':{'path':str(binary),'sha256':digest(binary),'abi_key':program.abi_key,
                   'problem_hash':program.problem_hash,'cache_key':program.cache_key},
        'model_binaries':binaries,
        'files':{name:digest(directory/name) for name in ('program.cpp','program.ir.json','compiled-manifest.json')},
        'root_scientific_approval':False}
    (directory/'provenance.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
    return row


def persisted_capture(world,capture,save,on_failure):
    """Retain a complete capture immediately, or record actual failure without invented image."""
    original=None
    def invoke():
        nonlocal original
        try:return capture()
        except Exception as error:
            original=error
            raise
    image,failures=collective_attempt(world,invoke)
    if any(failures):
        _,persistence_failures=collective_attempt(world,lambda:on_failure(failures))
        if original is not None:
            original.add_note('capture evidence persistence failures: '+repr(persistence_failures))
            raise original
        raise RuntimeError('peer capture failed: '+repr(failures))
    collective_call(world,lambda:save(image))
    return image
