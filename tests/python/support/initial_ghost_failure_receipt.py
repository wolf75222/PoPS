"""Test-only early bind diagnostic; does not manufacture runtime snapshots."""
import json
from pathlib import Path

def save_early_bind_receipt(directory, *, rank, world_size, failures, owner_count, image_count, target_count, native, artifact_identity, component_manifest):
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('invalid early bind rank authority')
    if type(failures) is not list or len(failures) != world_size:
        raise ValueError('early bind failures must contain every rank')
    for failure in failures:
        if failure is not None and (type(failure) not in (tuple,list) or len(failure)!=3 or type(failure[0]) is not str or type(failure[1]) is not str or type(failure[2]) is not bool):
            raise ValueError('malformed observed collective failure')
    if any(type(value) is not int or value < 0 for value in (owner_count,image_count,target_count)):
        raise ValueError('invalid observed early bind counts')
    if type(native) is not dict or set(native)!= {'path','sha256'} or any(type(v) is not str or not v for v in native.values()):
        raise ValueError('invalid Native provenance')
    if type(artifact_identity) is not str or not artifact_identity or type(component_manifest) is not dict:
        raise ValueError('invalid compiled component provenance')
    directory=Path(directory)
    log=directory/f'callback-rank{rank}.log'
    receipt={'schema':'sol61.initial-ghost-early-bind-diagnostic@1','scope':'diagnostic only; no rollback or image certification','rank':rank,'world_size':world_size,'failures':failures,'observed_counts':{'owners':owner_count,'images':image_count,'targets':target_count},'callback_log':log.read_text() if log.exists() else None,'native':dict(native),'artifact_identity':artifact_identity,'component_manifest':component_manifest}
    path=directory/f'early-bind-rank{rank}.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    return path
