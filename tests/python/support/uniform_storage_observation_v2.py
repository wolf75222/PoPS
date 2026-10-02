"""Test-only @2 persistence; no numerical operation or Native substitute."""
import hashlib,json
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_observation(directory,phase,rank,observation,values,clock):
    if phase not in ('first','second') or type(rank) is not int or rank<0:
        raise ValueError('exact observation phase and rank required')
    if clock != {'time':observation.time,'macro_step':observation.macro_step}:
        raise ValueError('observation and public owner clock differ')
    stem=phase+'.rank'+str(rank)
    names=(stem+'.local.bin',stem+'.complete.bin',stem+'.valid.npy',stem+'.json')
    if any((directory/name).exists() for name in names):raise ValueError('observation phase already persisted')
    (directory/names[0]).write_bytes(observation.rank_local)
    (directory/names[1]).write_bytes(observation.complete)
    np.save(directory/names[2],values,allow_pickle=False)
    proof={'schema':'pops.uniform-storage-observation-phase@2','phase':phase,'rank':rank,
           'contract':observation.contract,'clock':clock,'valid_shape':list(values.shape),
           'valid_dtype':values.dtype.str,'files':{name:digest(directory/name) for name in names[:3]},
           'scope':'readonly initial storage; no step or rollback','root_scientific_approval':False}
    (directory/names[3]).write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
    return {'receipt':names[3],'sha256':digest(directory/names[3])}

def retain_uniform_provenance(artifact,native,directory):
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from tests.python.support.m16_explicit_native_capture import dump_retained_program
    if type(artifact) is not CompiledSimulationArtifact:raise TypeError('exact compiled artifact required')
    artifact.verify()
    if len(artifact.layout_programs)!=1:raise ValueError('one exact Uniform partition required')
    row,=artifact.layout_programs;row.verify()
    if row.target!='system' or artifact.program is not row.program:
        raise ValueError('exact Uniform Program required')
    if set(row.block_names)!={block.name for block in artifact.blocks}:
        raise ValueError('full model partition required')
    dump_retained_program(row.program,directory)
    models=[]
    for block in artifact.blocks:
        evidence=block.model.source_provenance(require_complete=True)
        name=block.name+'.model.cpp';block.model.dump_cpp(directory/name)
        models.append({'block':block.name,'source':evidence,'file':name,'sha256':digest(directory/name),
                       'binary':str(Path(block.model.so_path).resolve()),'binary_sha256':digest(block.model.so_path)})
    proof={'schema':'pops.uniform-storage-provenance@2','artifact_identity':artifact.artifact_identity.token,
           'layout_id':row.layout_id,'layout_identity':row.identity.token,'target':row.target,
           'blocks':list(row.block_names),'native_path':str(Path(native.__file__).resolve()),
           'native_sha256':digest(native.__file__),'program_binary':str(Path(row.program.so_path).resolve()),
           'program_binary_sha256':digest(row.program.so_path),'models':models,
           'files':{name:digest(directory/name) for name in ('program.cpp','program.ir.json')},
           'root_scientific_approval':False}
    (directory/'provenance.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')


def authenticate_owner_shard(observation,rank,ranks):
    """Test-only actual wire validation, including all grown bits and empty owners."""
    from tests.review.sol61_amr_full_carrier_offline import decode
    if type(rank) is not int or type(ranks) is not int or not 0<=rank<ranks:
        raise ValueError('exact rank/world authority required')
    local=decode(np.frombuffer(observation.rank_local,dtype=np.uint8))
    complete=decode(np.frombuffer(observation.complete,dtype=np.uint8))
    if local['ranks']!=ranks or complete['ranks']!=ranks or local['shard']!=rank or complete['shard']!=-1:
        raise ValueError('rank/world/shard authority differs')
    for key in ('dim','real','levels','blocks'):
        if local[key]!=complete[key]:raise ValueError('local/complete authority differs')
    if local['dim']!=observation.dimension:raise ValueError('observation dimension differs')
    expected=[patch for patch in complete['patches'] if patch['owner'] in (-1,rank)]
    if local['patches']!=expected:raise ValueError('local/complete full-grown owner rows differ')
