"""Immediate State-only evidence capture for the installed CP9 witness."""
import json
from pathlib import Path
import numpy as np
from tests.python.support.evolved_stage_v_capture import pin,save_json,retain_model_sources
from tests.python.support.m16_explicit_native_capture import dump_retained_program


def retain_provenance(artifact,directory):
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.codegen.compile_provenance import artifact_sidecar_path
    if type(artifact) is not CompiledSimulationArtifact:raise TypeError('exact compiled artifact required')
    artifact.verify();rows=tuple(artifact.layout_programs)
    if len(rows)!=1:raise ValueError('one Uniform Program partition required')
    row,=rows;row.verify()
    names=tuple(block.name for block in artifact.blocks)
    if artifact.program is not row.program or row.target!='system' or len(set(row.block_names))!=len(row.block_names) or set(row.block_names)!=set(names):raise ValueError('full Uniform partition differs')
    root=Path(directory)/'provenance';root.mkdir()
    models=[retain_model_sources(block.model,root,index,block.name) for index,block in enumerate(artifact.blocks)]
    program=row.program;folder=root/'program';folder.mkdir()
    dump_retained_program(program,folder)
    binary=folder/'program.so';binary.write_bytes(Path(program.so_path).read_bytes())
    sidecar=folder/'program.artifact.json';sidecar.write_bytes(Path(artifact_sidecar_path(program.so_path)).read_bytes())
    companions={}
    for source in Path(program.so_path).parent.glob(Path(program.so_path).name+'.*'):
        if source.is_file() and not source.name.endswith('.lock'):
            destination=folder/source.name;destination.write_bytes(source.read_bytes());companions[source.name]=pin(destination)
    manifest=save_json(root/'compiled-manifest.json',artifact.manifest().to_dict())
    return save_json(root/'receipt.json',{'schema':'pops.uniform-state-checkpoint-provenance@2','before_bind':True,
        'artifact_identity':artifact.artifact_identity.token,'blocks':list(names),'models':models,'manifest':manifest,
        'program':{'layout_identity':row.identity.token,'block_names':list(row.block_names),'DSO':pin(binary),
                   'sidecar':pin(sidecar),'cpp':pin(folder/'program.cpp'),'ir':pin(folder/'program.ir.json'),'companions':companions},'root_approval':False})


def persist_phase(directory,label,rank,ranks,observation,clock,values):
    """Persist raw typed data before validation; no synthetic carrier/value image."""
    root=Path(directory);files={}
    def blob(name,value):
        path=root/name;path.write_bytes(value);files[name]=pin(path)
    blob(label+'.rank%d.carriers'%rank,observation.rank_local)
    # Each participant retains its actual complete image for all-rank equality audit.
    blob(label+'.rank%d.complete.carriers'%rank,observation.complete)
    metadata={'schema':'pops.uniform-state-checkpoint-phase@2','phase':label,'rank':rank,'ranks':ranks,
              'contract':observation.contract,'capture_complete':bool(values),'dimension':observation.dimension,'time':observation.time,'macro_step':observation.macro_step,'runtime_clock':list(clock),
              'blocks':list(values),'files':files}
    for index,(name,value) in enumerate(values.items()):
        filename=label+'.rank%d.block%d.npy'%(rank,index);np.save(root/filename,value,allow_pickle=False)
        files[filename]=pin(root/filename)
    if rank==0:
        blob(label+'.carriers',observation.complete)
        save_json(root/(label+'.clock.json'),list(clock))
    return save_json(root/(label+'.rank%d.phase.json'%rank),metadata)


def validate_phase(observation,clock,values):
    from tests.review.sol61_amr_full_carrier_offline import decode
    image=decode(np.frombuffer(observation.complete,dtype=np.uint8).copy())
    if image['shard']!=-1 or image['levels']!=1 or image['real']!=64 or image['dim']!=observation.dimension or image['blocks']!=list(values):raise ValueError('complete phase authority differs')
    local=decode(np.frombuffer(observation.rank_local,dtype=np.uint8).copy())
    if tuple(local[k] for k in ('dim','real','ranks','levels','blocks'))!=tuple(image[k] for k in ('dim','real','ranks','levels','blocks')) or local['patches']!=[p for p in image['patches'] if p['owner'] in (-1,local['shard'])]:raise ValueError('local/complete grown bits differ')
    if (observation.time,observation.macro_step)!=tuple(clock):raise ValueError('observation/runtime clock differs')
    for index,(name,value) in enumerate(values.items()):
        if type(value) is not np.ndarray or value.dtype!=np.float64:raise ValueError('valid dtype differs')
        covered=np.zeros(value.shape[1:],dtype=np.uint8)
        for row in image['patches']:
            if row['key'][0]!=index:continue
            axes=row['axes'];shape=tuple(axis[3]-axis[2]+1 for axis in reversed(axes))
            grown=np.asarray(row['bits'],dtype=np.uint64).reshape((row['components'],)+shape)
            source=(slice(None),)+tuple(slice(axis[0]-axis[2],axis[1]-axis[2]+1) for axis in reversed(axes))
            target=(slice(None),)+tuple(slice(axis[0],axis[1]+1) for axis in reversed(axes))
            if not np.array_equal(grown[source],value.view(np.uint64)[target]):raise ValueError('complete/valid bits differ')
            covered[target[1:]]+=1
        if not np.all(covered==1):raise ValueError('valid coverage differs')
