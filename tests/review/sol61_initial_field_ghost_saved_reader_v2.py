"""Bounded offline initial Field/Ghost exports reader@1; no PoPS import.

Expected pins are externally supplied, never minted by this reader.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from tests.python.support.initial_field_ghost_native_oracle import check_observed, load_observed, checkpoint_primary_clock, DT, CONTRACT
from tests.review.sol61_amr_full_carrier_offline import decode


def require(ok, message):
    if not ok: raise ValueError(message)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def receive(directory, expected_pins):
    directory=Path(directory)
    require(type(expected_pins) is dict and bool(expected_pins), "external pins required")
    files={p.name for p in directory.iterdir() if p.is_file()}
    require(files == set(expected_pins), "complete export inventory differs")
    for name, digest in expected_pins.items():
        require(Path(name).name==name and type(digest) is str and len(digest)==64, "pin format differs")
        require(hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest, "external export pin differs")
    provenance=json.loads((directory/'provenance.json').read_text())
    checkpoint=Path(provenance['checkpoint']['path'])
    require(checkpoint.parent.resolve()==directory.resolve() and checkpoint.name in expected_pins, 'checkpoint pin authority differs')
    require(expected_pins[checkpoint.name]==provenance['checkpoint']['sha256'], 'checkpoint projection pin differs')
    expected_clock=checkpoint_primary_clock(checkpoint)
    require(provenance['contract']==CONTRACT and provenance['potential_accessor_may_materialize'] is True, "provider observation scope differs")
    signature=provenance['component_signature']['inferred_boundary_expression']
    require(signature['schema']=='inferred-boundary-expression-component@1' and len(signature['expressions'])==2 and len(signature['dependencies']['fields'])==1 and not signature['dependencies']['states'], "inferred dependency contract differs")
    snapshots={}; results={}
    halo=[['pops.amr.accepted-halo-preparation@1','candidate_accepted_clock','all_state_components','1','1']]
    for phase,steps in (('initial',0),('accepted',1),('reloaded',1)):
        blob=(directory/(phase+'-carriers.bin')).read_bytes()
        image=decode(np.frombuffer(blob,dtype=np.uint8))
        require(image['blocks']==['marker'] and image['shard']==-1, "carrier profile differs")
        valid=arrays(directory/(phase+'-valid.npz')); field=arrays(directory/(phase+'-fields.npz'))
        require(set(valid)==set(field)=={'0-0','0-1','1-0','1-1'}, "two-level array inventory differs")
        metadata=json.loads((directory/(phase+'-metadata.json')).read_text())
        require(metadata['phase']==phase and metadata['accepted_clock']==[steps*DT,steps] and metadata['accepted_halo_contract']==halo, "accepted clock/Halo differs")
        require(metadata['carrier_sha256']==hashlib.sha256(blob).hexdigest(), "carrier metadata pin differs")
        native=ast.literal_eval(metadata['native_metadata_repr'])
        require(native[2]==halo and list(native[-1][:2])==metadata['accepted_clock'], "native metadata projection differs")
        fields=[]
        for level,size in ((0,8),(1,16)):
            state=valid[f'{level}-0']; mask=valid[f'{level}-1']; phi=field[f'{level}-0']; fm=field[f'{level}-1']
            require(state.dtype==np.float64 and state.shape in ((size,size,2),(2,size,size)), "valid state shape differs")
            require(mask.dtype==np.bool_ and mask.shape==(size,size) and phi.dtype==np.float64 and phi.shape==mask.shape, "Field support differs")
            require(fm.dtype==mask.dtype and fm.shape==mask.shape and fm.tobytes()==mask.tobytes(), "Field mask differs")
            fields.append((phi,fm))
        for level,size in ((0,8),(1,16)):
            selected=[p for p in image['patches'] if p['key'][1]==level]
            boxes=metadata['writer_geometry_boxes'][level]
            require(len(selected)==len(boxes), "carrier geometry inventory differs")
            for patch,box in zip(selected,boxes,strict=True):
                (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
                require(list(box)==[xl,yl,xh+1,yh+1] and gxl<=xl-1 and gxh>=xh+1 and gyl<=yl-1 and gyh>=yh+1, "writer/grown geometry differs")
                values=np.asarray(patch['bits'],dtype=np.uint64).reshape(2,gyh-gyl+1,gxh-gxl+1)
                state=valid[f'{level}-0']
                canonical=state if state.shape==(2,size,size) else state.transpose(2,0,1)
                require(values[:,yl-gyl:yh-gyl+1,xl-gxl:xh-gxl+1].tobytes()==canonical.view(np.uint64)[:,yl:yh+1,xl:xh+1].tobytes(), "carrier/valid state projection differs")
        results[phase]=(check_observed(blob,[mask for phi,mask in fields],load_observed(directory,phase,image['ranks']),steps,expected_clock=expected_clock) if phase!='reloaded' else {'scope':'no new producer witness; restart State/Ghost/cache bits only'}) # Original composite F, FE, physical xmin/tangential-valid Ghost
        snapshots[phase]=(blob,valid,field,metadata['native_metadata_repr'])
    absence=json.loads((directory/'reloaded-field-candidate-absence.json').read_text())
    require(absence=={'schema':'pops.amr.field-candidate-observation@1','observations':[]}, 'restart producer witness was inherited or recreated')
    a,b=snapshots['accepted'],snapshots['reloaded']
    require(a[0]==b[0] and a[3]==b[3], "restart grown carrier/history/diagnostic/clock differs")
    for left,right in ((a[1],b[1]),(a[2],b[2])):
        for key in left:
            require(left[key].dtype==right[key].dtype and left[key].shape==right[key].shape and left[key].tobytes()==right[key].tobytes(), "restart valid/Field bits differ")
    checkpoint=Path(provenance['checkpoint']['path'])
    require(checkpoint.parent.resolve()==directory.resolve() and checkpoint.name in expected_pins, "checkpoint pin authority differs")
    require(expected_pins[checkpoint.name]==provenance['checkpoint']['sha256'], "checkpoint projection pin differs")
    cp=arrays(checkpoint)
    require(int(cp['pops_amr_checkpoint_version'])==12 and json.loads(str(cp['amr_accepted_contract']))['schema_version']==9, "checkpoint versions differ")
    require(cp['state_carriers_checkpoint'].dtype==np.uint8 and cp['state_carriers_checkpoint'].ndim==1 and cp['state_carriers_checkpoint'].tobytes()==a[0], "checkpoint full carrier differs")
    return {'reader':'sol61.initial-field-ghost.saved@2','native_authority':False,'root_scientific_approval':False,'post_accessor_Field_scope':'retained-stage-cache bits only','math':results,'restart_full_carrier_and_Field_bits':True}
