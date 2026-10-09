"""Observe actual accepted storage without refreshing or repairing any ghost."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.amr_snapshots import composite_active_mask
from tests.review.sol61_amr_full_carrier_offline import decode


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest())


def ghost_diagnostics(image):
    """Classify exact SoA bits by encoded valid/grown boxes; no runtime values invented."""
    archive=decode(np.frombuffer(image,dtype=np.uint8))
    rows=[]
    one=int(np.array([1.],dtype=np.float64).view(np.uint64)[0])
    for patch in archive["patches"]:
        axes=patch["axes"]
        shape=tuple(ghi-glo+1 for lo,hi,glo,ghi in reversed(axes))
        valid=np.zeros(shape,dtype=bool)
        slices=tuple(slice(lo-glo,hi-glo+1) for lo,hi,glo,ghi in reversed(axes))
        valid[slices]=True
        bits=np.asarray(patch["bits"],dtype=np.uint64).reshape(patch["components"],*shape)
        constant=bits[0];bad=constant!=one
        rows.append(dict(key=list(patch["key"]),owner=patch["owner"],axes=[list(a) for a in axes],components=patch["components"],
            valid_cells=int(valid.sum()),ghost_cells=int((~valid).sum()),
            constant_valid_nonone=int((bad & valid).sum()),constant_ghost_nonone=int((bad & ~valid).sum()),
            constant_ghost_zero=int(((constant==0) & ~valid).sum()),
            constant_nonone_offsets=np.flatnonzero(bad).tolist(),
            grown_bits_sha256=sha256(bits.tobytes()).hexdigest(),
            valid_bits_sha256=sha256(bits[:,valid].tobytes()).hexdigest(),
            ghost_bits_sha256=sha256(bits[:,~valid].tobytes()).hexdigest()))
    return rows


def capture(world,runtime,artifact,native,directory,phase,tag_contract,package_path):
    """All ranks enter public reads; rank0 persists before the strict guard can fail."""
    directory=Path(directory)
    image=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
    levels=collective_call(world,runtime.n_levels)
    values={}
    for level in range(levels):
        values[f"state_marker_{level}"]=np.asarray(collective_call(world,lambda level=level:runtime.block_level_state_global("marker",level)))
        values[f"active_{level}"]=np.asarray(collective_call(world,lambda level=level:composite_active_mask(runtime,level,refinement_ratio=2)))
    boxes=collective_call(world,lambda:tuple(runtime.patch_boxes()))
    clock=collective_call(world,lambda:[runtime.time(),runtime.macro_step()])
    with collective_check(world):
        if int(world.rank)==0:
            directory.mkdir(parents=True,exist_ok=True)
            blob=directory/(phase+"-state-carriers.bin")
            with blob.open("xb") as stream:stream.write(image)
            valid=directory/(phase+"-valid.npz")
            with valid.open("xb") as stream:np.savez(stream,**values)
            metadata=dict(schema="sol61.public-tag-phase-capture@1",phase=phase,rank=0,size=int(world.size),
                clock=clock,patch_boxes=[list(row) for row in boxes],tag_selection_contract=tag_contract,
                artifact_identity=artifact.artifact_identity.token,native_abi_version=native.__abi_version__,
                package=pin(package_path),native=pin(native.__file__),carriers=pin(blob),valid_npz=pin(valid),
                ghost_diagnostics=ghost_diagnostics(image),observation_only=True,ghost_refresh_performed=False)
            path=directory/(phase+"-metadata.json")
            with path.open("x") as stream:json.dump(metadata,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
            index=directory/(phase+"-index.json")
            with index.open("x") as stream:json.dump(dict(schema="sol61.public-tag-phase-index@1",metadata=pin(path),carriers=pin(blob),valid_npz=pin(valid)),stream,sort_keys=True,indent=2);stream.write("\n")
    return directory/(phase+"-index.json")
