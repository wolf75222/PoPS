"""Explicit pinned Native export audit; no native execution by these tests."""
import json,os
from pathlib import Path
from tests.review.sol61_initial_field_ghost_saved_reader_v4 import receive


def test_actual_typed_native_export_passes_scoped_reader():
    authority=json.loads(Path(os.environ['SOL61_TYPED_EXPORT_PINS']).read_text())
    assert len(authority['pins'])==int(os.environ.get('SOL61_TYPED_EXPECTED_FILES','28'))
    blob=(Path(authority['case_directory'])/'initial-carriers.bin').read_bytes()
    assert decode(np.frombuffer(blob,dtype=np.uint8))['ranks']==int(os.environ.get('SOL61_TYPED_EXPECTED_RANKS','1'))
    result=receive(authority['case_directory'],authority['pins'])
    assert result['restart_full_carrier_and_Field_bits'] is True
    assert result['native_authority'] is False and result['root_scientific_approval'] is False


# All mutations below operate only on temporary copies with synthetic test pins.
# They cannot grant authority to modified data or change the received originals.
import shutil,hashlib
import numpy as np
import pytest
from tests.review.sol61_amr_full_carrier_offline import decode
from tests.review.test_sol61_field_candidate_oracle_independent import encode

@pytest.mark.parametrize('mutation',('tick','grown_strip','signature_string','field_dependency','missing_invocation','writer_transposed'))
def test_copied_actual_export_mutations_are_rejected(tmp_path,mutation):
    authority=json.loads(Path(os.environ['SOL61_TYPED_EXPORT_PINS']).read_text())
    original=Path(authority['case_directory']);directory=tmp_path/'copied';directory.mkdir()
    for name in authority['pins']:shutil.copyfile(original/name,directory/name)
    provenance=json.loads((directory/'provenance.json').read_text())
    provenance['checkpoint']['path']=str(directory/Path(provenance['checkpoint']['path']).name)
    ranks=int(os.environ.get('SOL61_TYPED_EXPECTED_RANKS','1'))
    for rank in range(ranks):
        for phase in ('initial','accepted'):
            file=directory/f'{phase}-field-candidate-rank{rank}.json'
            doc=json.loads(file.read_text())
            for row in doc['observations']:
                row['carrier']['path']=str(directory/Path(row['carrier']['path']).name)
                if phase=='accepted' and mutation=='tick' and rank==ranks-1:row['point']['tick']=999
                if phase=='accepted' and mutation=='grown_strip':
                    path=Path(row['carrier']['path']);image=decode(np.frombuffer(path.read_bytes(),dtype=np.uint8))
                    for patch in image['patches']:
                        (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
                        if xl==0:
                            bits=list(patch['bits'])
                            for y in range(yl,yh+1):bits[(y-gyl)*(gxh-gxl+1)+(-1-gxl)]=int(np.asarray(12345.,dtype=np.float64).view(np.uint64))
                            patch['bits']=tuple(bits)
                    raw=encode(image);path.write_bytes(raw)
                    row['carrier'].update(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
            if phase=='accepted' and mutation=='missing_invocation' and rank==ranks-1:doc['observations']=doc['observations'][:-1]
            file.write_text(json.dumps(doc))
    if mutation=='writer_transposed':
        file=directory/'initial-metadata.json';meta=json.loads(file.read_text())
        meta['writer_geometry_boxes']=[[[box[1],box[0],box[3],box[2]] for box in level] for level in meta['writer_geometry_boxes']]
        file.write_text(json.dumps(meta))
    signature=provenance['component_signature']['inferred_boundary_expression']
    if mutation=='signature_string':provenance['component_signature']['inferred_boundary_expression']=json.dumps(signature)
    if mutation=='field_dependency':signature['dependencies']['fields']=[]
    (directory/'provenance.json').write_text(json.dumps(provenance))
    synthetic_pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    with pytest.raises((AssertionError,ValueError)):
        receive(directory,synthetic_pins)
