"""Versioned SDK16 export envelope around historical saved-physics @1; no PoPS."""
import argparse,json,os,re
import xml.etree.ElementTree as ET
from pathlib import Path
from tests.review import sol61_m17_saved_physics_offline as v1
SOURCE='9bdc6acd7a07b11d9ddfd6eebd75f514add51ef3'
NATIVE='3f4e01cfa9d69dbd1148519a947c25895814cca4b5e010c5348ff9f7e41a3fff'
HEADER='ae40d2c0bd718f7bed15da474dd30106f48aa69dc7c7f56303e4c06dd4e9efe4'
SCHEMA='sol61.m17.saved-physics-export@2'

def public_order(value):
    if type(value) is not tuple:raise ValueError('expected public basis malformed')
    v1.require(len(value)==15 and all(type(x) is tuple and len(x)==2 and all(type(k) is int for k in x) for x in value) and set(value)==set(v1.I),'expected public basis malformed')
    return value

def identity_profile(identity,source):
    v1.require(identity['source_commit']==SOURCE and source['head']==SOURCE and len(source['files'])==1188,'Source freeze/inventory differs')
    v1.require(identity['native_sha256']==NATIVE and 'headers='+HEADER+';' in identity['abi_key'],'Native/header profile differs')
    v1.require(type(identity['verified_source_files']) is int and identity['verified_source_files']==1144,'installed source inventory differs')

def receive_export(reception_path,expected_reception_sha,expected_order):
    reception_path=Path(reception_path)
    v1.require(type(expected_reception_sha) is str and re.fullmatch('[0-9a-f]{64}',expected_reception_sha) is not None and v1.sha(reception_path)==expected_reception_sha,'external ROOT reception pin differs')
    receipt=v1.load(reception_path)
    v1.require(receipt['schema']=='root.api040.native-transfer-received@1','transfer reception schema differs')
    result=receipt['pytest_result']
    v1.require(type(result['returncode']) is int and result['returncode']==0 and all(type(value) is int for value in result['counts'].values()) and result['status']=='passed' and result['counts']=={'tests':1,'failures':0,'errors':0,'skipped':0},'pytest did not close one Native case')
    root=Path(receipt['received']);pin_path=Path(receipt['transfer_pin'])
    v1.require(v1.sha(pin_path)==receipt['transfer_pin_sha256'],'transfer pin differs')
    pin=v1.load(pin_path)
    v1.require(pin['schema']=='root.api040.native-transfer@1' and pin['archive_sha256']==receipt['archive_sha256'],'archive pin differs')
    v1.require(pin['result']['source_freeze']==SOURCE and type(pin['result']['world_request']) is int and pin['result']['world_request']==1 and all(type(pin['result'][key]) is int and pin['result'][key]==0 for key in ('original_exit','preservation_exit')),'source/result/Serial profile differs')
    paths=set();regular=links=0
    for row in pin['members']:
        name=row['path'];relative=Path(name)
        v1.require(type(name) is str and not relative.is_absolute() and '..' not in relative.parts and name not in paths,'member alias/escape differs');paths.add(name)
        path=root/relative
        if row['kind']=='file':
            v1.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) and type(row['bytes']) is int and path.stat().st_size==row['bytes'] and v1.sha(path)==row['sha256'],'export member differs: '+name);regular+=1
        elif row['kind']=='symlink':
            v1.require(path.is_symlink() and os.readlink(path)==row['target'],'export symlink differs');links+=1
        else:raise ValueError('unknown export member kind')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_symlink() or p.is_file()}
    v1.require(actual==paths and type(receipt['regular_files']) is int and type(receipt['symlinks']) is int and regular==receipt['regular_files'] and links==receipt['symlinks'],'export closure differs')
    for before,after in (('before/identity.json','after/identity.json'),('before/source-files.json','after/source-files.json'),('source-before.json','source-after.json'),('installed-before.json','installed-after.json'),('preserved-before.json','preserved-after.json')):
        v1.require((root/before).read_bytes()==(root/after).read_bytes(),'before/after inventory differs')
    xml=ET.parse(root/'output/pytest.xml').getroot()
    v1.require(len(list(xml.iter('testcase')))==1 and not any(list(xml.iter(tag)) for tag in ('failure','error','skipped')),'JUnit Native case differs')
    identity=v1.load(root/'before/identity.json');source=v1.load(root/'source-before.json')
    identity_profile(identity,source)
    candidates=[root/Path(name).parent for name in paths if name.endswith('/fan-li15-composition/receipt.json')]
    v1.require(len(candidates)==1,'scientific receipt missing/ambiguous')
    directory,=candidates;author=v1.load(directory/'authoring.json');fixture=v1.load(directory/'receipt.json')
    expected_order=public_order(expected_order)
    v1.require(tuple(tuple(x) for x in author['order'])==expected_order,'authored component routing differs')
    v1.require(fixture['native']['sha256']==NATIVE,'fixture Native differs')
    program=directory/'program.ir.json';cpp=directory/'program.cpp'
    v1.require(program.is_file() and cpp.is_file(),'retained program missing')
    # Actual remote Program path belongs to the pinned export cache, not an imported DSO.
    program_sha=fixture['program']['sha256']
    binaries=[root/row['path'] for row in pin['members'] if row['kind']=='file' and row['path'].endswith('.so')]
    v1.require(len(binaries)==2 and sum(v1.sha(p)==program_sha for p in binaries)==1,'Program/model DSO closure differs')
    physics=v1.receive(directory)
    return {'schema':SCHEMA,'source':SOURCE,'native_sha256':NATIVE,'header_signature':HEADER,'production_files':1188,'installed_source_files':1144,'regular_files':regular,'symlinks':links,'order':[list(a) for a in expected_order],'physics':physics,'root_reception_sha256':expected_reception_sha,'root_scientific_approval':False,'full_M17_qualification':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reception');p.add_argument('--sha256',required=True);p.add_argument('--order',choices=('canonical','reverse'),required=True);a=p.parse_args()
    order=v1.I if a.order=='canonical' else tuple(reversed(v1.I))
    print(json.dumps(receive_export(a.reception,a.sha256,order),indent=2))
