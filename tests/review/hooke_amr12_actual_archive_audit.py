"""Inspect frozen actual native receipt files using stdlib only; no PoPS import."""
import ast, hashlib, json, pathlib, struct, sys, xml.etree.ElementTree as ET, zipfile
BASE=pathlib.Path(sys.argv[1])
DSO='13b3d024eed5f2776e96dbee6f674fe2e7f48ccf3ff31382b13749444cfe18f7'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def arrays(p):
 with zipfile.ZipFile(p) as z:return {n:z.read(n) for n in z.namelist()}
def doubles(b):
 start=10 if b[6]==1 else 12;n=struct.unpack('<H' if start==10 else '<I',b[8:start])[0];h=ast.literal_eval(b[start:start+n].decode());assert h['descr']=='<f8';return b[start+n:]
for suffix,size in [('serial',1),('mpi2',2)]:
 directory=BASE/f'installed-sdka37-amr12-refusals-{suffix}-dim2'
 before=directory/('identity.json' if size==1 else 'before/identity.json');a=json.loads(before.read_text());b=json.loads((directory/'after/identity.json').read_text())
 for key in ('source_commit','source_diff_sha256','native_file','native_sha256','abi_key','verified_source_files','source_files_sha256'):assert a[key]==b[key],key
 assert a['schema_version']==2 and a['verified_source_files']==1132 and a['source_commit'].startswith('84284034') and a['native_sha256']==DSO
 assert 'headers=a37c25c95' in a['abi_key'] and ';dim=2' in a['abi_key'] and ';mpi=1' in a['abi_key']
 xmls=[directory/'pytest.xml'] if size==1 else [directory/f'rank{r}.xml' for r in range(size)]
 for path in xmls:
  suites=ET.parse(path).getroot().findall('testsuite');assert sum(int(s.get('tests','0')) for s in suites)==1
  assert not any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped'))
 receipts=sorted(directory.rglob('receipt-rank*.json'));assert len(receipts)==size
 loaded=[json.loads(path.read_text()) for path in receipts]
 if size==2:
  assert loaded[0]['refusals']==loaded[1]['refusals']
  assert loaded[0]['snapshots']['baseline']['files']['state_bytes']['sha256']==loaded[1]['snapshots']['baseline']['files']['state_bytes']['sha256']
 for rank,path in enumerate(receipts):
  r=json.loads(path.read_text());assert (r['rank'],r['size'],r['dimension'],r['width'])==(rank,size,2,2)
  assert r['native']['sha256']==DSO and r['native']['path']==a['native_file'];assert sha(r['fixture']['path'])==r['fixture']['sha256']
  assert len(r['refusals'])==9+(size==2)
  for name,rows in r['refusals'].items():assert len(rows)==size and all(row and row[0] in ('ValueError','TypeError','RuntimeError') for row in rows)
  snapshots=r['snapshots'];x,y,z=(snapshots[k] for k in ('baseline','contradicted','rolled-back'))
  for snap in snapshots.values():
   for rec in [*snap['files'].values(),*snap['levels']]:assert sha(rec['path'])==rec['sha256']
  for k in ('state_bytes','diagnostics_bytes'):assert pathlib.Path(x['files'][k]['path']).read_bytes()==pathlib.Path(z['files'][k]['path']).read_bytes()
  assert x['metadata']==z['metadata'] and x['diagnostic_bits']==z['diagnostic_bits'] and x['temporal']==z['temporal']
  for old,new in zip(x['levels'],z['levels']):assert arrays(old['path'])==arrays(new['path'])
  old=arrays(x['levels'][0]['path'])['Q0.npy'];changed=arrays(y['levels'][0]['path'])['Q0.npy'];v,w=doubles(old),doubles(changed)
  # Positive Q0: each changed float word must advance exactly one ULP; MPI peers
  # see the assembled global projection, not a private substitute array.
  pairs=list(zip(struct.unpack('<'+'Q'*(len(v)//8),v),struct.unpack('<'+'Q'*(len(w)//8),w)))
  diffs=[(a,b) for a,b in pairs if a!=b];assert diffs and all(b==a+1 for a,b in diffs)
  assert x['diagnostic_bits']==y['diagnostic_bits'] and x['temporal']==y['temporal']
  if size==2:
   worker=json.loads((directory/f'rank{rank}.identity.json').read_text());assert worker['native_sha256']==DSO and worker['native_file']==a['native_file'] and (worker['rank'],worker['ranks'],worker['dimension'])==(rank,size,2)
  print(json.dumps({'archive':suffix,'rank':rank,'xml_pass':True,'refusals':len(r['refusals']),'rollback_bytes_exact':True,'Q0_changed_one_ulp_cells':len(diffs),'parent_inventory_files':1132,'DSO':DSO},sort_keys=True))
print('ARCHIVED_NATIVE_READ_ONLY audit passed; no scientific AMR topology receipt')
