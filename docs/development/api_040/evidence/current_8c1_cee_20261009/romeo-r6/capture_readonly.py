from pathlib import Path
import datetime,hashlib,json,subprocess
BASE=Path(__file__).parent
now=datetime.datetime.now(datetime.timezone.utc)
previous=sorted(BASE.glob('snapshot-*.json'))
if previous:
 old=json.loads(previous[-1].read_text());elapsed=(now-datetime.datetime.fromisoformat(old['remote']['utc'])).total_seconds();assert elapsed>=60,'At most one remote observation per minute'
else:assert (now-datetime.datetime.fromisoformat('2026-10-09T02:33:56.773794+00:00')).total_seconds()>=60
program=r'''
from pathlib import Path
import datetime,json,hashlib,subprocess
root=Path('/gpfs/scratch/rmdraux/PoPS-final-full-native-cuda-dim2-gh200-um-98804c68-20261009')
assert root.stat().st_uid==100267 and root.stat().st_mode&0o777==0o700
commands=[]
for argv in [['squeue','-j','738213,738214','-h','-o','%i|%j|%T|%R|%u'],['sacct','-j','738213,738214','-n','-X','-P','-o','JobID,State,ExitCode,Elapsed,NodeList']]:
 r=subprocess.run(argv,text=True,capture_output=True,timeout=20);commands.append({'argv':argv,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
names=['r6-admission.json','results/r6-dispatch-progress.jsonl','results/official-build-entered-r6.json','results/official-build-r6.log','results/r6-native-finally-statuses.json','results/gpu-native-admission-r6.json','results/r6-conda-env-list.txt','results/r6-conda-selection.json','results/r6-selected-environment.json','results/r6-prior-r5-failure-admission.json','results/kokkos-unified-profile-manifest-r6.json','results/managed-pointer-launch-probe-r6.json','results/build-738213-r6.log','results/runtime-738214-r6.log']
rows=[]
for name in names:
 p=root/name;row={'relative':name,'exists':p.exists(),'symlink':p.is_symlink()}
 if p.is_file():
  data=p.read_bytes();row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mode=oct(p.stat().st_mode&0o777),uid=p.stat().st_uid)
  if len(data)<=131072:row['text']=data.decode('utf8',errors='replace')
  else:row['tail']=data[-16000:].decode('utf8',errors='replace')
 rows.append(row)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Source':'98804c6849c9966e05b5397126bab983a5fa0227','SDK':'de2b5ef24a9fae48ab143d4200aab2885c9400dcd43712fa74928f5f2c4e954a','namespace':str(root),'commands':commands,'files':rows,'scope':'Read-only text/JSON+queue/accounting only; no ARM/Conda/Native/build/install/submission/cancel'}))
'''
argv=['rtk','proxy','ssh','romeo','/usr/bin/python3 -'];r=subprocess.run(argv,input=program,text=True,capture_output=True,timeout=90)
assert r.returncode==0,r.stderr
remote=json.loads(r.stdout)
record={'remote':remote,'argv':argv,'exit':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr':r.stderr,'no_remote_write_or_executable_launch':True}
name='snapshot-'+remote['utc'].replace(':','').replace('+','_')+'.json';p=BASE/name
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
files={x['relative']:x for x in remote['files']}
summary={'snapshot':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'utc':remote['utc'],'accounting':remote['commands'][1]['stdout'],'file_states':[{k:x.get(k) for k in ('relative','exists','bytes','sha256')} for x in remote['files']],'bounded_logs':{n:x.get('tail',x.get('text',''))[-5000:] for n,x in files.items() if n.endswith('.log')},'env_selection':files['results/r6-selected-environment.json'].get('text'),'entry':files['results/official-build-entered-r6.json'].get('text'),'finally':files['results/r6-native-finally-statuses.json'].get('text')}
print(json.dumps(summary,indent=2))
