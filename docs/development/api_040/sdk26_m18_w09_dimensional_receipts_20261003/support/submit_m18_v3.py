"""Submit one real, dimension-authenticated SDK26 M18 Native qualification."""
import argparse,base64,datetime,hashlib,json
from pathlib import Path
import shlex,subprocess,sys

BASE=Path(__file__).resolve().parent
ROOT='/project/r250127/api040-composition15-mpich-20261002-v15'
FREEZE='51b0eec9dcfd660098cc5531eed6c400185c8dad'
FIXTURE=Path('/Users/romaindespoulain/dev/tmp/PoPS-sol61-m18-w09/tests/python/integration/runtime/test_m18_w09_declared_domain_runtime.py')
REMOTE=r'''
import base64,hashlib,json,pathlib,subprocess,sys
r=json.loads(sys.argv[1]);root=pathlib.Path(r['root']);source=root/'source'
qual=root/'qualification-sdk26-m18-dim2-20261003'
result_path=qual/('native-submission-result-'+r['operation']+'.json')
request_path=qual/('native-submission-request-'+r['operation']+'.json')
if result_path.exists():
 data=json.loads(result_path.read_text());assert data['request']==r;print(json.dumps(data,sort_keys=True));raise SystemExit(0)
if request_path.exists():raise SystemExit('Previous outcome unknown; inspect SLURM, do not duplicate')
head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert head==r['source_freeze'],'Source freeze differs'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
live=subprocess.check_output(['squeue','-u','rmdraux','-h','-o','%i|%T|%j|%N|%M'],text=True)
env=qual/'envs'/'pops_api040_sdk26_m18_dim2';assert (env/'bin/python').is_file()
build=json.loads((qual/'build-job-732416'/'result.json').read_text());assert build['original_exit']==0 and build['original_sdk25_preserved_exit']==0 and build['dimension']==2
def install(path,key):
 body=base64.b64decode(r[key+'_base64'],validate=True);assert hashlib.sha256(body).hexdigest()==r[key+'_sha256']
 if path.exists():assert path.read_bytes()==body,'Existing artifact differs'
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(body)
script=root/'preparation'/('native-sdk26-m18-dim%d-%s.sbatch'%(r['dimension'],r['script_sha256'][:16]))
fixture=source/'tests/python/integration/runtime/test_m18_w09_declared_domain_runtime.py'
# Frozen tracked fixture must already exist; never mutate Source to submit.
body=base64.b64decode(r['fixture_base64'],validate=True)
assert hashlib.sha256(body).hexdigest()==r['fixture_sha256']
assert fixture.is_file() and fixture.read_bytes()==body,'Frozen fixture differs'
install(script,'script')
test=str(fixture.relative_to(source))+'::test_declared_quadrature_near_boundary_and_w09'
exports='ALL,'+','.join(k+'='+str(v) for k,v in {'NATIVE_RUN_LABEL':r['label'],'NATIVE_WORLD_SIZE':r['world_size'],'NATIVE_SOURCE_FREEZE':r['source_freeze'],'NATIVE_TEST_NODE':test,'NATIVE_FIXTURE_SHA256':r['fixture_sha256']}.items())
command=['sbatch','--parsable','--ntasks='+str(r['world_size']),'--export='+exports,'--chdir='+str(root),'--output='+str(qual/'jobs'/('native-'+r['label']+'-%j.out')),'--error='+str(qual/'jobs'/('native-'+r['label']+'-%j.err')),str(script)]
with request_path.open('x') as f:json.dump({'request':r,'submission_argv':command,'active_jobs_before':live.splitlines()},f,sort_keys=True)
p=subprocess.run(command,text=True,capture_output=True)
out={'request':r,'source_head':head,'test':test,'active_jobs_before':live.splitlines(),'submission_argv':command,'sbatch_returncode':p.returncode,'job_result':p.stdout.strip(),'sbatch_stderr':p.stderr}
with result_path.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(out,sort_keys=True));raise SystemExit(p.returncode)
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--world-size',type=int,choices=(1,2),required=True);a=p.parse_args()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 script=(BASE/'native-dim2-v3.sbatch').read_bytes();fixture=FIXTURE.read_bytes()
 assert hashlib.sha256(fixture).hexdigest()=='d8533e19434279b287cadf119e6b4a068d4f5eee90b3886b927e0ddc26e95a12'
 r={'root':ROOT,'dimension':2,'world_size':a.world_size,'source_freeze':FREEZE,'operation':stamp,'label':f'sdk26-m18-world{a.world_size}-v3'}
 for name,body in [('script',script),('fixture',fixture)]:r[name+'_base64']=base64.b64encode(body).decode('ascii');r[name+'_sha256']=hashlib.sha256(body).hexdigest()
 request=BASE/f'native-request-dim2-world{a.world_size}-{stamp}.json'
 with request.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
 result=subprocess.run(['ssh','romeo',shlex.join(['/usr/bin/python3','-',json.dumps(r,separators=(',',':'))])],input=REMOTE,text=True,capture_output=True)
 data={'request_path':str(request),'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),'submitter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ssh_returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
 if result.returncode==0:data['remote_result']=json.loads(result.stdout)
 receipt=BASE/f'native-submission-dim2-world{a.world_size}-{stamp}.json'
 with receipt.open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'receipt':str(receipt),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'returncode':result.returncode,'job':data.get('remote_result',{}).get('job_result')}))
 if result.returncode:sys.stderr.write(result.stderr)
 return result.returncode

if __name__=='__main__':raise SystemExit(main())
