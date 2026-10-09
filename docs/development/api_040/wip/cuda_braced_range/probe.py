from pathlib import Path
import datetime,hashlib,json,os,platform,re,shlex,stat,struct,subprocess,sys

ROOT=Path('/gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1')
PREFIX=Path('/gpfs/projet/r250127/rmdraux/api040-composition15-mpich-20261002-v15/qualification-sdk49-namedrhs-6d898599-cuda-dim2-20261008/envs/pops_api040_sdk49_namedrhs_6d898599_dim2')
HOST='/project/r250127/rmdraux/sol61-gpu-bootstrap-20261004/compiler13/bin/aarch64-conda-linux-gnu-c++'
CUDA='/apps/2025/manual_install/cuda_eviden/12.6'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def row(p):return {'bytes':p.stat().st_size,'sha256':sha(p),'mode':stat.S_IMODE(p.stat().st_mode)}
def borrow():
 d=load(ROOT/'borrowed-input-pins.json')
 return {'files':{n:row(Path(n)) for n in d['files']},'headers':{n:row(Path(n)) for n in d['headers']}}
def payload():
 for n,pin in load(ROOT/'payload-file-pins.json').items():
  assert row(ROOT/n)==pin,'owned payload mismatch: '+n
def command(a):
 try:
  r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  return {'argv':a,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except Exception as error:return {'argv':a,'exit':None,'stdout':'','stderr':repr(error)}

assert ROOT.resolve()==ROOT and ROOT.stat().st_uid==100267 and stat.S_IMODE(ROOT.stat().st_mode)==0o700
assert platform.machine()=='aarch64'
go=load(ROOT/'root-go.json')
assert go['RootGO'] is True and go['reviewed_plan_sha256']==sha(ROOT/'plan.json')
assert go['reviewed_probe_argvs_sha256']==sha(ROOT/'probe-argvs.json')
assert go['reviewed_probe_script_sha256']==sha(ROOT/'probe.py')
payload()
assert not (ROOT/'outputs').exists(),'one execution only; preserve all prior output'
out=ROOT/'outputs';out.mkdir(mode=0o700)
expected=load(ROOT/'borrowed-input-pins.json')
try:before=borrow()
except Exception as error:
 save(out/'borrowed-before.json',{'error':repr(error)})
 save(out/'summary.json',{'results':[],'paired_evidence_gate_passed':False,'preflight_failure':'borrowed input snapshot unavailable','compiler_executed':False,'GPU_native_qualified':False,'scientific_qualified':False})
 sys.exit(1)
save(out/'borrowed-before.json',before)
if before!={k:expected[k] for k in ['files','headers']}:
 save(out/'summary.json',{'results':[],'paired_evidence_gate_passed':False,'preflight_failure':'borrowed input pins mismatch','compiler_executed':False,'GPU_native_qualified':False,'scientific_qualified':False})
 sys.exit(1)
versions={n:command(a) for n,a in {'nvcc':[CUDA+'/bin/nvcc','--version'],'host':[HOST,'--version'],'scheduler':['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],'gpu':['nvidia-smi','--query-gpu=uuid,name,pci.bus_id','--format=csv,noheader']}.items()}
save(out/'allocation-and-toolchain.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'machine':platform.machine(),'commands':versions,'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'}})
if not (versions['nvcc']['exit']==0 and 'V12.6.85' in versions['nvcc']['stdout'] and versions['host']['exit']==0 and '13.4.0' in versions['host']['stdout']):
 save(out/'summary.json',{'results':[],'paired_evidence_gate_passed':False,'preflight_failure':'actual compiler version mismatch','compiler_TU_executed':False,'GPU_native_qualified':False,'scientific_qualified':False})
 sys.exit(1)
results=[]
try:
 for variant,args in load(ROOT/'probe-argvs.json').items():
  d=out/variant;d.mkdir(mode=0o700);(d/'keep').mkdir(mode=0o700);(d/'tmp').mkdir(mode=0o700)
  env=os.environ.copy()
  env.update(CUDA_ROOT=CUDA,NVCC_WRAPPER_DEFAULT_COMPILER=HOST,TMPDIR=str(d/'tmp'),NVCC_WRAPPER_TMPDIR=str(d/'tmp'),NVCC_WRAPPER_SHOW_COMMANDS_BEING_RUN='1',PS4='+ ',CUDA_CACHE_PATH=str(out/'cuda-cache'),CCACHE_DIR=str(out/'ccache'),XDG_CACHE_HOME=str(out/'cache'))
  for n in ['PYTHONPATH','PYTHONOPTIMIZE','NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS']:env.pop(n,None)
  invocation=['/bin/bash','-x',*args]
  save(d/'outer-argv.json',{'argv':invocation,'environment_changes':{k:env[k] for k in ['CUDA_ROOT','NVCC_WRAPPER_DEFAULT_COMPILER','TMPDIR','NVCC_WRAPPER_TMPDIR','NVCC_WRAPPER_SHOW_COMMANDS_BEING_RUN','PS4']}})
  (d/'outer-argv.nul').write_bytes(b'\0'.join(x.encode() for x in invocation)+b'\0')
  start=datetime.datetime.now(datetime.timezone.utc).isoformat()
  compile_exit=None;launch_error=None
  with (d/'stdout.log').open('w') as stdout,(d/'xtrace-and-stderr.log').open('w') as stderr:
   try:
    r=subprocess.run(invocation,cwd=d,env=env,stdout=stdout,stderr=stderr)
    compile_exit=r.returncode
   except Exception as error:
    launch_error=repr(error);stderr.write('\nProbe launch exception: '+launch_error+'\n')
  trace=(d/'xtrace-and-stderr.log').read_text(errors='replace')
  inner=[];inner_parse_errors=[]
  for line in trace.splitlines():
   if line.startswith('+ '+CUDA+'/bin/nvcc '):
    try:inner.append(shlex.split(line[2:]))
    except ValueError as error:inner_parse_errors.append({'line':line,'error':str(error)})
  save(d/'actual-inner-argv.json',{'capture':'xtrace-derived expanded NVCC command from original wrapper; no kernel/syscall instrumentation','trace_sha256':sha(d/'xtrace-and-stderr.log'),'argvs':inner,'parse_errors':inner_parse_errors})
  for i,a in enumerate(inner):(d/('actual-inner-argv-'+str(i)+'.nul')).write_bytes(b'\0'.join(x.encode() for x in a)+b'\0')
  keep={p.relative_to(d).as_posix():row(p) for p in (d/'keep').rglob('*') if p.is_file() and stat.S_ISREG(p.lstat().st_mode)}
  save(d/'keep-inventory.json',keep)
  ii_files=[n for n,pin in keep.items() if Path(n).name.startswith('amr_layout_transfer.') and n.endswith('.ii') and pin['bytes']>0]
  cudafe1_cpp_files=[n for n,pin in keep.items() if Path(n).name=='amr_layout_transfer.cudafe1.cpp' and pin['bytes']>0]
  obj=d/'amr_layout_transfer.cpp.o'
  elf={'path':str(obj),'compiler_output_path':args[args.index('-o')+1],'exists':obj.exists(),'regular_file':False,'bytes':0,'sha256':None,'ELF_magic':False,'ELF_class':None,'ELF_data_encoding':None,'e_type':None,'e_machine':None,'ELF64_LE_AArch64_relocatable':False}
  if obj.exists():
   elf['regular_file']=stat.S_ISREG(obj.lstat().st_mode)
   if elf['regular_file']:
    body=obj.read_bytes();elf.update(bytes=len(body),sha256=sha(obj),ELF_magic=body[:4]==b'\x7fELF')
    if len(body)>=64:
     elf.update(ELF_class=body[4],ELF_data_encoding=body[5],e_type=struct.unpack_from('<H',body,16)[0],e_machine=struct.unpack_from('<H',body,18)[0])
     elf['ELF64_LE_AArch64_relocatable']=body[:7]==b'\x7fELF\x02\x01\x01' and elf['e_type']==1 and elf['e_machine']==183
  baseline_failures={str(n):bool(re.search(r'amr_layout_transfer\.cpp:'+str(n)+r":\d+: error: expected ';' before '}' token",trace)) for n in [232,992,1361]}
  facts={'xtrace_derived_inner_argv_available':bool(inner) and not inner_parse_errors,'nonempty_kept_ii_available':bool(ii_files),'nonempty_kept_cudafe1_cpp_available':bool(cudafe1_cpp_files),'compile_exit_is_zero':compile_exit==0,'compile_failed':compile_exit is not None and compile_exit!=0,'baseline_failures':baseline_failures,'object_is_actual_compiler_output':elf['path']==elf['compiler_output_path'],'object_is_nonempty_regular_ELF64_LE_AArch64_ET_REL':elf['regular_file'] and elf['bytes']>=64 and elf['ELF64_LE_AArch64_relocatable']}
  common_gate=all(facts[n] for n in ['xtrace_derived_inner_argv_available','nonempty_kept_ii_available','nonempty_kept_cudafe1_cpp_available'])
  evidence_gate=common_gate and (facts['compile_failed'] and all(baseline_failures.values()) if variant=='baseline' else facts['compile_exit_is_zero'] and facts['object_is_actual_compiler_output'] and facts['object_is_nonempty_regular_ELF64_LE_AArch64_ET_REL'])
  result={'variant':variant,'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compile_exit':compile_exit,'launch_error':launch_error,'inner_nvcc_commands':len(inner),'inner_argv_capture_scope':'xtrace-derived; no kernel/syscall instrumentation','keep_files':len(keep),'ii_files':ii_files,'cudafe1_cpp_files':cudafe1_cpp_files,'object':elf,'facts':facts,'evidence_gate_passed':evidence_gate,'Native_GPU7_built':False,'scientific_qualified':False}
  save(d/'result.json',result);results.append(result)
  if not evidence_gate:
   save(d/'gate-failure.json',{'variant':variant,'facts':facts,'result_sha256':sha(d/'result.json'),'reason':'required compiler/intermediate/object evidence incomplete; result and original raw logs retained'})
   break
finally:
 after=None;borrowed_error=None
 try:after=borrow()
 except Exception as error:borrowed_error=repr(error)
 save(out/'borrowed-after.json',after if after is not None else {'error':borrowed_error})
 paired_gate=len(results)==2 and all(r['evidence_gate_passed'] for r in results) and after==before
 save(out/'summary.json',{'results':results,'paired_evidence_gate_passed':paired_gate,'borrowed_unchanged':after==before,'borrowed_after_error':borrowed_error,'Source':'6d898599a3f66fa9aa3c049ba0f3114f1960b6dc','GPU_native_qualified':False,'scientific_qualified':False,'scope':'only original and candidate runtime AMR TU compiles; emitted originals retained; no CPP/PDE/Native import'})
sys.exit(0 if paired_gate else 1)
