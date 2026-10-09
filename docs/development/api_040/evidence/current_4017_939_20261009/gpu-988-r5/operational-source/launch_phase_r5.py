"""Future outer MPI launcher; identity+pytest stay in each MPI rank process."""
import csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(os.environ['FUTURE_GPU_ROOT'])
if not(root/'plan.json').is_file():raise SystemExit('Root reviewed materialization/newSource/baseline/GO pending')
root=root;plan=json.loads((root/'plan.json').read_text())
assert plan['dispatch_admission']['authorized']is True
phase=next(p for p in plan['runtime']['phases']if p['id']==os.environ['FUTURE_GPU_PHASE'])
out=root/'results/r5'/phase['id'];out.mkdir(mode=0o700,parents=True,exist_ok=True)
command=[str(Path(sys.prefix)/'bin/mpiexec'),'-n',str(phase['world']),sys.executable,str(root/'phase_rank_runner_r5.py')]
nsys=shutil.which('nsys');tool_receipts={};trace_status='PROOF_GAP_PROFILER_UNAVAILABLE';actual_command=command
if nsys:
 for key,args in [('version',[nsys,'--version']),('profile_help',[nsys,'profile','--help']),('stats_help',[nsys,'stats','--help'])]:
  r=subprocess.run(args,text=True,capture_output=True);tool_receipts[key]={'argv':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 help_text=tool_receipts['profile_help']['stdout']+tool_receipts['profile_help']['stderr']
 if all(flag in help_text for flag in ('--trace','--output','--sample','--cpuctxsw','--trace-fork-before-exec')):
  actual_command=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--trace-fork-before-exec=true','--output='+str(out/'actual-cuda-profile'),*command]
  trace_status='ACTUAL_TRACE_REQUESTED_OUTCOME_PENDING'
 else:trace_status='PROOF_GAP_PROFILER_REQUIRED_FEATURE_UNAVAILABLE'
(out/'before-launch.json').write_text(json.dumps({'command':actual_command,'unwrapped_MPI_command':command,'tool_receipts':tool_receipts,'trace_status':trace_status,'scope':'actual requested launcher only; no fabricated events'},indent=2,sort_keys=True)+'\n')
with(out/'outer-launch.log').open('x')as log:r=subprocess.run(actual_command,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
reports=list(out.glob('actual-cuda-profile*.nsys-rep'));kernel_rows=[];stats_receipt=None
if r.returncode==0 and reports:
 stats_command=[nsys,'stats','--report','cuda_gpu_kern_sum','--format','csv',str(reports[0])]
 stats=subprocess.run(stats_command,text=True,capture_output=True);stats_receipt={'argv':stats_command,'exit':stats.returncode,'stdout':stats.stdout,'stderr':stats.stderr}
 (out/'actual-cuda-kernel-stats.csv').write_text(stats.stdout);(out/'actual-cuda-kernel-stats.stderr').write_text(stats.stderr)
 lines=stats.stdout.splitlines();start=next((i for i,line in enumerate(lines)if 'Name'in line and ('Instances'in line or 'Num Calls'in line)),None)
 if stats.returncode==0 and start is not None:
  for row in csv.DictReader(lines[start:]):
   count=row.get('Instances',row.get('Num Calls','0'))
   try:positive=int(str(count).replace(',',''))>0
   except(ValueError,TypeError):positive=False
   if row.get('Name')and positive:kernel_rows.append(row)
 trace_status='ACTUAL_CUDA_KERNEL_TABLE_RETAINED_PEER_BINDING_PENDING' if kernel_rows else'PROOF_GAP_PROFILER_REPORT_PARSE_OR_KERNEL_TABLE_UNAVAILABLE'
elif trace_status=='ACTUAL_TRACE_REQUESTED_OUTCOME_PENDING':trace_status='PROOF_GAP_PROFILE_RUN_FAILED_OR_REPORT_MISSING'
facts={'MPI_command_exit':r.returncode,'trace_status':trace_status,'reports':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in reports],'actual_kernel_rows':kernel_rows,'stats_receipt':stats_receipt,'scope':'Actual trace if produced; peer must bind it to new Native/JIT/kernel/source/ranks. Numeric PASS or compiledCUDA flag cannot fill a trace gap.'}
(out/'cuda-launch-evidence.json').write_text(json.dumps(facts,indent=2,sort_keys=True)+'\n')
raise SystemExit(r.returncode)
