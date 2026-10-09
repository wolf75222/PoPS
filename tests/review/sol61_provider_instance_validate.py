"""Bounded Source/Host replay with original stdout/stderr; never installs/imports Native."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def pin(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    receipts=[]
    def run(label,command,expected=0):
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
        stdout=out/(label+'.stdout');stderr=out/(label+'.stderr')
        stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
        receipt={'label':label,'argv':command,'returncode':result.returncode,
                 'expected_returncode':expected,'stdout':pin(stdout),'stderr':pin(stderr)}
        receipts.append(receipt)
        if result.returncode!=expected:
            (out/'failed-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
            raise RuntimeError(label+' failed; original output retained')
        return result
    replay=str(ROOT/'tests/review/sol61_provider_instance_source_replay.py')
    for mode in ('baseline-legacy','candidate-legacy','baseline-negative','candidate-default','candidate-joint'):
        result=run(mode,[sys.executable,replay,mode,str(out/mode)],1 if mode.endswith('negative') else 0)
        if mode.endswith('negative') and b'consumed field publication cannot share a model-definition provider key' not in result.stderr:
            raise RuntimeError('baseline did not reproduce the real cardinality refusal')
    legacy=[]
    for path in sorted((out/'baseline-legacy').glob('*.cpp')):
        candidate=out/'candidate-legacy'/path.name
        if path.read_bytes()!=candidate.read_bytes():raise RuntimeError('legacy CPP drift: '+path.name)
        legacy.append({'original':pin(path),'candidate':pin(candidate),'equal_bytes':True})
    prefix=Path(sys.prefix)
    common=['/usr/bin/clang++','-Xpreprocessor','-fopenmp','-std=c++20',
            '-DPOPS_NATIVE_DIM=2','-DPOPS_HAS_KOKKOS=1','-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI=1',
            '-I'+str(ROOT/'include'),'-I'+str(prefix/'include')]
    actual_cpp=[p for mode in ('candidate-default','candidate-joint') for p in sorted((out/mode).glob('*.cpp'))]
    run('actual-program-and-model-host-syntax',common+['-fsyntax-only']+[str(p) for p in actual_cpp])
    executable=out/'instance-storage-host'
    run('real-registry-host-build',common+[str(ROOT/'tests/review/sol61_provider_instance_storage.cpp'),
        '-L'+str(prefix/'lib'),'-lkokkoscore','-lomp','-Wl,-rpath,'+str(prefix/'lib'),'-o',str(executable)])
    env['OMP_NUM_THREADS']='2';env['OMP_PROC_BIND']='false'
    run('real-registry-host-run',[str(executable)])
    report={'schema':'sol61.field-provider-instance-source-host@1','scope':'Source/Host; no installed PoPS/MPI/GPU/science',
            'base':'b0a9409fa412575a0e5bd47b26705e6afe88f1a6',
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'legacy_byte_comparisons':legacy,'actual_cpp':[pin(p) for p in actual_cpp],
            'executable':pin(executable),'commands':receipts}
    (out/'receipt.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('baseline refusal, 3 legacy byte equalities, 10 actual Host TUs, registry/carrier/rollback/retry PASS')


if __name__=='__main__':main()
