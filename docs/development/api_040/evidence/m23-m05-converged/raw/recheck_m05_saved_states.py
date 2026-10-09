"""Independent periodic shear recomputation from native NPZ and raw exchange ledgers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(base):
    results=[]
    for path in sorted(base.rglob('state_*.npz')):
        with np.load(path,allow_pickle=False) as state:
            n=int(state['cells']); dt=float(state['dt']); time=float(state['time'])
            u0,u1,u2=[state[k].reshape(n) for k in ('initial','before_last','final')]
        assert all(np.isfinite(u).all() for u in (u0,u1,u2))
        initial=np.sinc(1/n)*np.sin(2*np.pi*(np.arange(n)+.5)/n)
        lam=4*n*n*np.sin(np.pi/n)**2
        factor=1-.03*lam*.1/n
        continuous=initial*np.exp(-.03*(2*np.pi)**2*.1)
        faces=.03*n*(np.roll(u1,-1)-u1)
        rate=n*(faces-np.roll(faces,1))
        work=float(np.dot(u1,rate)/n)
        jump=np.roll(u1,-1)-u1
        dissipation=-.03*n*float(np.dot(jump,jump))
        energy_change=float((np.dot(u2,u2)-np.dot(u1,u1))/(2*n))
        correction=.5*dt*dt*float(np.dot(rate,rate))/n
        ledger_path=path.with_name('ledger_%d.json'%n)
        records=json.loads(ledger_path.read_text())
        assert len(records)==2*n
        increments=np.zeros(n); seen=set(); contexts=set(); occurrences=set(); error=0.
        for row in records:
            assert all(np.isfinite(row[k]) for k in ('numerical_flux','integrated_amount','face_measure','temporal_weight'))
            cell,axis,side=[int(t.split(':')[1]) for t in row['quadrature_identity'].split('/')]
            assert 0<=cell<n and axis==0 and side in (0,1)
            sign=1 if side else -1
            assert row['orientation']==sign and row['multiplicity']==1
            assert row['face_measure']==1 and abs(row['temporal_weight']-dt)<2e-14
            expected=faces[cell if side else (cell-1)%n]
            error=max(error,abs(row['numerical_flux']-expected))
            assert abs(row['integrated_amount']-sign*expected*row['temporal_weight'])<5e-13
            key=(row['operation_identity'],row['occurrence_identity'],row['evaluation_context'],row['quadrature_identity'])
            assert key not in seen;seen.add(key)
            contexts.add(row['evaluation_context']);occurrences.add(row['occurrence_identity'])
            increments[cell]+=row['integrated_amount']
        assert len(contexts)==len(occurrences)==1
        metrics={'initial_error':float(np.max(np.abs(u0-initial))),
                 'before_error':float(np.max(np.abs(u1-initial*factor**(n-1)))),
                 'discrete_error':float(np.max(np.abs(u2-initial*factor**n))),
                 'continuous_error':float(np.max(np.abs(u2-continuous))),
                 'mass_error':float(abs(np.mean(u2-u0))),
                 'semidiscrete_identity_error':abs(work-dissipation),
                 'fe_energy_identity_error':abs(energy_change-dt*dissipation-correction),
                 'ledger_flux_error':float(error),
                 'ledger_increment_error':float(np.max(np.abs(increments-(u2-u1)/n)))}
        assert metrics['initial_error']<2e-14
        assert max(metrics[k] for k in ('before_error','discrete_error'))<5e-12
        assert metrics['continuous_error']<2e-4 and metrics['ledger_flux_error']<2e-12
        assert max(metrics[k] for k in ('mass_error','semidiscrete_identity_error','fe_energy_identity_error','ledger_increment_error'))<5e-13
        assert energy_change<0 and work<0 and dissipation<0 and abs(time-.1)<3e-14
        results.append({'path':str(path.relative_to(base)),'sha256':sha(path),
                        'ledger_sha256':sha(ledger_path),'cells':n,'records':len(records),**metrics})
    assert {r['cells'] for r in results}=={32,64,128} and len(results)==3
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--serial',type=Path,required=True)
    parser.add_argument('--mpi',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'schema_version':1,'status':'passed','script_sha256':sha(Path(__file__)),
            'numpy_version':np.__version__,'serial':check(args.serial),'mpi2':check(args.mpi)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for name in ('serial','mpi2'):
        print(name,[(r['cells'],r['discrete_error'],r['ledger_increment_error']) for r in report[name]])
