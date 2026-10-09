"""Recompute Hall phase/amplitude and independent matrix ledgers from saved native states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recheck(base):
    hall, matrix = [], []
    for path in sorted(base.rglob('state_*.npz')):
        with np.load(path,allow_pickle=False) as state:
            cells, eta, dt, time = (float(state[k]) for k in ('cells','eta_h','dt','time'))
            cells = int(cells)
            order = state['order']
            initial, final = (state[k][np.argsort(order)] for k in ('initial','final'))
        x = 2*np.pi*(np.arange(cells)+.5)/cells
        expected_initial = np.array((1.,.2))[:,None]*np.sinc(2/cells)*np.cos(2*x)
        h = 2*np.pi/cells
        frequency = eta*4*np.sin(h)**2/h**2
        z = frequency*dt
        amplification = (1-.5*z*z-1j*z)**100
        u0, u1 = initial[0]+1j*initial[1], final[0]+1j*final[1]
        measured = np.vdot(u0,u1)/np.vdot(u0,u0)
        row = {'path':str(path.relative_to(base)),'sha256':sha(path),
               'cells':cells,'eta_h':eta,'order':order.tolist(),
               'initial_error':float(np.max(np.abs(initial-expected_initial))),
               'discrete_error':float(np.max(np.abs(u1-amplification*u0))),
               'phase_error':float(abs(np.angle(measured/amplification))),
               'norm_error':float(abs(np.linalg.norm(u1)/np.linalg.norm(u0)-abs(amplification))),
               'observed_signed_phase':float(np.angle(measured)),
               'continuous_signed_phase':-eta*4*.2,
               'continuous_phase_error':float(abs(np.angle(measured/np.exp(-1j*eta*4*.2)))),
               'continuous_norm_defect':float(abs(np.linalg.norm(u1)/np.linalg.norm(u0)-1))}
        assert dt == .002 and abs(time-.2)<3e-14
        assert row['initial_error'] < 3e-14
        assert max(row[k] for k in ('discrete_error','phase_error','norm_error')) < 3e-10
        hall.append(row)
    d = np.outer([1.,2.,-1.],[1.,2.,-1.])
    r = np.array(((0.,-.3,.2),(.3,0.,-.1),(-.2,.1,0.)))
    for path in sorted(base.rglob('accepted_matrix.npz')):
        with np.load(path,allow_pickle=False) as state:
            initial, final, increment = (state[k] for k in ('initial','final','increment'))
            order, dt = state['order'],float(state['dt'])
        n=initial.shape[1]
        def rate(u):
            return 2*(d+r)@((np.roll(u,1,axis=1)-2*u+np.roll(u,-1,axis=1))*n*n)
        predictor=initial+dt*rate(initial)
        expected=.5*(initial+predictor+dt*rate(predictor))
        ledger_path=path.with_name('accepted_ledger.json')
        ledger=json.loads(ledger_path.read_text())
        assert len(ledger)==3*n*2*2*2
        seen=set(); derived=np.zeros_like(initial); flux_error=0
        for row in ledger:
            context=row['evaluation_context']
            stages=[s for s in (0,1) if "StagePoint(name='ssprk2_stage_%d'"%s in context]
            assert len(stages)==1
            w=(d+r)@(initial if stages[0]==0 else predictor)
            cell,axis,side,component=[int(t.split(':')[1]) for t in row['quadrature_identity'].split('/')]
            occurrence=int(row['occurrence_identity'].rsplit(':',1)[1])
            assert 0<=cell<n and axis==0 and side in (0,1) and 0<=component<3 and occurrence in (0,1)
            key=(context,row['occurrence_identity'],row['quadrature_identity'])
            assert key not in seen; seen.add(key)
            c=int(order[component]); left,right=(cell,(cell+1)%n) if side else ((cell-1)%n,cell)
            flux=n*(w[c,right]-w[c,left]); sign=1 if side else -1
            weight=dt/2*(.5,1.5)[occurrence]
            assert row['multiplicity']==1 and row['face_measure']==1 and row['orientation']==sign
            assert row['temporal_weight']==weight
            flux_error=max(flux_error,abs(row['numerical_flux']-flux))
            assert abs(row['integrated_amount']-sign*flux*weight)<3e-13
            derived[c,cell]+=row['integrated_amount']
        errors={'state_error':float(np.max(np.abs(final-expected))),
                'flux_error':float(flux_error),
                'ledger_update_error':float(np.max(np.abs(derived-(final-initial)/n))),
                'saved_ledger_error':float(np.max(np.abs(derived-increment)))}
        assert errors['state_error']<3e-12 and errors['flux_error']<3e-12
        assert max(errors[k] for k in ('ledger_update_error','saved_ledger_error'))<3e-13
        matrix.append({'path':str(path.relative_to(base)),'sha256':sha(path),
                       'ledger_sha256':sha(ledger_path),'records':len(ledger),**errors})
    assert len(hall)==5 and len(matrix)==2
    return {'hall':hall,'matrix':matrix}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--serial',type=Path,required=True)
    parser.add_argument('--mpi',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results={'schema_version':1,'status':'passed','script_sha256':sha(Path(__file__)),
             'numpy_version':np.__version__,'serial':recheck(args.serial),'mpi2':recheck(args.mpi)}
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    for name in ('serial','mpi2'):
        rows=results[name]
        print(name, 'Hall saved error',max(r['discrete_error'] for r in rows['hall']),
              'ledger error',max(r['ledger_update_error'] for r in rows['matrix']))
