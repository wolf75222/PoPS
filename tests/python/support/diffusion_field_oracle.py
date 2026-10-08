"""Independent FV Fourier oracle. Imports neither PoPS nor author_case."""
import json
from pathlib import Path
import numpy as np

def references(inputs,method):
    nx,ny=inputs['cells'];dt=inputs['dt'];kappa=inputs['kappa']
    x=(np.arange(nx,dtype=np.float64)+.5)/nx
    mode=np.broadcast_to(np.sinc(1/nx)*np.sin(2*np.pi*x),(1,ny,nx))
    lam=4*nx**2*np.sin(np.pi/nx)**2
    z=dt*kappa*lam
    euler=1-z
    accepted=euler if method=='euler' else 1-z+z*z/2
    b,a=inputs['mean'],inputs['amplitude']
    u0=b+a*mode
    Y=b+a*euler*mode
    u=b+a*accepted*mode
    coeff=inputs['rhs_offset']+inputs['field_stage_fraction']*dt
    phi=coeff*(b+a*euler*mode/(1+lam))
    wrong_live=coeff*(b+a*mode/(1+lam))
    wrong_time=inputs['rhs_offset']*(b+a*euler*mode/(1+lam))
    wrong_final=coeff*(b+a*accepted*mode/(1+lam))
    round_bound=4096*np.finfo(np.float64).eps*(abs(b)+abs(a))
    field_bound=nx*ny*abs(coeff)*(abs(b)+abs(a))*inputs['solver_rtol']+inputs['solver_atol']+round_bound
    assert 2*kappa*dt*(nx*nx+ny*ny)<1, 'predeclared explicit diffusion stability condition'
    assert np.max(np.abs(phi-wrong_live))>100*field_bound
    assert np.max(np.abs(phi-wrong_time))>100*field_bound
    if method=='ssprk2':assert np.max(np.abs(phi-wrong_final))>100*field_bound
    return dict(initial=u0,predictor=Y,accepted=u,phi_stage=phi,
        wrong_accepted_at_stage=wrong_live,wrong_field_time=wrong_time,wrong_final_state=wrong_final,
        lambda_h=float(lam),z=float(z),euler_gain=float(euler),accepted_gain=float(accepted),
        state_bound=float(round_bound),field_bound=float(field_bound))

def check_saved(folder):
    folder=Path(folder);inputs=json.loads((folder/'inputs.json').read_text())
    r=references(inputs,inputs['method'])
    results={}
    for key in ['initial','predictor','accepted','phi_stage']:
        actual=np.load(folder/(key+'.npy'),allow_pickle=False)
        expected=r[key]
        assert actual.shape==expected.shape and actual.dtype==np.dtype('float64')
        assert np.isfinite(actual).all()
        bound=r['field_bound'] if key=='phi_stage' else r['state_bound']
        error=float(np.max(np.abs(actual-expected)))
        assert error<=bound,(key,error,bound)
        results[key]=dict(max_abs_error=error,bound=bound)
    return dict(status='PASS',Source=inputs['Source'],Native=inputs['Native'],method=inputs['method'],
        actual_context_scope=inputs['actual_context_scope'],lambda_h=r['lambda_h'],
        euler_gain=r['euler_gain'],accepted_gain=r['accepted_gain'],comparisons=results,
        wrong_live_state_field_separation=float(np.max(np.abs(r['phi_stage']-r['wrong_accepted_at_stage']))),
        wrong_time_field_separation=float(np.max(np.abs(r['phi_stage']-r['wrong_field_time']))),
        wrong_final_state_field_separation=float(np.max(np.abs(r['phi_stage']-r['wrong_final_state']))))

if __name__=='__main__':
    import sys
    print(json.dumps(check_saved(sys.argv[1]),indent=2))
