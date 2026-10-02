"""Analytic Source witnesses for independent offline reader; no Native data minted."""
import json, numpy as np, pytest
from tests.review import sol61_m17_saved_physics_offline as r

def test_closed_wick_gaussian_flux_and_zero_regularization():
    u,v,a,b,c=.3,-.2,.7,.13,.5;rho=1.7
    raw=np.array([rho*r.gaussian(p,q,u,v,a,b,c) for p,q in r.I])
    expected=np.array([rho*r.gaussian(p+1,q,u,v,a,b,c) for p,q in r.I])
    np.testing.assert_allclose(r.flux(raw),expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(r.product(raw,np.linspace(-.02,.03,15)),0,rtol=0,atol=2e-14)

def test_non_gaussian_mixture_fourier_rhs_is_conservative_on_ten_rows():
    def mixture(weight):
        return np.array([weight*r.gaussian(p,q,.4,-.2,.6,.08,.5)+(1-weight)*r.gaussian(p,q,-.3,.25,.7,-.07,.6) for p,q in r.I])
    line=np.stack([mixture(.45+.05*np.sin(2*np.pi*k/16)) for k in range(16)],axis=1)
    a,b=r.step(line,24),r.step(line,48)
    assert np.max(np.abs(a-b))<3e-13
    for k,index in enumerate(r.I):
        if sum(index)<4:assert abs(a[k].mean()-line[k].mean())<3e-13
    assert np.max(np.abs(a-line))>1e-8

def test_json_duplicate_and_nonfinite_refused(tmp_path):
    p=tmp_path/'input.json'
    for text in ('{"dt":1,"dt":2}','{"dt":NaN}'):
        p.write_text(text)
        with pytest.raises(ValueError):r.load(p)

def test_invalid_rho_and_covariance_refuse():
    for raw in (np.zeros(15),np.ones(15)):
        with np.errstate(divide='ignore',invalid='ignore'):
            with pytest.raises(ValueError):r.primitives(raw)


def test_distinct_combinatorial_formula_matches_existing_generating_oracle():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'examples/migration/scientific/api040_m17_oracle.py'
    spec=importlib.util.spec_from_file_location('source_only_crosscheck',path)
    other=importlib.util.module_from_spec(spec);spec.loader.exec_module(other)
    raw=np.array([.6*r.gaussian(p,q,.4,-.2,.6,.08,.5)+.4*r.gaussian(p,q,-.3,.25,.7,-.07,.6) for p,q in r.I])
    jump=np.linspace(-.025,.019,15)
    np.testing.assert_allclose(r.flux(raw),other.flux(raw,(1.,0.)),rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(r.product(raw,jump),other.primary_product(raw,jump,(1.,0.)),rtol=2e-13,atol=2e-13)
