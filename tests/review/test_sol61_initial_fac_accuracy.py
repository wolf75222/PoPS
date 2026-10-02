"""Independent operator and declared stopping-budget checks; Source only."""
from pathlib import Path
import sys
import numpy as np
import pops
from tests.review.sol61_initial_fac_operator_audit import residuals
from tests.python.support.initial_field_ghost_native_case import build,FIELD_FORCING_BOUND,FIELD_RESIDUAL_BUDGET,FIELD_RELATIVE_TOL
from tests.python.support.initial_field_ghost_native_oracle import ORIGINAL_F_BOUND,CONTRACT

def test_constant_operator_neumann_and_composite_partition():
    ca=np.ones((8,8),dtype=bool);ca[:,2:6]=False;fa=np.repeat(np.repeat(~ca,2,0),2,1)
    fields=[(np.full((8,8),2.),ca),(np.full((16,16),2.),fa)]
    r=residuals(fields,[np.full((8,8),2.),np.full((16,16),2.)])
    assert all(np.max(np.abs(row[mask]))==0. for row,(_,mask) in zip(r,fields))
    perturbed=[(row.copy(),mask) for row,mask in fields];perturbed[1][0][5,5]+=1e-8
    bad=residuals(perturbed,[np.full((8,8),2.),np.full((16,16),2.)])
    assert max(np.nanmax(np.abs(row)) for row in bad)>ORIGINAL_F_BOUND

def test_real_source_plan_authenticates_tighter_existing_relative_policy():
    root=Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve().is_relative_to(root/'python') and 'pops._pops' not in sys.modules
    case,layout=build();plan=pops.resolve(pops.validate(case),layout=layout);plan.verify()
    field,=plan.field_plans.values();options=field.native_options
    assert FIELD_FORCING_BOUND==16.25
    assert FIELD_RELATIVE_TOL*FIELD_FORCING_BOUND<=FIELD_RESIDUAL_BUDGET
    assert FIELD_RESIDUAL_BUDGET==ORIGINAL_F_BOUND/8
    assert CONTRACT.endswith('@2')
    # Authenticated provider descriptor must actually carry this policy; no test-only display.
    assert FIELD_RELATIVE_TOL==options['solver_provider']['resolution']['native_contract']['options']['fac.rel_tol']
    assert options['solver_provider']['resolution']['native_contract']['options']['mg.rel_tol']==1e-8
    assert options['solver_provider']['resolution']['native_contract']['options']['fac.abs_tol']==0.
