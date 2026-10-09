"""Independent synthetic selection identities; never Native carrier receipts."""
from copy import deepcopy
import pytest
from tests.python.support.initial_ghost_failure_selection import select_xmin_owner,require_selection_agreement
from tests.review.test_sol61_initial_ghost_selection import wire


@pytest.mark.parametrize("mutation",["boolean_target","nonowner","empty","boolean_world","upper_hash","duplicate_owner","row_boolean","zero_world"])
def test_owned_selection_refuses_new_forged_identities(mutation):
    valid=select_xmin_owner(wire(owners=(0,)),2)
    expected=deepcopy(valid);rows=[deepcopy(valid),deepcopy(valid)]
    if mutation=="boolean_target":expected["target"]=False
    if mutation=="nonowner":expected["target"]=1
    if mutation=="empty":expected["eligible"]=[]
    if mutation=="boolean_world":expected["world_size"]=True
    if mutation=="upper_hash":expected["carrier_sha256"]="A"*64
    if mutation=="duplicate_owner":expected["eligible"]=[0,0]
    if mutation=="row_boolean":rows[1]["target"]=False
    if mutation=="zero_world":expected["world_size"]=0
    with pytest.raises(ValueError):require_selection_agreement(rows,expected)


def test_real_zero_owner_and_replicated_owner_are_nonvacuous():
    zero=select_xmin_owner(wire(owners=(0,)),2)
    assert zero["eligible"]==[0] and zero["target"]==0
    assert require_selection_agreement([zero,zero],zero)["agreed"]==zero
    replicated=select_xmin_owner(wire(owners=(-1,)),2)
    assert replicated["eligible"]==[0,1] and replicated["target"]==1
