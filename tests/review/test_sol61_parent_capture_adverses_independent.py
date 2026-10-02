"""Independent Source schema counterexamples; no Native data or authority minted."""
import pytest
from tests.review.test_sol61_initial_ghost_selection import wire
from tests.python.support.initial_ghost_parent_capture import classify_parent_capture,CANONICAL_REFUSAL

@pytest.mark.parametrize('mutation',['bool-world','rank1-message','int-retry','partial-success','payload-with-refusal'])
def test_no_alias_or_alternative_error_can_certify_intermediate_disposition(mutation):
    exact=('RuntimeError',CANONICAL_REFUSAL,True)
    args=dict(payload=None,failures=(exact,exact),before_blob=wire(owners=(-1,)),world_size=2,phase='parent-rejected')
    if mutation=='bool-world':args['world_size']=True
    elif mutation=='rank1-message':args['failures']=(exact,('RuntimeError',CANONICAL_REFUSAL.replace('rank 0','rank 1'),True))
    elif mutation=='int-retry':args['failures']=(exact,('RuntimeError',CANONICAL_REFUSAL,1))
    elif mutation=='partial-success':args['failures']=(exact,None)
    else:args['payload']=args['before_blob']
    with pytest.raises(ValueError):classify_parent_capture(**args)
