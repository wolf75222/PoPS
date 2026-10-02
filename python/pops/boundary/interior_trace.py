"""Explicit primary-State interior trace for public pointwise Ghost expressions."""
from pops.fields.boundary_values import BoundaryValue


class InteriorTrace(BoundaryValue):
    """Nearest valid cell along the active boundary axes, as provided by GhostV1.

    This is a State trace, never a Field face/ghost sampling conversion. The native
    request already carries this authenticated packed view for its primary state.
    """
    def __init__(self,state,component=None):
        if getattr(state,'kind',None)!='state':
            raise TypeError('InteriorTrace requires a State; Field support cannot become an interior State trace')
        super().__init__(state,component)

    def resolve_references(self,resolver):
        resolved=super().resolve_references(resolver)
        return InteriorTrace(resolved.handle,resolved.component)

    def __pops_ir_key__(self,recurse):
        return ('interior_trace',self.handle.qualified_id,self.component)


def interior_trace(state,component=None):
    return InteriorTrace(state,component)
