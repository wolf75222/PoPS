"""An independently solved Field history remapped by the same finite Python algebra."""
from .continuation_regrid_amr import build_continuation_regrid_amr


def build_independent_continuation_regrid(**kwargs):
    return build_continuation_regrid_amr(independent_previous=True, **kwargs)
