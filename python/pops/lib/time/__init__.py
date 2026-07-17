"""Canonical pre-implemented time Programs.

Every factory returns an ordinary :class:`pops.Program` built from the same
public operations available to user code.  Factories take exact block-owned
state handles and exact model operator handles; they do not select physics by
name, boolean flags, hidden defaults, or a preset-specific runtime route.
"""

from .euler import FORWARD_EULER_TABLEAU, ForwardEuler
from .imex import IMEX, IMEX_EULER_TABLEAU
from .multistep import AdamsBashforth, BDF
from .predictor_corrector import PredictorCorrector
from .rk import ButcherTableau, RK4, RK4_TABLEAU, RungeKutta, SSPRK2_TABLEAU
from .ssprk import SSPRK2, SSPRK3, SSPRK3_TABLEAU
from .strang import Lie, Strang

# Importing submodules installs them as package attributes.  Two historical
# lowercase callables had those exact names; remove the ambiguous package
# attributes so only the final factories form the public surface.  Explicit
# ``import pops.lib.time.rk`` remains normal Python module access.
globals().pop("rk", None)
globals().pop("strang", None)

__all__ = [
    "AdamsBashforth",
    "BDF",
    "ButcherTableau",
    "FORWARD_EULER_TABLEAU",
    "ForwardEuler",
    "IMEX",
    "IMEX_EULER_TABLEAU",
    "Lie",
    "PredictorCorrector",
    "RK4",
    "RK4_TABLEAU",
    "RungeKutta",
    "SSPRK2",
    "SSPRK2_TABLEAU",
    "SSPRK3",
    "SSPRK3_TABLEAU",
    "Strang",
]
