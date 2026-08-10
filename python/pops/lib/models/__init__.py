"""pops.lib.models -- provided physical models (pure facade compositions).

Currently this contains the explicitly 2V/2D moment specializations
(:class:`pops.lib.models.moments.HyQMOM15` / :class:`pops.lib.models.moments.Gaussian`) and
the exact-rank Cartesian electrostatic-Lorentz authoring helper
(:func:`author_electrostatic_lorentz`, ADC-637) consumed by the generic condensed-implicit
operator path.

DEFER (no generator to wrap): ``lib.models.fluids`` (Euler / IsothermalEuler) and
``lib.models.mhd`` (IdealMHD) -- there is no ``build_euler`` / ``build_mhd`` generator in
``physics``, so a model package there would invent surface (see the PR-D blueprint DEFER
list). They land when a fluids/mhd generator does.
"""
from .electrostatic_lorentz import LORENTZ_J_NAME, author_electrostatic_lorentz
from .moments import HyQMOM15, Gaussian

__all__ = ["HyQMOM15", "Gaussian", "LORENTZ_J_NAME", "author_electrostatic_lorentz"]
