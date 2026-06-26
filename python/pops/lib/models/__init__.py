"""pops.lib.models -- provided physical models (pure facade compositions).

Currently the moment models (:class:`pops.lib.models.moments.HyQMOM15` /
:class:`pops.lib.models.moments.Gaussian`).

DEFER (no generator to wrap): ``lib.models.fluids`` (Euler / IsothermalEuler) and
``lib.models.mhd`` (IdealMHD) -- there is no ``build_euler`` / ``build_mhd`` generator in
``physics``, so a model package there would invent surface (see the PR-D blueprint DEFER
list). They land when a fluids/mhd generator does.
"""
from .moments import HyQMOM15, Gaussian

__all__ = ["HyQMOM15", "Gaussian"]
