"""Exact lowering of the public AMR layout into the legacy native config."""
from __future__ import annotations

from typing import Any


def amr_config_from_layout(layout: Any) -> Any:
    """Build ``AmrSystemConfig`` without dropping authored hierarchy semantics."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import (
        FrozenRegrid,
        NATIVE_MAX_LEVELS,
        NATIVE_RATIOS,
        PatchClustering,
        PatchLayout,
        RegridEvery,
    )

    base = layout.base
    cfg: Any = AmrSystemConfig()
    cfg.n = int(base.n)
    cfg.L = float(base.L)
    cfg.periodic = bool(base.periodic)

    # AmrSystemConfig has no hierarchy-transition payload. Refuse intent it cannot transport; the
    # N-level native bootstrap is exposed only after resolved transitions reach the binding seam.
    if int(layout.max_levels) != NATIVE_MAX_LEVELS:
        raise NotImplementedError(
            "pops.bind: AMR(max_levels=%d) cannot be lowered by AmrSystemConfig; the current "
            "public native adapter requires exactly max_levels=%d until resolved hierarchy "
            "transitions are transported (no silent level-count substitution)"
            % (layout.max_levels, NATIVE_MAX_LEVELS)
        )
    if int(layout.ratio) not in NATIVE_RATIOS:
        raise NotImplementedError(
            "pops.bind: AMR ratio %d is unsupported; native ratio-2 kernels cannot honor this "
            "transition exactly" % layout.ratio
        )
    if layout.nesting is not None:
        raise NotImplementedError(
            "pops.bind: AMR.nesting is not transported by AmrSystemConfig; refusing the policy "
            "before runtime instead of ignoring its buffer/lookahead requirements"
        )

    regrid = layout.regrid
    if isinstance(regrid, RegridEvery):
        cfg.regrid_every = int(regrid.steps)
    elif regrid is None or isinstance(regrid, FrozenRegrid):
        cfg.regrid_every = 0
    else:
        raise TypeError(
            "pops.bind: AMR.regrid must be a pops.mesh.amr.RegridEvery(n) / FrozenRegrid() "
            "(got %r)" % type(regrid).__name__
        )

    patches = layout.patches
    if isinstance(patches, PatchLayout):
        cfg.distribute_coarse = bool(patches.distribute_coarse)
        cfg.coarse_max_grid = int(patches.coarse_max_grid)
    elif patches is not None:
        raise TypeError(
            "pops.bind: AMR.patches must be a pops.mesh.amr.PatchLayout(...) (got %r)"
            % type(patches).__name__
        )

    # Zero values preserve the native Berger-Rigoutsos defaults.
    clustering = getattr(layout, "clustering", None)
    if isinstance(clustering, PatchClustering):
        cfg.cluster_min_efficiency = float(clustering.min_efficiency)
        cfg.cluster_min_box_size = int(clustering.min_box_size)
        cfg.cluster_max_box_size = int(clustering.max_box_size)
    elif clustering is not None:
        raise TypeError(
            "pops.bind: AMR.clustering must be a pops.mesh.amr.PatchClustering(...) (got %r)"
            % type(clustering).__name__
        )
    return cfg


__all__ = ["amr_config_from_layout"]
