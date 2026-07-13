"""Exact lowering of the public AMR layout into the legacy native config."""
from __future__ import annotations

from typing import Any


def amr_config_from_layout(layout: Any, *, hierarchy: Any = None) -> Any:
    """Build ``AmrSystemConfig`` without dropping authored hierarchy semantics."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import (
        FrozenRegrid,
        LEGACY_CONFIG_LEVELS,
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

    if hierarchy is not None:
        from pops.mesh.amr import ResolvedHierarchy

        if type(hierarchy) is not ResolvedHierarchy:
            raise TypeError("pops.bind: hierarchy must be an exact ResolvedHierarchy")
        transitions = hierarchy.plan.transitions
        if any(row.dimension != 2 or row.ratio != (2, 2) for row in transitions):
            raise NotImplementedError(
                "pops.bind: native AMR requires exact two-dimensional ratio-(2,2) transitions"
            )
        buffers = {row.buffer for row in transitions}
        lookaheads = {row.lookahead for row in transitions}
        if len(buffers) != 1 or len(lookaheads) != 1:
            raise NotImplementedError(
                "pops.bind: native AMR exposes one global nesting buffer/lookahead; "
                "varying per-transition values cannot be lowered exactly"
            )
        buffer = next(iter(buffers))
        if len(set(buffer)) != 1:
            raise NotImplementedError(
                "pops.bind: native AMR cannot lower anisotropic nesting buffers"
            )
        cfg.level_count = hierarchy.plan.level_count
        cfg.regrid_margin = buffer[0]
        cfg.regrid_grow = next(iter(lookaheads))
        cfg.explicit_bootstrap = True
        cluster = hierarchy.plan.clustering.options.to_data()
        patches = hierarchy.plan.patch_generation.options.to_data()
        balance = hierarchy.plan.load_balance.options.to_data()
        if set(cluster) - {
            "native_route", "minimum_efficiency", "minimum_box_size", "maximum_box_size"
        }:
            raise ValueError("pops.bind: unsupported native clustering option")
        if set(patches) - {"native_route", "distribute_coarse", "coarse_max_grid"}:
            raise ValueError("pops.bind: unsupported native patch-generation option")
        if set(balance) != {"native_route"}:
            raise ValueError("pops.bind: round_robin load balance accepts no extra options")
        cfg.cluster_min_efficiency = float(cluster.get("minimum_efficiency", 0.0))
        cfg.cluster_min_box_size = int(cluster.get("minimum_box_size", 0))
        cfg.cluster_max_box_size = int(cluster.get("maximum_box_size", 0))
        cfg.distribute_coarse = bool(patches.get("distribute_coarse", False))
        cfg.coarse_max_grid = int(patches.get("coarse_max_grid", 0))

    # AmrSystemConfig has no hierarchy-transition payload. Refuse intent it cannot transport; the
    # N-level native bootstrap is exposed only after resolved transitions reach the binding seam.
    if hierarchy is None and int(layout.max_levels) != LEGACY_CONFIG_LEVELS:
        raise NotImplementedError(
            "pops.bind: AMR(max_levels=%d) cannot be lowered by AmrSystemConfig; the current "
            "public native adapter requires exactly max_levels=%d until resolved hierarchy "
            "transitions are transported (no silent level-count substitution)"
            % (layout.max_levels, LEGACY_CONFIG_LEVELS)
        )
    if hierarchy is not None and int(layout.max_levels) != hierarchy.plan.level_count:
        raise ValueError(
            "pops.bind: layout max_levels and ResolvedHierarchy level_count disagree"
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
    if hierarchy is None:
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
    if hierarchy is not None:
        pass
    elif isinstance(clustering, PatchClustering):
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
