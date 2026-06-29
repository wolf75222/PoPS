"""Lower typed AMR layout descriptors to native runtime configuration.

This module contains the useful AMR-specific lowering that used to live in the
legacy Case orchestration wrapper. It is not a public compile/bind API: callers
construct an ``AmrSystem`` and then use ``sim.install(...)``.
"""


def amr_config_from_layout(layout):
    """Build an ``AmrSystemConfig`` from a :class:`pops.mesh.layouts.AMR` descriptor."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import FrozenRegrid, PatchLayout, RegridEvery

    layout.validate()
    base = layout.base
    cfg = AmrSystemConfig()
    cfg.n = int(base.n)
    cfg.L = float(base.L)
    cfg.periodic = bool(base.periodic)

    regrid = layout.regrid
    if isinstance(regrid, RegridEvery):
        cfg.regrid_every = int(regrid.steps)
    elif regrid is None or isinstance(regrid, FrozenRegrid):
        cfg.regrid_every = 0
    else:
        raise TypeError(
            "AMR.regrid must be a pops.mesh.amr.RegridEvery(n) / FrozenRegrid() "
            "(got %r)" % type(regrid).__name__)

    patches = layout.patches
    if isinstance(patches, PatchLayout):
        cfg.distribute_coarse = bool(patches.distribute_coarse)
        cfg.coarse_max_grid = int(patches.coarse_max_grid)
    elif patches is not None:
        raise TypeError(
            "AMR.patches must be a pops.mesh.amr.PatchLayout(...) (got %r)"
            % type(patches).__name__)
    return cfg


def flow_amr_layout(sim, layout, n_blocks=1):
    """Flow a typed AMR refinement descriptor onto an ``AmrSystem`` instance."""
    criterion = getattr(layout, "refine", None)
    if criterion is not None:
        _apply_refine_criterion(sim, criterion, is_multiblock=n_blocks > 1)


def _apply_refine_criterion(sim, criterion, is_multiblock=False):
    """Lower one typed refinement criterion to native refinement calls."""
    from pops.mesh.amr import Refine, TagUnion

    if isinstance(criterion, TagUnion):
        for c in criterion.criteria:
            _apply_refine_criterion(sim, c, is_multiblock=is_multiblock)
        return
    if not isinstance(criterion, Refine):
        raise TypeError(
            "AMR refine criterion must be a pops.mesh.amr.Refine / TagUnion (got %r)"
            % type(criterion).__name__)
    threshold = criterion.threshold
    if threshold is None:
        raise ValueError(
            "Refine criterion has no threshold (use Refine.on(subject).above(value))")
    subject = _refine_subject_name(criterion.subject)
    if criterion.predicate == "gradient_above" and subject in ("phi", "grad phi", "potential"):
        sim.set_phi_refinement(float(threshold))
        return
    if is_default_density_subject(subject):
        sim.set_refinement(float(threshold))
        return
    sim.set_refinement(float(threshold), variable=subject)


def _refine_subject_name(subject):
    """The plain string name of a Refine subject."""
    if isinstance(subject, str):
        return subject
    name = getattr(subject, "name", None)
    return name if isinstance(name, str) else None


def is_default_density_subject(subject):
    """True when a Refine subject names the density / component 0 default."""
    if subject is None:
        return True
    return subject in ("Density", "density", "rho", "n", "ne")


__all__ = ["amr_config_from_layout", "flow_amr_layout", "is_default_density_subject"]
