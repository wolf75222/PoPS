"""Lower exact mesh plans onto native runtime configuration seams.

This module contains only backend lowering.  Runtime selection lives in
``_runtime_executor`` and therefore never depends on historical target strings.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._amr_bind_lowering import amr_config_from_layout


def system_config_from_layout(layout: Any) -> Any:
    """Build the native uniform config from an authenticated layout descriptor."""
    from pops._bootstrap import SystemConfig

    mesh = layout.mesh
    cfg = SystemConfig()
    cfg.n = int(mesh.n)
    cfg.L = float(mesh.L)
    cfg.periodic = bool(mesh.periodic)
    return cfg


def flow_amr_layout(
    sim: Any,
    layout: Any,
    n_blocks: Any = 1,
    *,
    bind_schema: Any = None,
    params: Any = None,
) -> None:
    """Lower a typed AMR refinement criterion before native block installation."""
    criterion = getattr(layout, "refine", None)
    if criterion is not None:
        _apply_refine_criterion(
            sim,
            criterion,
            is_multiblock=n_blocks > 1,
            bind_schema=bind_schema,
            params=params,
        )


def flow_bootstrap_tagging(sim: Any, bootstrap: Any, params: Any) -> None:
    """Lower the resolved owner-qualified threshold indicator without heuristics."""
    from pops.mesh.amr import Above

    graph = bootstrap.tagging.graph
    if type(graph.refine) is not Above or graph.coarsen is not None:
        raise NotImplementedError(
            "pops.bind: native bootstrap currently lowers Above(density, threshold) "
            "without a coarsen root"
        )
    predicate = graph.refine
    if predicate.threshold not in params:
        raise ValueError("pops.bind: bootstrap tagging threshold is missing from resolved params")
    if predicate.indicator.block_ref is None:
        raise ValueError("pops.bind: bootstrap tag indicator must be block-qualified")
    sim._set_bootstrap_refinement(
        predicate.indicator.block_ref.local_id,
        predicate.indicator.local_id,
        float(params[predicate.threshold]),
        bootstrap.tagging.qualified_id,
    )


def _apply_refine_criterion(
    sim: Any,
    criterion: Any,
    is_multiblock: bool = False,
    *,
    bind_schema: Any = None,
    params: Any = None,
) -> None:
    """Lower one authenticated refinement criterion to native AMR seams."""
    from pops.mesh.amr import Refine, TagUnion

    if isinstance(criterion, TagUnion):
        for child in criterion.criteria:
            _apply_refine_criterion(
                sim,
                child,
                is_multiblock=is_multiblock,
                bind_schema=bind_schema,
                params=params,
            )
        return
    if not isinstance(criterion, Refine):
        raise TypeError(
            "pops.bind: AMR refine criterion must be a pops.mesh.amr.Refine / TagUnion "
            "(got %r)" % type(criterion).__name__
        )
    if not getattr(criterion, "references_authenticated", False):
        raise ValueError(
            "pops.bind: Refine criterion references were not authenticated by Case.resolve; "
            "run it through pops.compile(problem, layout=...) instead of attaching a raw or "
            "canonical-looking Handle directly to a compiled/runtime layout"
        )
    threshold = criterion.threshold
    if threshold is None:
        raise ValueError(
            "pops.bind: Refine criterion has no threshold "
            "(use Refine.on(subject).above(value))"
        )
    threshold = _refine_threshold_value(threshold, bind_schema, params)

    from pops.model import Handle

    if not isinstance(criterion.subject, Handle):
        raise NotImplementedError(
            "pops.bind: [amr:expression_indicator unavailable] Refine subject %s is a semantic "
            "indicator expression. Its Handle leaves were validated and resolved at compile, but "
            "the current native AMR runtime only lowers direct declaration Handle selectors and "
            "the dedicated potential-gradient predicate. Add the expression-indicator backend "
            "capability before running this criterion; it is never flattened to a variable name."
            % type(criterion.subject).__name__
        )
    subject = _refine_subject_name(criterion.subject)
    if criterion.predicate == "gradient_above" and subject in (
        "phi",
        "grad phi",
        "potential",
    ):
        sim.set_phi_refinement(float(threshold))
        return
    if _is_default_density_subject(subject):
        sim.set_refinement(float(threshold))
        return
    if not is_multiblock:
        raise NotImplementedError(
            "pops.bind: refining on %r is a multi-block AMR feature; the single-block AMR route "
            "refines on the density (component 0) only. Refine on the density "
            "(Refine.on(Density).above(...)), or use the |grad phi| tag "
            "(Refine.on(phi).gradient_above(...))." % (subject,)
        )
    sim.set_refinement(float(threshold), variable=subject)


def _refine_threshold_value(threshold: Any, schema: Any, params: Any) -> Any:
    """Resolve one canonical parameter threshold from the effective bind mapping."""
    from pops.ir import ValueExpr
    from pops.model import ParamHandle

    handle = threshold.handle if isinstance(threshold, ValueExpr) else threshold
    if not isinstance(handle, ParamHandle):
        return threshold
    if schema is None:
        raise ValueError("pops.bind: parameterized AMR threshold requires BindSchema")
    slot = schema.slot(handle)
    if slot.handle not in (params or {}):
        raise ValueError("pops.bind: resolved params are missing AMR threshold %s" % slot.qid)
    return params[slot.handle]


def _refine_subject_name(subject: Any) -> Any:
    """Lower one canonical Handle to the native variable token at the runtime boundary."""
    from pops.model import Handle

    if not isinstance(subject, Handle):
        raise TypeError(
            "pops.bind: Refine subject must be a resolved pops.model.Handle, got %r; strings "
            "are not declaration identities" % type(subject).__name__
        )
    if not subject.is_resolved:
        raise ValueError(
            "pops.bind: Refine subject %s is still authoring-owned; compile must resolve every "
            "reference through Case.resolve before runtime lowering" % subject.qualified_id
        )
    return subject.local_id


def _is_default_density_subject(subject: Any) -> bool:
    """Return whether the subject denotes native component-zero density."""
    if subject is None:
        return True
    return subject in ("Density", "density", "rho", "n", "ne")


__all__ = [
    "amr_config_from_layout",
    "flow_amr_layout",
    "flow_bootstrap_tagging",
    "system_config_from_layout",
]
