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
    """Compile one authenticated tagging graph to the native data-only VM."""
    data = bootstrap.tagging.runtime_tagging_data(params)
    if type(data) is not dict or data.get("schema_version") != 1 \
            or data.get("graph_type") != "amr_tagging_runtime":
        raise ValueError("pops.bind: tagging provider returned an unsupported runtime manifest")

    registrations = {}
    for row in data.get("lowerings", ()):
        if type(row) is not dict or row.get("schema_version") != 1:
            raise ValueError("pops.bind: malformed tagging lowering registration")
        node_type = row.get("node_type")
        lowering = row.get("lowering", {})
        if not isinstance(node_type, str) or not node_type \
                or lowering.get("kind") != "tag_lowering" \
                or lowering.get("local_id") != node_type:
            raise ValueError("pops.bind: unauthenticated tagging lowering registration")
        if node_type in registrations:
            raise ValueError("pops.bind: duplicate tagging lowering registration")
        registrations[node_type] = lowering.get("qualified_id")

    leaves: list[tuple[str, str, int, float]] = []

    def compile_node(node: Any) -> tuple[list[int], list[int]]:
        if type(node) is not dict or node.get("schema_version") != 1:
            raise ValueError("pops.bind: malformed tagging expression node")
        node_type = node.get("node_type")
        if node_type not in registrations:
            raise ValueError("pops.bind: tagging node lacks an authenticated lowering")
        leaf_op = _TAG_LEAF_OPS.get(node_type)
        if leaf_op is not None:
            indicator = node.get("indicator")
            if type(indicator) is not dict or indicator.get("kind") not in {"state", "field"}:
                raise TypeError("pops.bind: native tag leaves require a state/field Handle")
            block = indicator.get("block_ref")
            if type(block) is not dict or not isinstance(block.get("local_id"), str):
                raise ValueError("pops.bind: native tag leaves must be block-qualified")
            variable = node.get("variable", indicator.get("local_id"))
            threshold = node.get("threshold")
            if not isinstance(variable, str) or not variable \
                    or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise TypeError("pops.bind: malformed native tag leaf")
            leaves.append((block["local_id"], variable, leaf_op, float(threshold)))
            return [leaf_op], [len(leaves) - 1]

        logical_op = _TAG_LOGICAL_OPS.get(node_type)
        if logical_op is None:
            raise NotImplementedError(
                "pops.bind: native tagging provider %r is registered but has no VM opcode"
                % node_type
            )
        children = node.get("children")
        if node_type == "not":
            children = (node.get("child"),)
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError("pops.bind: logical tagging node has no children")
        ops: list[int] = []
        args: list[int] = []
        for child in children:
            child_ops, child_args = compile_node(child)
            ops.extend(child_ops)
            args.extend(child_args)
        ops.append(logical_op)
        args.append(len(children))
        return ops, args

    refine_ops, refine_args = compile_node(data["refine"])
    coarsen = data.get("coarsen")
    coarsen_ops, coarsen_args = ([], []) if coarsen is None else compile_node(coarsen)
    hysteresis = data.get("hysteresis")
    if type(hysteresis) is not dict or hysteresis.get("hysteresis_type") != "min_cycles":
        raise ValueError("pops.bind: unsupported AMR hysteresis manifest")
    min_cycles = hysteresis.get("min_cycles")
    if isinstance(min_cycles, bool) or not isinstance(min_cycles, int) or min_cycles < 0:
        raise ValueError("pops.bind: AMR hysteresis min_cycles must be an integer >= 0")
    sim._set_bootstrap_tagging(
        [row[0] for row in leaves],
        [row[1] for row in leaves],
        [row[2] for row in leaves],
        [row[3] for row in leaves],
        refine_ops,
        refine_args,
        coarsen_ops,
        coarsen_args,
        min_cycles,
        str(hysteresis.get("equality")),
        str(data.get("conflict_policy")),
        bootstrap.tagging.qualified_id,
    )


# Stable native VM ABI. New node implementations register one opcode here; the graph compiler never
# dispatches on Python classes and the native loop never calls Python.
_TAG_LEAF_OPS = {
    "above": 1,
    "below": 2,
    "magnitude_above": 3,
    "gradient_above": 4,
    "gradient_below": 5,
}
_TAG_LOGICAL_OPS = {"any_of": 16, "all_of": 17, "not": 18}


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
