"""pops.codegen.program_emit_params : compiled-Program RUNTIME-parameter routing + metadata (ADC-510).

Extracted from ``pops.codegen.program_codegen`` so each emitter module fits the Spec-4 line budget.
A compiled time Program whose physics reads ``m.value(m.param(RuntimeParam(...)))`` carries the value in
a per-PROGRAM-block ``pops::RuntimeParams`` owned by the System (not the .so closure), so the value can
be changed at run time WITHOUT recompiling (Spec 5 C5). This module computes the param ROUTING (which
program block reads which runtime parameter, at which stable index, with which default) and emits the
``pops_program_param_*`` metadata table the .so exports. The SAME ``_program_param_entries`` drives the
C++ install-time seed (System::install_program), the Python bind-time route
(System._install_program_params via the artifact BindSchema) and this metadata, so all
three agree byte-for-byte. The per-cell read of the parameter (``params.get(index)`` bound from
``ctx.program_params(block)``) lives in ``program_emit_kernels`` / ``program_emit_model_kernels``.
"""
from __future__ import annotations

import json
from typing import Any

from pops.codegen.program_emit_kernels import _has_runtime_param, _model_impl
from pops.codegen.program_models import ProgramModelGraph, model_for_node


_MODEL_PARAM_OPS = frozenset({
    "source", "apply", "solve_local_linear", "rhs", "solve_local_nonlinear",
})


def _required_param_block_index(block_idx: Any, block: Any, value: Any) -> int:
    """Route a parameter-reading node through an exact Program block declaration."""
    if block is None:
        raise ValueError(
            "runtime parameter node %r is not block-qualified" % getattr(value, "name", value))
    try:
        index = block_idx[block]
    except KeyError:
        raise ValueError(
            "runtime parameter node %r references block %r outside Program._block_indices()"
            % (getattr(value, "name", value), block)) from None
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("invalid runtime parameter block index %r" % index)
    return index


def _op_model_exprs(impl: Any, v: Any) -> list:
    """The model coefficient Expr a model-kernel op @p v lowers, so the param routing can detect which
    runtime parameters it reads (ADC-510). Mirrors the kernel emitters' expr selection:
      - ``source`` / ``rhs`` named source -> impl._source_terms[name];
      - ``apply`` / ``solve_local_linear`` -> the linear_source matrix (flat);
      - ``rhs`` named fluxes -> impl._flux_terms[name]['x'/'y'];
      - ``solve_local_nonlinear`` -> the residual sub-block's source / apply terms.
    Returns a flat Expr list (possibly empty). @p impl is the HyperbolicModel."""
    out = []
    src = getattr(impl, "_source_terms", {}) or {}
    lin = getattr(impl, "_linear_sources", {}) or {}
    flux = getattr(impl, "_flux_terms", {}) or {}
    if v.op == "source":
        out += list(src.get(v.attrs.get("source"), []))
    elif v.op == "apply" or v.op == "solve_local_linear":
        for row in lin.get(v.attrs.get("linear_source"), []):
            out += list(row)
    elif v.op == "rhs":
        for s in (v.attrs.get("sources") or []):
            if s != "default":
                out += list(src.get(s, []))
        for f in (v.attrs.get("fluxes") or []):
            if f != "default" and f in flux:
                out += list(flux[f].get("x", [])) + list(flux[f].get("y", []))
    elif v.op == "solve_local_nonlinear":
        for w in v.attrs.get("residual_block", []):
            if w.op == "source":
                out += list(src.get(w.attrs.get("source"), []))
            elif w.op == "apply":
                for row in lin.get(w.attrs.get("linear_source"), []):
                    out += list(row)
    return out


def _runtime_param_refs_in(exprs: Any) -> tuple[Any, ...]:
    """Runtime parameter references in deterministic qualified-identity order."""
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children
    refs = {}
    stack = list(exprs)
    while stack:
        e = stack.pop()
        if isinstance(e, RuntimeParamRef):
            handle = getattr(e, "handle", None)
            key = getattr(handle, "qualified_id", None) or e.name
            refs.setdefault(key, e)
        else:
            stack.extend(_children(e))
    return tuple(refs[key] for key in sorted(refs))


def _qualified_param_identity(ref: Any, block: Any, *, graph_aware: bool) -> tuple[Any, str]:
    """Authenticate one read against its block/model owner and return ``(owner, qualified_id)``."""
    owner = block.model_owner_path.canonical()
    handle = getattr(ref, "handle", None)
    if handle is None:
        if graph_aware:
            raise ValueError(
                "runtime parameter %r in block %r has no owner-qualified ParamHandle"
                % (ref.name, block.local_id))
        return owner, ref.name
    actual = handle.owner_path.canonical()
    if actual != owner:
        raise ValueError(
            "runtime parameter %r in block %r belongs to model owner %s, not %s"
            % (ref.name, block.local_id, actual, owner))
    return owner, handle.qualified_id


def program_param_entries(program: Any, model: Any) -> list:
    """Per (PROGRAM block, runtime parameter) entries the .so exports + Python routes (ADC-510).

    Walk the Program ops (descending into control-flow / apply / residual sub-blocks); for each
    model-coefficient op that READS a runtime parameter, record one entry (program block index,
    parameter name, stable WITHIN-block index, declaration default). The within-block index is the
    model's STABLE runtime index (sorted-name order, assigned by assign_runtime_indices, matching the
    lowered ``params.get(idx)`` and the per-block RuntimeParams the System seeds / sets). Once one
    kernel reads a parameter, the complete stable table for that block is emitted: omitting an unused
    lower index would create an ABI hole before a later ``params.get(idx)``. Deduplicated by
    (block, name); sorted by (block, index) so the metadata table and Python routing agree on order.
    Empty when no model kernel reads a runtime param. @p model the physical model the Program lowers
    (None -> no entries)."""
    if model is None:
        return []
    graph_aware = type(model) is ProgramModelGraph
    block_idx = program._block_indices()
    seen = set()
    emitted_names = {}
    model_params = {}
    entries = []
    for v in _all_program_ops(program):
        if v.op not in _MODEL_PARAM_OPS:
            continue
        emit_model = model_for_node(model, v)
        impl = _model_impl(emit_model)
        cache_key = id(impl)
        cached = model_params.get(cache_key)
        if cached is None:
            if not getattr(impl, "has_runtime_params", lambda: False)():
                model_params[cache_key] = ({}, {})
                continue
            nodes = impl.assign_runtime_indices()
            by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
            by_identity = {
                getattr(getattr(node, "handle", None), "qualified_id", None): node
                for node in nodes
                if getattr(node, "handle", None) is not None
            }
            cached = by_name, by_identity
            model_params[cache_key] = cached
        by_name, by_identity = cached
        if not by_name:
            continue
        exprs = _op_model_exprs(impl, v)
        if not exprs or not _has_runtime_param(exprs):
            continue
        blk = _required_param_block_index(block_idx, v.block, v)
        for ref in _runtime_param_refs_in(exprs):
            name = ref.name
            owner, qualified_id = _qualified_param_identity(
                ref, v.block, graph_aware=graph_aware)
            indexed_node = by_name.get(name)
            if indexed_node is None:
                raise ValueError(
                    "runtime parameter %s for block %r/model owner %s is absent from that "
                    "model's assigned RuntimeParam table"
                    % (qualified_id, v.block.local_id, owner))
            declaration = by_identity.get(qualified_id)
            if graph_aware and (
                    declaration is None or declaration.name != indexed_node[1].name):
                raise ValueError(
                    "runtime parameter %s does not match block %r/model owner %s declaration"
                    % (qualified_id, v.block.local_id, owner))
            route = (blk, owner, qualified_id)
            collision_key = (blk, name)
            prior = emitted_names.get(collision_key)
            if prior is not None and prior != route:
                raise ValueError(
                    "runtime parameter name collision for block %r: %r maps to both %s and %s"
                    % (v.block.local_id, name, prior[2], qualified_id))
            emitted_names[collision_key] = route
            if route in seen:
                continue
            index, node = indexed_node
            seen.add(route)
            # The metadata ABI is explicitly double-valued; this is the target-precision
            # lowering boundary, not a mutation/coercion of the authoring literal.
            entries.append((blk, name, index, float(node.value)))
        # RuntimeParams is indexed by the model's complete stable declaration table.  If the only
        # read is at index N, indices 0..N-1 still have to be materialised from BindSchema; compacting
        # the vector here would silently redirect the generated ``params.get(N)`` read.
        for index, node in enumerate(nodes):
            handle = getattr(node, "handle", None)
            if graph_aware and handle is None:
                raise ValueError(
                    "runtime parameter %r in block %r has no owner-qualified ParamHandle"
                    % (node.name, v.block.local_id))
            qualified_id = getattr(handle, "qualified_id", None) or node.name
            route = (blk, v.block.model_owner_path.canonical(), qualified_id)
            collision_key = (blk, node.name)
            prior = emitted_names.get(collision_key)
            if prior is not None and prior != route:
                raise ValueError(
                    "runtime parameter name collision for block %r: %r maps to both %s and %s"
                    % (v.block.local_id, node.name, prior[2], qualified_id))
            emitted_names[collision_key] = route
            if route in seen:
                continue
            seen.add(route)
            entries.append((blk, node.name, index, float(node.value)))
    entries.sort(key=lambda e: (e[0], e[2]))
    return entries


def _all_program_ops(program: Any) -> Any:
    """Iterate every op of the Program, descending into control-flow + apply + residual sub-blocks (the
    same flat walk the lowerability guards use; the sub-block ops are not in program._values)."""
    for v in program._values:
        yield v
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            blk = v.attrs.get(key)
            if isinstance(blk, (list, tuple)):
                yield from blk


def emit_program_params(program: Any, model: Any = None) -> str:
    """C++ source of the RUNTIME-parameter metadata the .so exports (ADC-510, Spec 5 C5): per flat
    parameter i, its PROGRAM block index, its stable WITHIN-block index (sorted-name order, the index
    the lowered runtime read uses), its NAME and its declaration DEFAULT. install_program reads it to
    SEED each block's RuntimeParams to the defaults; Python's _install_program_params reads the same
    routing (via the carried Program + model) to map the bound values to set_program_params, rejecting an
    unknown name. A Program reading no runtime parameter emits count 0 (no seed, the kernels read no
    param). NOT called from any hot kernel."""
    entries = program_param_entries(program, model)
    blocks = ", ".join(str(b) for b, _, _, _ in entries)
    indices = ", ".join(str(i) for _, _, i, _ in entries)
    # Declaration defaults are bind-plan data, not generated-code identity.  The compiled carrier is
    # seeded neutrally and the immutable BindSchema installs either the supplied value or its explicit
    # declaration default before a kernel can run.
    defaults = ", ".join("0.0" for _ in entries)
    name_cases = "".join('    case %d: return %s;\n' % (k, json.dumps(nm))
                         for k, (_, nm, _, _) in enumerate(entries))

    def ival(accessor: Any, csv: Any) -> str:
        return ('extern "C" int pops_program_param_%s(int i) {\n'
                '  static const int v[] = {%s};\n'
                '  return (i >= 0 && i < %d) ? v[i] : -1;\n}\n'
                % (accessor, csv if entries else "0", len(entries)))

    return (
        "// RUNTIME-parameter metadata (ADC-510, Spec 5 C5): per flat parameter, its PROGRAM block\n"
        "// index, its stable within-block index (sorted-name order, the index the lowered runtime read\n"
        "// uses) and its name. Carrier values are neutral here; BindSchema installs defaults/values.\n"
        "// NOT called from any hot kernel.\n"
        'extern "C" int pops_program_param_count() { return %d; }\n' % len(entries) +
        ival("block", blocks) +
        ival("index", indices) +
        'extern "C" const char* pops_program_param_name(int i) {\n'
        '  switch (i) {\n%s    default: return "";\n  }\n}\n' % name_cases +
        'extern "C" double pops_program_param_default(int i) {\n'
        '  static const double v[] = {%s};\n'
        '  return (i >= 0 && i < %d) ? v[i] : 0.0;\n}\n'
        % (defaults if entries else "0.0", len(entries)))
