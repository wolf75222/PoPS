"""pops.codegen.program_emit_params : compiled-Program RUNTIME-parameter routing + metadata (ADC-510).

Extracted from ``pops.codegen.program_codegen`` so each emitter module fits the Spec-4 line budget.
A compiled time Program whose physics reads a ``dsl.Param(..., kind="runtime")`` carries the value in
a per-PROGRAM-block ``pops::RuntimeParams`` owned by the System (not the .so closure), so the value can
be changed at run time WITHOUT recompiling (Spec 5 C5). This module computes the param ROUTING (which
program block reads which runtime parameter, at which stable index, with which default) and emits the
``pops_program_param_*`` metadata table the .so exports. The SAME ``_program_param_entries`` drives the
C++ install-time seed (System::install_program), the Python bind-time route
(System._install_program_params via CompiledProblem.runtime_param_routes) and this metadata, so all
three agree byte-for-byte. The per-cell read of the parameter (``params.get(index)`` bound from
``ctx.program_params(block)``) lives in ``program_emit_kernels`` / ``program_emit_model_kernels``.
"""
from __future__ import annotations

import json
from typing import Any

from pops.codegen.program_emit_kernels import _has_runtime_param, _model_impl


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


def _runtime_param_names_in(exprs: Any) -> set:
    """The set of runtime-parameter NAMES read anywhere in @p exprs (a RuntimeParamRef walk)."""
    from pops.ir.values import RuntimeParamRef
    from pops.ir.visitors import _children
    names = set()
    stack = list(exprs)
    while stack:
        e = stack.pop()
        if isinstance(e, RuntimeParamRef):
            names.add(e.name)
        else:
            stack.extend(_children(e))
    return names


def program_param_entries(program: Any, model: Any) -> list:
    """Per (PROGRAM block, runtime parameter) entries the .so exports + Python routes (ADC-510).

    Walk the Program ops (descending into control-flow / apply / residual sub-blocks); for each
    model-coefficient op that READS a runtime parameter, record one entry (program block index,
    parameter name, stable WITHIN-block index, declaration default). The within-block index is the
    model's STABLE runtime index (sorted-name order, assigned by assign_runtime_indices, matching the
    lowered ``params.get(idx)`` and the per-block RuntimeParams the System seeds / sets). Deduplicated by
    (block, name); sorted by (block, index) so the metadata table and Python routing agree on order.
    Empty when no model kernel reads a runtime param. @p model the physical model the Program lowers
    (None -> no entries)."""
    if model is None:
        return []
    impl = _model_impl(model)
    if not getattr(impl, "has_runtime_params", lambda: False)():
        return []
    nodes = impl.assign_runtime_indices()  # stable sorted order used by kernel emission
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    block_idx = program._block_indices()
    seen = set()
    entries = []
    for v in _all_program_ops(program):
        exprs = _op_model_exprs(impl, v)
        if not exprs or not _has_runtime_param(exprs):
            continue
        blk = _required_param_block_index(block_idx, v.block, v)
        for name in _runtime_param_names_in(exprs):
            indexed_node = by_name.get(name)
            if indexed_node is None or (blk, name) in seen:
                continue
            index, node = indexed_node
            seen.add((blk, name))
            # The metadata ABI is explicitly double-valued; this is the target-precision
            # lowering boundary, not a mutation/coercion of the authoring literal.
            entries.append((blk, name, index, float(node.value)))
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


def program_param_routes(program: Any, model: Any) -> tuple:
    """Routing of @p program's RUNTIME parameters to the per-PROGRAM-block ``set_program_params``
    vectors (ADC-510, Spec 5 C5). Returns ``(per_block, defaults)``: ``per_block`` maps a program block
    index to the COMPLETE list of param names in within-block index order (the order
    ``System.set_program_params`` expects), ``defaults`` maps a name to its declaration value. Built
    from the SAME ``program_param_entries`` the .so metadata uses, so the Python bind route and the C++
    seed/read agree byte-for-byte. Empty dicts when @p program / @p model is None or no runtime param is
    read."""
    if program is None or model is None:
        return {}, {}
    per_block = {}
    defaults = {}
    for blk, name, idx, default in program_param_entries(program, model):
        vec = per_block.setdefault(blk, [])
        if len(vec) <= idx:
            vec.extend([None] * (idx + 1 - len(vec)))
        vec[idx] = name
        defaults[name] = default
    return per_block, defaults


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
    defaults = ", ".join(repr(d) for _, _, _, d in entries)
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
        "// uses), its name and declaration default. install_program seeds the per-block RuntimeParams;\n"
        "// Python routes the bound values to set_program_params. NOT called from any hot kernel.\n"
        'extern "C" int pops_program_param_count() { return %d; }\n' % len(entries) +
        ival("block", blocks) +
        ival("index", indices) +
        'extern "C" const char* pops_program_param_name(int i) {\n'
        '  switch (i) {\n%s    default: return "";\n  }\n}\n' % name_cases +
        'extern "C" double pops_program_param_default(int i) {\n'
        '  static const double v[] = {%s};\n'
        '  return (i >= 0 && i < %d) ? v[i] : 0.0;\n}\n'
        % (defaults if entries else "0.0", len(entries)))
