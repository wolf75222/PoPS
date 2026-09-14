"""Fail-closed native kernel for a common affine velocity-moment push-forward."""
from __future__ import annotations

from typing import Any

from pops.codegen.program_emit_kernels import (
    _cell_locals, _coeff_cpp, _has_runtime_param, _kernel_close, _kernel_open, _model_impl,
)
from pops.codegen.program_emit_model_kernels import _linear_source_rows, _provider_binding


def emit_affine_moment_kernel(
    model: Any, attrs: Any, old: str, mean: str, output: str, status: str,
    active: str, block: int, *, provider_plans: Any, consumer_qid: str,
) -> list[str]:
    impl = _model_impl(model)
    order = attrs["order"]
    count = (order + 1) * (order + 2) // 2
    if len(impl.cons_names) != count:
        raise ValueError("affine_moment_update model does not own the declared complete moment basis")
    rows = _linear_source_rows(impl, attrs["linear_operator"])
    x, y = 1, order + 1
    entries = (rows[x][x], rows[x][y], rows[y][x], rows[y][y])
    provider = _provider_binding(impl, entries, provider_plans, consumer_qid)
    impl.assign_runtime_indices()
    params = block if _has_runtime_param(entries) else None
    body = _kernel_open(output, old, params, provider_binding=provider, program_block=block)
    position = next(i for i, line in enumerate(body) if "pops::for_each_cell" in line)
    body[position:position] = [
        "  const auto meanA = std::as_const(%s).fab(li).view();" % mean,
        "  const auto statusA = %s.fab(li).view();" % status,
        "  const bool has_active = %s != nullptr;" % active,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> activeA = has_active"
        " ? std::as_const(*%s).fab(li).view()"
        " : pops::FieldView<const pops::Real, pops::kNativeDimension>{};" % active,
    ]
    body.append("    if (has_active && !(activeA(index, 0) >= pops::Real(0.5))) {")
    for component in range(count):
        body.append("      outA(index, %d) = %sA(index, %d);" % (component, old, component))
    body.extend(["      statusA(index, 0) = pops::Real(0);", "      return;", "    }"])
    body.extend("    " + line for line in _cell_locals(
        impl, entries, old, with_cons=True, with_prim=True, provider_binding=provider))
    for label, expression in zip(("jxx", "jxy", "jyx", "jyy"), entries, strict=True):
        body.append("    const pops::Real %s = %s;" % (label, expression.to_cpp()))
    body.append("    pops::Real old_moments[%d], endpoint[%d], mapped[%d];" % (count, count, count))
    for component in range(count):
        body.append("    old_moments[%d] = %sA(index, %d);" % (component, old, component))
        body.append("    endpoint[%d] = meanA(index, %d);" % (component, component))
    body.extend([
        "    const bool skew = std::isfinite(jxy) && jxx == pops::Real(0)"
        " && jyy == pops::Real(0) && jyx == -jxy;",
        "    const bool valid = skew && pops::moments::affine_velocity_push_forward<%d>("
        "old_moments, endpoint, jxy, %s, mapped);" % (order, _coeff_cpp(attrs["theta_dt"])),
        "    statusA(index, 0) = valid ? pops::Real(0) : pops::Real(1);",
    ])
    for component in range(count):
        body.append("    outA(index, %d) = valid ? mapped[%d] : old_moments[%d];"
                    % (component, component, component))
    body.extend(_kernel_close())
    return body
