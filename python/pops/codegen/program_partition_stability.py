"""Local convex stability budgets for independently evaluated spatial partitions."""
from fractions import Fraction

from pops.codegen.program_emit_kernels import _coeff_metadata_terms, _model_impl
from pops.identity.scalar import scalar_cpp


_DIAGNOSTIC = (
    "composed transport/diffusion update needs an explicit convex affine stability certificate; "
    "retain nonnegative state weights summing to one and first-power dt rate weights at "
    "their exact input stage"
)


def has_independent_diffusion_transport(model):
    impl = _model_impl(model)
    plan = getattr(model, "_resolved_operations", getattr(impl, "_resolved_operations", None))
    if plan is None:
        return False
    transport = False
    diffusion = False
    authenticated = True
    for row in plan.operations:
        method = row.guarantees.get("numerical_method")
        if not method:
            continue
        if method.get("method") == "finite_volume":
            transport = True
            proof = row.guarantees.get("explicit_transport_frequency")
            authenticated &= (proof is not None
                              and proof.get("provider") == "native_endpoint_model_wave_envelope")
        diffusion |= method.get("method") in {"diffusion", "tensor_diffusion"}
    if transport and diffusion and not authenticated:
        raise ValueError("composed transport stability requires an authenticated resolved face envelope")
    return transport and diffusion


def _affine_terms(value, stops=()):
    result = {}

    def walk(node, weight):
        if node.op != "linear_combine" or node.id in stops:
            entry = result.setdefault(node.id, (node, {}))[1]
            for power, amount in weight.items():
                entry[power] = entry.get(power, Fraction()) + amount
            return
        for child, factors in zip(node.inputs, node.attrs["coeffs"], strict=True):
            product = {}
            for power, numerator, denominator in _coeff_metadata_terms(factors):
                for outer, amount in weight.items():
                    product[power + outer] = (product.get(power + outer, Fraction())
                                              + amount * Fraction(numerator, denominator))
            walk(child, product)

    walk(value, {0: Fraction(1)})
    return tuple((node, {power: amount for power, amount in weight.items() if amount})
                 for node, weight in result.values() if any(weight.values()))


def _is_rate(value):
    return value.op == "diffusive_rhs" or (
        value.op == "rhs" and value.attrs.get("flux", True))


def partition_stability_groups(value, *, include_transport=False):
    """Return (input state, convex weight, ((rate, dt weight), ...)) groups.

    Stop at each exact rate input, so an earlier stage is a value, never extra current-stage
    quadrature. This is a sufficient local Shu--Osher decomposition, not a method/order claim.
    An unconsumed sum of rates has no state budget; its consuming state update supplies it.
    """
    if value.op != "linear_combine":
        return ()
    leaves = _affine_terms(value)
    rates = [node for node, _ in leaves if _is_rate(node)]
    if not rates:
        return ()
    if not include_transport and not (any(node.op == "rhs" for node in rates)
                                      and any(node.op == "diffusive_rhs" for node in rates)):
        return ()
    bases = {node.inputs[0].id: node.inputs[0] for node in rates}
    terms = _affine_terms(value, bases)
    rates = [(node, weight) for node, weight in terms if _is_rate(node)]
    states = [(node, weight) for node, weight in terms if not _is_rate(node)]
    if not states:
        return ()
    if any(node.vtype != "state" or set(weight) != {0} or weight[0] < 0
           for node, weight in states):
        raise ValueError(_DIAGNOSTIC)
    if sum(weight[0] for _, weight in states) != 1:
        raise ValueError(_DIAGNOSTIC)
    if any(set(weight) != {1} or weight[1] < 0 for _, weight in rates):
        raise ValueError(_DIAGNOSTIC)
    budgets = {node.id: weight[0] for node, weight in states}
    groups = {}
    for rate, weight in rates:
        state = rate.inputs[0]
        if budgets.get(state.id, 0) <= 0 or rate.block != state.block:
            raise ValueError(_DIAGNOSTIC)
        if rate.attrs.get("schedule") is not None:
            raise ValueError(_DIAGNOSTIC + "; scheduled rate frequency needs its own scoped carrier")
        entry = groups.setdefault(state.id, (state, budgets[state.id], []))
        if entry[2] and entry[2][0][0].point != rate.point:
            raise ValueError(_DIAGNOSTIC + "; distinct evaluation points cannot share one budget")
        entry[2].append((rate, weight[1]))
    return tuple((state, alpha, tuple(rows)) for state, alpha, rows in groups.values())


def emit_transport_frequency(value, var, lines, *, model, block_index):
    """Capture the selected envelope while this exact rate's inputs/providers are current."""
    if not has_independent_diffusion_transport(model):
        return
    from pops.codegen.program_emit_kernels import _named_fluxes

    if _named_fluxes(value) is not None:
        raise ValueError("composed transport stability requires the selected native finite-volume envelope")
    dimension = len(_model_impl(model)._flux)
    inverse_spacing = " + ".join("1/ctx.geometry().spacing(%d)" % axis
                                 for axis in range(dimension))
    frequency = "transport_frequency_%d" % value.id
    lines.append("const pops::Real %s = ctx.max_wave_speed(%d,%s) * (%s);"
                 % (frequency, block_index, var[value.inputs[0].id], inverse_spacing))
    var[("partition_frequency", value.id)] = frequency


def emit_partition_stability(value, var, lines, *, block_index, include_transport=False):
    groups = partition_stability_groups(value, include_transport=include_transport)
    for index, (_, alpha, rates) in enumerate(groups):
        terms = []
        for rate, beta in rates:
            token = var.get(("partition_frequency", rate.id))
            if token is None:
                raise ValueError(_DIAGNOSTIC + "; rate frequency is unavailable in this region")
            terms.append("%s * %s" % (scalar_cpp(beta), token))
        frequency = "partition_frequency_%d_%d" % (value.id, index)
        lines.append("const pops::Real %s = %s;" % (frequency, " + ".join(terms)))
        lines.append(
            "if (!(std::isfinite(dt) && dt >= 0 && std::isfinite(%s) && %s >= 0 && "
            "dt * %s <= %s * (1 + 32 * std::numeric_limits<pops::Real>::epsilon())))"
            % (frequency, frequency, frequency, scalar_cpp(alpha)))
        lines.append(
            '  ctx.consume_pointwise_evaluation_status(%d,%d,2,"combined_transport_diffusion_stability",502);'
            % (block_index, value.id))
        # Immutable evidence stays local when control-flow emitters copy the variable map.
        key = ("partition_stability_checked",)
        var[key] = var.get(key, frozenset()) | frozenset(
            rate.id for rate, _ in rates if rate.op == "diffusive_rhs")


def require_deferred_partition_bounds(var):
    if not var.get(("partition_stability_deferred",), frozenset()) <= var.get(
            ("partition_stability_checked",), frozenset()):
        raise ValueError(_DIAGNOSTIC + "; an explicit diffusion rate lacks its consuming bound")
