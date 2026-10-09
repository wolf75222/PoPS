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
    # Optional planning must also accept legacy groups with no numerical plan. Only
    # a selected resolved path needs the complete emitter authority checked below.
    facade = getattr(model, "_dsl", model)
    impl = getattr(facade, "_m", facade)
    plan = getattr(model, "_resolved_operations", getattr(impl, "_resolved_operations", None))
    if plan is None:
        return False
    _model_impl(model)
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
                              and proof.get("provider") == "native_incident_face_stability")
        diffusion |= method.get("method") in {"diffusion", "tensor_diffusion"}
    if transport and diffusion and not authenticated:
        raise ValueError("composed transport stability requires an authenticated resolved face envelope")
    return transport and diffusion


def _unguarded(value):
    # A successful acceptance guard returns the same value. Its condition is
    # neither a new field nor an explicit update. Project-and-recheck branches
    # are deliberately not aliases: they may return a different field.
    while value.op == "acceptance_guard":
        value = value.inputs[0]
    return value


def _rate_state(value):
    return _unguarded(value.inputs[value.attrs.get("target_input", 0)])


def _affine_terms(value, stops=()):
    result = {}

    def walk(node, weight):
        node = _unguarded(node)
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
    return value.op in {"diffusive_rhs", "principal_rate"} or (
        value.op == "rhs" and value.attrs.get("flux", True))


def explicit_update_consumers(program):
    """Identify state combinations that can advance accepted explicit state.

    An RHS is an observation until a state combination consumes it. Follow the
    accepted result and the stage inputs of rates that feed it, but do not
    interpret a nonlinear solver's residual or predictor as an explicit step.
    """
    updates, seen = set(), set()
    observed_rates = {}

    def only_rate_observations(node):
        node = _unguarded(node)
        if node.id not in observed_rates:
            # Issued SSA is acyclic. A repeated unfinished node cannot certify a rate sum.
            observed_rates[node.id] = False
            observed_rates[node.id] = (_is_rate(node) or (
                node.op == "linear_combine" and bool(node.inputs)
                and all(only_rate_observations(child) for child in node.inputs)))
        return observed_rates[node.id]

    def walk(node, *, state_consumer=True):
        # The same scratch can first contribute to a state sum and later be sampled
        # as a State. Its actual State consumption must still receive a budget.
        role = (node.id, state_consumer)
        if role in seen:
            return
        seen.add(role)
        if node.op == "linear_combine":
            if node.vtype == "state" and (
                    state_consumer or not only_rate_observations(node)):
                updates.add(node.id)
            for child in node.inputs:
                walk(child, state_consumer=False)
        elif node.op == "principal_rate":
            # Coupled groups may sample a predictor from another block while
            # advancing the target block; each sampled explicit stage matters.
            for sampled in node.inputs:
                walk(sampled)
        elif _is_rate(node):
            index = node.attrs.get("target_input", 0)
            if type(index) is int and 0 <= index < len(node.inputs):
                walk(node.inputs[index])
        elif node.op in {"while", "range"}:
            walk(node.inputs[0])
            body = node.attrs.get("body")
            if body is None or not hasattr(body, "op"):
                raise ValueError("explicit loop has no authenticated body result")
            walk(body)
        elif node.op == "branch":
            for key in ("true_result", "false_result"):
                child = node.attrs.get(key)
                if hasattr(child, "op"):
                    walk(child)
        elif node.op == "solve_spatial_nonlinear":
            from pops.time._program.spatial_solve import validate_spatial_commit
            validate_spatial_commit(program, node)
            walk(node.inputs[0])
        elif node.op in {"solve_local_nonlinear", "solve_coupled_implicit",
                         "solve_implicit_source"}:
            return
        elif node.op == "acceptance_guard":
            walk(node.inputs[0], state_consumer=state_consumer)
        elif node.vtype == "state" and len(node.inputs) == 1:
            walk(node.inputs[0], state_consumer=state_consumer)

    for value in program._commits.values():
        walk(value)
    return frozenset(updates)


def _retain_convex_stage_values(terms):
    """Recover a positive certificate from retained affine SSA stage identities.

    A Butcher-form update can spend a rate evaluated at a stage that is absent
    from its written final state sum. Substitute that *exact* already evaluated
    stage using rational coefficients. For example U0+dt/2*(R0+R1) admits
    U0/2+U1/2+dt/2*R1 when U1=U0+dt*R0. The emitted floating-point arithmetic is
    untouched. This is a sufficient certificate, not an SSP claim for every RK
    tableau; negative or non-affine identities still require authored evidence.
    """
    table = {node.id: (node, dict(weight)) for node, weight in terms}
    missing = {_rate_state(node).id: _rate_state(node)
               for node, _ in terms if _is_rate(node)
               and _rate_state(node).id not in table}
    for state in sorted(missing.values(), key=lambda node: node.id, reverse=True):
        if state.op != "linear_combine":
            continue
        beta = sum(weight.get(1, Fraction()) for node, weight in table.values()
                   if _is_rate(node) and _rate_state(node).id == state.id)
        if beta <= 0:
            continue
        expansion = _affine_terms(state)
        if (not expansion or
                any(set(weight) != ({1} if _is_rate(node) else {0}) or
                    next(iter(weight.values())) <= 0 or
                    (not _is_rate(node) and node.vtype != "state")
                    for node, weight in expansion)):
            continue
        if sum(weight[0] for node, weight in expansion if not _is_rate(node)) != 1:
            continue
        capacity = min(table.get(node.id, (None, {}))[1].get(power, Fraction()) / amount
                       for node, weight in expansion for power, amount in weight.items())
        alpha = min(beta, capacity)
        if alpha <= 0:
            continue
        for node, weight in expansion:
            existing = table[node.id][1]
            for power, amount in weight.items():
                existing[power] -= alpha * amount
        table[state.id] = (state, {0: alpha})
    return tuple((node, {power: amount for power, amount in weight.items() if amount})
                 for node, weight in table.values() if any(weight.values()))


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
    bases = {_rate_state(node).id: _rate_state(node) for node in rates}
    terms = _affine_terms(value, bases)
    terms = _retain_convex_stage_values(terms)
    rates = [(node, weight) for node, weight in terms if _is_rate(node)]
    states = [(node, weight) for node, weight in terms if not _is_rate(node)]
    if not states:
        if value.vtype == "state":
            raise ValueError(_DIAGNOSTIC + "; an explicit state consumer has no state weight")
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
        state = _rate_state(rate)
        if budgets.get(state.id, 0) <= 0 or rate.block != state.block:
            raise ValueError(_DIAGNOSTIC)
        if rate.attrs.get("schedule") is not None:
            raise ValueError(_DIAGNOSTIC + "; scheduled rate frequency needs its own scoped carrier")
        entry = groups.setdefault(state.id, (state, budgets[state.id], []))
        if entry[2] and entry[2][0][0].point != rate.point:
            raise ValueError(_DIAGNOSTIC + "; distinct evaluation points cannot share one budget")
        entry[2].append((rate, weight[1]))
    return tuple((state, alpha, tuple(rows)) for state, alpha, rows in groups.values())


def emit_transport_frequency(value, var, lines, *, model, block_index, target="system"):
    """Capture the selected envelope while this exact rate's inputs/providers are current."""
    if not has_independent_diffusion_transport(model):
        return
    from pops.codegen.program_emit_kernels import _named_fluxes

    if _named_fluxes(value) is not None:
        raise ValueError("composed transport stability requires the selected native finite-volume envelope")
    frequency = "transport_frequency_%d" % value.id
    trace = ("," + (var.get(("rhs_input_trace", value.id)) or "nullptr")
             if target == "amr_system" else "")
    lines.append("const pops::Real %s = ctx.evaluated_transport_frequency(%d,%s,%d%s);"
                 % (frequency, block_index, var[value.inputs[0].id], value.id, trace))
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


def emit_user_face_stability(value, var, lines, *, block_index):
    """Check explicit consumers of numerical faces with one shared convex budget."""
    def selected(node):
        return any((kind, node.id) in var for kind in (
            "user_face_frequency", "principal_frequency", "path_frequency"))

    if not any(selected(node)
               for node, _ in _affine_terms(value)):
        return
    groups = partition_stability_groups(value, include_transport=True)
    for index, (_, alpha, rates) in enumerate(groups):
        if not any(selected(rate) for rate, _ in rates):
            continue
        terms = []
        for rate, beta in rates:
            token = var.get(("partition_frequency", rate.id))
            if token is None:
                raise ValueError(_DIAGNOSTIC + "; numerical face frequency is unavailable")
            terms.append("%s * %s" % (scalar_cpp(beta), token))
        frequency = "user_face_update_frequency_%d_%d" % (value.id, index)
        lines.append("const pops::Real %s = %s;" % (frequency, " + ".join(terms)))
        lines.append(
            "ctx.consume_pointwise_evaluation_status(%d,%d,"
            "(!std::isfinite(%s) || %s < 0 || !std::isfinite(dt) || dt < 0 || "
            "dt * %s > %s * static_cast<pops::Real>(ctx.numerical_face_courant()) "
            "* (1 + 32 * std::numeric_limits<pops::Real>::epsilon())) ? 1 : 0,"
            "\"user_face_numerical_stability\",503);"
            % (block_index, value.id, frequency, frequency, frequency, scalar_cpp(alpha)))


def require_deferred_partition_bounds(var):
    if not var.get(("partition_stability_deferred",), frozenset()) <= var.get(
            ("partition_stability_checked",), frozenset()):
        raise ValueError(_DIAGNOSTIC + "; an explicit diffusion rate lacks its consuming bound")
