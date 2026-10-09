"""Accepted transport observations from exact resolved rates and conservative commits."""
from fractions import Fraction

from pops.identity import make_identity

from pops.codegen.program_emit_kernels import _coeff_cpp, _coeff_metadata_terms, _model_impl
from pops.codegen.program_partition_stability import has_independent_diffusion_transport


def retained_transport_selection(value, model):
    if value.op != "rhs" or not value.attrs.get("flux", True):
        return None
    # A standalone FV rate retains its actual face carrier only when a persistent integral
    # explicitly consumes that evaluation. Existing composition ledgers keep their own authority.
    requested = any(row[1] == value.id
                    for row in getattr(value.prog, "_integral_transfers", ()))
    if not requested and not has_independent_diffusion_transport(model):
        return None
    impl = _model_impl(model)
    plan = getattr(model, "_resolved_operations", getattr(impl, "_resolved_operations", None))
    if plan is None:
        return None
    matches = [operation for operation in plan.operations
               if operation.guarantees.get("program_evaluation", {}).get("node_id") == value.id
               and operation.guarantees.get("program_evaluation", {}).get("operation") == "rhs"
               and operation.guarantees.get("numerical_method", {}).get("method") == "finite_volume"]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("accepted transport requires one exact resolved numerical evaluation")
    operation = matches[0]
    evaluation = next(row for row in plan.evaluations if row.identity == operation.evaluation)
    terms = [row for row in evaluation.occurrences if row.identity in operation.consumes
             and row.kind == "flux"]
    if len(terms) != 1 or terms[0].coefficient != -1:
        raise ValueError("accepted transport requires one exact negative-divergence occurrence")
    block = value.block
    if not block.is_resolved:
        block = block._resolved(block.owner_path.canonical())
    identity = make_identity("transport-operation", {
        "source": operation.guarantees["declaration_operation"],
        "block": block.canonical_identity(),
    }).token
    return identity, identity+"/occurrence:"+terms[0].identity


def declare_transport_faces(value, model, var, lines):
    selection = retained_transport_selection(value, model)
    if selection is None:
        return None
    name = "transport_faces_%d" % value.id
    lines.append("std::vector<pops::nd::FaceField<pops::kNativeDimension>> %s;" % name)
    var[("accepted_transport", value.id)] = (value, name, *selection)
    return name


def accepted_transport_regions(program, captured):
    """Version2 conservative affine quadratures, owned by their exact executable region.

    A child tick carries its incoming State with coefficient one. Its new face amounts inherit
    the global commit coefficient; the incoming carrier crosses the loop boundary only once.
    Nonunit carry, opaque transforms and scheduled bodies require a separate authenticated rule.
    """
    from pops.time._schedule.synchronization import SampleAndHold, relation_data
    if not captured:
        return {None: {}}
    owners, loops, body_cache = {}, {}, {}

    def index(values, owner):
        for value in values:
            owners[value.id] = owner
            if value.op == "subcycle":
                loops[value.id] = value
                index(value.attrs["body_block"], value.id)
    index(getattr(program, "_values", ()), None)

    def spatial_solve(value):
        node = value
        while len(node.inputs) == 1:
            if getattr(node, "attrs", {}).get("schedule") is not None:
                return None
            if node.op in {"solve_outcome", "solve_outcome_component", "local_transform"}:
                node = node.inputs[0]
            elif node.op == "linear_combine" and dict(node.attrs["coeffs"][0]) == {0: 1}:
                node = node.inputs[0]
            else:
                break
        return node if node.op == "solve_spatial_nonlinear" else None

    def multiply(left, right):
        result = {}
        for a, x in left.items():
            for b, y in right.items():
                result[a+b] = result.get(a+b, Fraction()) + x*y
        return {power: value for power, value in result.items() if value}

    def merge(destination, source, factor={0: Fraction(1)}):
        for key, weight in source.items():
            item = destination.setdefault(key, {})
            for power, value in multiply(weight, factor).items():
                item[power] = item.get(power, Fraction()) + value

    def hides(value):
        if value.id in captured:
            return True
        if value.op in {"rhs", "diffusive_rhs", "source", "coupled_rate"}:
            return False
        attrs = getattr(value, "attrs", {})
        children = list(value.inputs)
        for key in ("true_result", "false_result", "body", "result"):
            child = attrs.get(key)
            if hasattr(child, "op"):
                children.append(child)
        for key in ("true_block", "false_block", "body_block", "cond_block", "apply_block", "residual_block"):
            children.extend(attrs.get(key) or ())
        return any(hides(child) for child in children)

    def body(loop):
        if loop.id not in body_cache:
            data = decompose(loop.attrs["body"], loop.id, loop.inputs[0].id)
            carry = data.pop(("carry", loop.inputs[0].id), {})
            if carry != {0: Fraction(1)}:
                raise ValueError("child transport requires an authenticated unit affine carry boundary")
            body_cache[loop.id] = data
        return body_cache[loop.id]

    def decompose(value, owner, boundary=None):
        if value.id == boundary:
            return {("carry", boundary): {0: Fraction(1)}}
        if getattr(value, "attrs", {}).get("schedule") is not None and hides(value):
            raise ValueError("scheduled accepted transport requires its own scoped quadrature carrier")
        if value.id in captured:
            if owners.get(value.id) != owner:
                raise ValueError("transport capture crosses a region without its authenticated carry boundary")
            return {("rate", value.id): {0: Fraction(1)}}
        if value.op in {"rhs", "diffusive_rhs", "source", "coupled_rate"}:
            return {}
        solve = spatial_solve(value)
        if solve is not None:
            from pops.time._program.spatial_solve import validate_spatial_commit
            validate_spatial_commit(program, solve)
            return decompose(solve.inputs[0], owner, boundary)
        if value.op == "acceptance_guard":
            return decompose(value.inputs[0], owner, boundary)
        if value.op == "synchronize" and value.attrs.get("relation") == relation_data(SampleAndHold()):
            return decompose(value.inputs[0], owner, boundary)
        if value.op == "subcycle":
            if not hides(value):
                return {}
            body(value)  # Validate before recording any regional emission.
            data = decompose(value.inputs[0], owner, boundary)
            data[("loop", value.id)] = {0: Fraction(1)}
            return data
        if value.op != "linear_combine":
            if hides(value):
                raise ValueError("accepted transform hides transport outside authenticated affine quadrature")
            return {}
        result = {}
        for child, factors in zip(value.inputs, value.attrs["coeffs"], strict=True):
            factor = {power: Fraction(numerator, denominator)
                      for power, numerator, denominator in _coeff_metadata_terms(factors)}
            merge(result, decompose(child, owner, boundary), factor)
        return result

    result = {}
    def expand(data, owner):
        rates = result.setdefault(owner, {})
        for (kind, identifier), weight in data.items():
            weight = {power: value for power, value in weight.items() if value}
            if not weight:
                continue
            if kind == "loop":
                regional = {}
                merge(regional, body(loops[identifier]), weight)
                expand(regional, identifier)
            elif kind == "rate":
                merge(rates, {identifier: weight})
            else:
                raise ValueError("unresolved transport carry boundary")
    roots = {}
    for value in program._commits.values():
        merge(roots, decompose(value, None))
    expand(roots, None)
    for rates in result.values():
        for weight in rates.values():
            if set(weight) != {1} or weight[1] < 0:
                raise ValueError("transport accepted quadrature requires nonnegative exact dt weights in its local execution region")
    return result


def accepted_transport_quadrature(program, captured):
    return tuple(accepted_transport_regions(program, captured).get(None, {}).items())


def emit_accepted_transport_exchanges(program, var, block_indices, model, *, region=None, target="system"):
    from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges

    captured = {key[1]: item for key, item in var.items()
                if isinstance(key, tuple) and len(key) == 2 and key[0] == "accepted_transport"}
    from pops.codegen._resolved_block_operations import _program_values
    from pops.codegen.program_models import model_for_node
    selected = {value.id: value for value, _, _ in _program_values(program)
                if value.op == "rhs" and retained_transport_selection(
                    value, model_for_node(model, value)) is not None}
    if not selected:
        return []
    regions = accepted_transport_regions(program, selected)
    if target != "system" and any(owner is not None and rates for owner, rates in regions.items()):
        raise ValueError("regional accepted transport requires its authenticated Uniform realization")
    var[("accepted_transport_regions",)] = regions
    lines = []
    for identifier, weight in regions.get(region, {}).items():
        if identifier not in captured:
            raise ValueError("accepted transport face carrier is unavailable in this scope")
        value, faces, operation, occurrence = captured[identifier]
        if target == "system":
            from pops.time._evaluation_point import evaluation_stage_fraction
            stage = evaluation_stage_fraction(value)
            lines.extend(["{", "auto accepted_trace_frame_%d = ctx.logical_evaluation_scope(0, 1);" % identifier])
            lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
        lines.extend(emit_transport_exchanges(
            faces, operation, occurrence, "stage:"+str(value.point)+"/evaluation:"+str(value.id),
            _coeff_cpp(weight), program_block=block_indices[value.block],
            active_field=var[value.inputs[0].id]))
        if target == "system":
            lines.append("}")
    return lines
