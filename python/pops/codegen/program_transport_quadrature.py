"""Accepted transport observations from exact resolved rates and conservative commits."""
from fractions import Fraction

from pops.identity import make_identity

from pops.codegen.program_emit_kernels import _coeff_cpp, _coeff_metadata_terms, _model_impl
from pops.codegen.program_partition_stability import has_independent_diffusion_transport


def retained_transport_selection(value, model):
    if value.op != "rhs" or not value.attrs.get("flux", True) \
            or not has_independent_diffusion_transport(model):
        return None
    impl = _model_impl(model)
    plan = getattr(model, "_resolved_operations", getattr(impl, "_resolved_operations", None))
    matches = [operation for operation in plan.operations
               if operation.guarantees.get("program_evaluation", {}).get("node_id") == value.id
               and operation.guarantees.get("program_evaluation", {}).get("operation") == "rhs"]
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


def accepted_transport_quadrature(program, captured):
    """A fresh rate stops ancestry; a validated implicit stage carries its previous U only."""
    selected = {}

    def spatial_solve(value):
        node = value
        while len(node.inputs) == 1:
            if getattr(node, "attrs", {}).get("schedule") is not None:
                return None
            if node.op in {"solve_outcome", "solve_outcome_component", "local_transform"}:
                node = node.inputs[0]
            elif (node.op == "linear_combine"
                  and dict(node.attrs["coeffs"][0]) == {0: 1}):
                node = node.inputs[0]
            else:
                break
        return node if node.op == "solve_spatial_nonlinear" else None

    def hides_transport(value):
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
        return any(hides_transport(child) for child in children)

    def walk(value, weight):
        if getattr(value, "attrs", {}).get("schedule") is not None and hides_transport(value):
            raise ValueError("scheduled accepted transport requires its own scoped quadrature carrier")
        if value.id in captured:
            entry = selected.setdefault(value.id, {})
            for power, amount in weight.items():
                entry[power] = entry.get(power, Fraction()) + amount
            return
        if value.op in {"rhs", "diffusive_rhs", "source", "coupled_rate"}:
            return
        solve = spatial_solve(value)
        if solve is not None:
            from pops.time._program.spatial_solve import validate_spatial_commit
            validate_spatial_commit(program, solve)
            walk(solve.inputs[0], weight)
            return
        if value.op != "linear_combine":
            if hides_transport(value):
                raise ValueError("accepted transform hides transport outside authenticated affine quadrature")
            return
        for child, factors in zip(value.inputs, value.attrs["coeffs"], strict=True):
            product = {}
            for power, numerator, denominator in _coeff_metadata_terms(factors):
                for outer, amount in weight.items():
                    product[power+outer] = product.get(power+outer, Fraction()) + amount*Fraction(numerator, denominator)
            walk(child, product)

    for value in program._commits.values():
        walk(value, {0: Fraction(1)})
    result = []
    for identifier, weight in selected.items():
        weight = {power: amount for power, amount in weight.items() if amount}
        if not weight:
            continue
        if set(weight) != {1} or weight[1] < 0:
            raise ValueError("transport accepted quadrature requires nonnegative exact dt weights")
        result.append((identifier, weight))
    return tuple(result)


def emit_accepted_transport_exchanges(program, var, block_indices, model):
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
    lines = []
    for identifier, weight in accepted_transport_quadrature(program, selected):
        if identifier not in captured:
            raise ValueError("accepted transport face carrier is unavailable in this scope")
        value, faces, operation, occurrence = captured[identifier]
        lines.extend(emit_transport_exchanges(
            faces, operation, occurrence, "stage:"+str(value.point)+"/evaluation:"+str(value.id),
            _coeff_cpp(weight), program_block=block_indices[value.block],
            active_field=var[value.inputs[0].id]))
    return lines
