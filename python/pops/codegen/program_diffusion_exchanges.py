"""Exact accepted affine quadrature for constitutive diffusive faces."""
from fractions import Fraction
from pops.codegen.program_emit_kernels import _coeff_cpp, _coeff_metadata_terms
from pops.codegen.program_emit_diffusion import _emit_diffusive_accepted


def _single_forward_euler_with_frozen_inputs(program, rows):
    """Prove only one U+dt*R(U) commit, with every other state exactly unchanged.

    An independent field solve can prevent the global method recognizer from producing a
    tableau. This local accepted-update proof does not certify that broader Program as SSP.
    Solve reports and their failure guards remain ordinary preceding Program operations.
    """
    if len(rows) != 1 or rows[0][1] != {1: Fraction(1)}:
        return False
    rate = rows[0][0]
    if not rate.inputs:
        return False
    initial = rate.inputs[0]
    if initial.op != "state" or initial.block != rate.block or initial.point != rate.point:
        return False

    def atoms(value, coefficient):
        result = []
        pending = [(value, coefficient, False)]
        active = set()
        while pending:
            current, outer_weight, exiting = pending.pop()
            if exiting:
                active.remove(id(current))
                continue
            if id(current) in active:
                raise ValueError("accepted affine quadrature contains a cycle")
            if current.op != "linear_combine":
                result.append((current, outer_weight))
                continue
            active.add(id(current))
            pending.append((current, outer_weight, True))
            children = []
            for child, factors in zip(current.inputs, current.attrs["coeffs"], strict=True):
                product = {}
                for power, numerator, denominator in _coeff_metadata_terms(factors):
                    for outer, amount in outer_weight.items():
                        product[power+outer] = product.get(power+outer, Fraction()) + amount*Fraction(numerator,denominator)
                children.append((child, product, False))
            pending.extend(reversed(children))
        return result

    evolving = 0
    for state_ref, value in program._commits.items():
        terms = atoms(value, {0: Fraction(1)})
        if any(term.op not in {"state", "diffusive_rhs"} for term, _ in terms):
            return False
        state_terms = [(term, weight) for term, weight in terms if term.op == "state"]
        rate_terms = [(term, weight) for term, weight in terms if term.op == "diffusive_rhs"]
        if len(state_terms) != 1 or state_terms[0][1] != {0: Fraction(1)}:
            return False
        base = state_terms[0][0]
        if base.state_ref != state_ref or base.block != state_ref.block_ref:
            return False
        if rate_terms:
            if (len(rate_terms) != 1 or rate_terms[0][0] is not rate
                    or rate_terms[0][1] != {1: Fraction(1)} or base is not initial):
                return False
            evolving += 1
    return evolving == 1


def accepted_diffusive_quadrature(program, *, partition_stability_checked=(),
                                  authored_stability_rates=(), model_authority=None):
    """Stop at each fresh residual: a predictor's ancestry is not an accepted exchange."""
    selected = {}

    def hides_diffusion(value):
        # This predicate inspects a DAG, not the accepted affine multiplicities.
        # Keep the fresh-residual frontier while visiting each node at most once.
        pending = [value]
        seen = set()
        while pending:
            current = pending.pop()
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            if current.op == "diffusive_rhs":
                return True
            if current.op not in {"rhs", "source", "coupled_rate"}:
                pending.extend(reversed(current.inputs))
        return False

    def walk(value, weight):
        # Each affine path contributes, even when several paths share a node.
        # Reverse only the stack insertion to retain the authored ordinal order.
        pending = [(value, weight, False)]
        active = set()
        while pending:
            current, outer_weight, exiting = pending.pop()
            if exiting:
                active.remove(id(current))
                continue
            if id(current) in active:
                raise ValueError("accepted affine quadrature contains a cycle")
            if current.op == "acceptance_guard":
                from .program_accepted_ssp import successful_guard_input
                active.add(id(current))
                pending.append((current, outer_weight, True))
                pending.append((successful_guard_input(current), outer_weight, False))
                continue
            if current.op == "diffusive_rhs":
                entry = selected.setdefault(current.id, (current, {}))[1]
                for power, coefficient in outer_weight.items():
                    entry[power] = entry.get(power, Fraction()) + coefficient
                continue
            if current.op != "linear_combine":
                if hides_diffusion(current):
                    raise ValueError("accepted state transform hides diffusive exchanges outside an authenticated affine quadrature")
                continue
            active.add(id(current))
            pending.append((current, outer_weight, True))
            children = []
            for child, coefficients in zip(current.inputs, current.attrs["coeffs"], strict=True):
                product = {}
                for power, numerator, denominator in _coeff_metadata_terms(coefficients):
                    for outer, coefficient in outer_weight.items():
                        product[power+outer] = product.get(power+outer, Fraction()) + coefficient*Fraction(numerator,denominator)
                children.append((child, product, False))
            pending.extend(reversed(children))

    for value in program._commits.values():
        walk(value, {0: Fraction(1)})
    rows = tuple((value, {power: coefficient for power,coefficient in weight.items() if coefficient})
                 for value,weight in selected.values())
    if rows:
        from .program_accepted_ssp import prove_accepted_update_ssp
        certificate, reason = prove_accepted_update_ssp(program, model_authority=model_authority)
        if certificate is None and not all(rate.id in partition_stability_checked
                or rate.id in authored_stability_rates for rate, _ in rows):
            raise ValueError("explicit diffusive acceptance needs an SSP coefficient-one convex proof: " + reason)
        if any(set(weight) != {1} or (weight[1] < 0 and rate.id not in authored_stability_rates)
               for rate,weight in rows):
            raise ValueError("diffusive accepted quadrature must retain nonnegative exact dt weights")
    return rows


def emit_accepted_diffusive_exchanges(program, *, target="system", block_indices=None,
                                     partition_stability_checked=(), authored_stability_rates=(),
                                     model_authority=None):
    rows = accepted_diffusive_quadrature(
        program, partition_stability_checked=partition_stability_checked,
        authored_stability_rates=authored_stability_rates, model_authority=model_authority)
    if rows and target not in {"system", "amr_system"}:
        raise ValueError("diffusive accepted exchanges require a native install scope")
    lines = []
    for value, weight in rows:
        block = 0 if block_indices is None else block_indices[value.block]
        _emit_diffusive_accepted(value, "diffusion_prepared_%d" % value.id, lines,
            _coeff_cpp(weight), "stage:"+str(value.point)+"/evaluation:"+str(value.id), block)
    return lines
