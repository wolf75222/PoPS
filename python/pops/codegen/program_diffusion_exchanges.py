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
        if value.op != "linear_combine":
            return [(value, coefficient)]
        result = []
        for child, factors in zip(value.inputs, value.attrs["coeffs"], strict=True):
            product = {}
            for power, numerator, denominator in _coeff_metadata_terms(factors):
                for outer, amount in coefficient.items():
                    product[power+outer] = product.get(power+outer, Fraction()) + amount*Fraction(numerator,denominator)
            result.extend(atoms(child, product))
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


def accepted_diffusive_quadrature(program):
    """Stop at each fresh residual: a predictor's ancestry is not an accepted exchange."""
    selected = {}

    def hides_diffusion(value):
        if value.op == "diffusive_rhs":
            return True
        # Fresh residual evaluation consumes a predictor; its state ancestry is not another
        # contribution to the accepted affine quadrature.
        if value.op in {"rhs", "source", "coupled_rate"}:
            return False
        return any(hides_diffusion(child) for child in value.inputs)

    def walk(value, weight):
        if value.op == "diffusive_rhs":
            entry = selected.setdefault(value.id, (value, {}))[1]
            for power, coefficient in weight.items():
                entry[power] = entry.get(power, Fraction()) + coefficient
            return
        if value.op != "linear_combine":
            if hides_diffusion(value):
                raise ValueError("accepted state transform hides diffusive exchanges outside an authenticated affine quadrature")
            return
        for child, coefficients in zip(value.inputs, value.attrs["coeffs"], strict=True):
            product = {}
            for power, numerator, denominator in _coeff_metadata_terms(coefficients):
                for outer, coefficient in weight.items():
                    product[power+outer] = product.get(power+outer, Fraction()) + coefficient*Fraction(numerator,denominator)
            walk(child, product)

    for value in program._commits.values():
        walk(value, {0: Fraction(1)})
    rows = tuple((value, {power: coefficient for power,coefficient in weight.items() if coefficient})
                 for value,weight in selected.values())
    if rows:
        from pops.time import certify_program_graph
        certificate = certify_program_graph(program.to_graph())
        if (certificate.properties.ssp is None or certificate.properties.ssp.coefficient != 1) \
                and not _single_forward_euler_with_frozen_inputs(program, rows):
            raise ValueError("explicit diffusive exchange currently requires a certified SSP coefficient-one affine step")
        if any(set(weight) != {1} or weight[1] < 0 for _,weight in rows):
            raise ValueError("diffusive accepted quadrature must retain nonnegative exact dt weights")
    return rows


def emit_accepted_diffusive_exchanges(program, *, target="system"):
    rows = accepted_diffusive_quadrature(program)
    if rows and target != "system":
        raise ValueError("AMR diffusive accepted exchanges require the M7 composite face route")
    lines = []
    for value, weight in rows:
        _emit_diffusive_accepted(value, "diffusion_prepared_%d" % value.id, lines,
            _coeff_cpp(weight), "stage:"+str(value.point)+"/evaluation:"+str(value.id))
    return lines
