"""Exact accepted affine quadrature for constitutive diffusive faces."""
from fractions import Fraction
from pops.codegen.program_emit_kernels import _coeff_cpp, _coeff_metadata_terms
from pops.codegen.program_emit_diffusion import _emit_diffusive_accepted


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
        if certificate.properties.ssp is None or certificate.properties.ssp.coefficient != 1:
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
