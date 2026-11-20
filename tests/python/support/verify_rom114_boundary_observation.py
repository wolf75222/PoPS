"""Check the native PI observation against an independent high-precision oracle."""
from decimal import Decimal, localcontext
from math import ulp
from pathlib import Path


def observations(filename, label):
    return [float.fromhex(row.split()[-1]) for row in Path(filename).read_text().splitlines()
            if row.startswith(label)]


stock = observations("field-boundary.log", "ROM114_OBSERVED")
pi = observations("field-pi-boundary.log", "ROM114_OBSERVED")
assert stock == pi
assert len(pi) == 4 and pi[0] == pi[1]
assert .14 < pi[0] < .16 and .07 < pi[2] < .08
dt = 1 / 128
for label, previous in (("pi_prior_a", .025), ("pi_prior_b", .1)):
    computed = observations("field-pi-boundary.log", f"ROM114_PROPOSAL {label}")
    assert len(computed) == len(pi) and computed[0] == computed[1]
    with localcontext() as context:
        context.prec = 80
        real = Decimal.from_float
        h, tol = real(dt), real(.1)
        # Both factors are strictly positive in this observed trial.
        ideal = h * real(.9) * (tol / real(pi[0])).sqrt() * (real(previous) / tol).sqrt().sqrt()
        rounded = float(ideal)
        assert abs(computed[0] - rounded) <= 3 * ulp(rounded)
    print(label, "dt", dt.hex(), "error", pi[0].hex(), "prior", previous.hex(),
          "generated", computed[0].hex(), "independent rounding", rounded.hex())
first, second = [observations("field-pi-boundary.log", "ROM114_PROPOSAL " + label)[0]
                 for label in ("pi_prior_a", "pi_prior_b")]
assert 0 < first < second < dt
print("All four guard norm values match stock bitwise. Two observations of the rejected trial match.")
for variant in ("euler", "field", "field-pi"):
    plain = Path(f"{variant}-boundary.stock.cpp").read_text().splitlines()
    observed = Path(f"{variant}-boundary.observed.cpp").read_text().splitlines()
    assert observed.pop(0) == "#include <cstdio>"
    assert [line for line in observed if "std::fprintf(stderr, \"ROM114_" not in line] == plain
print("All source snapshots differ only by observation calls within each authored graph.")
