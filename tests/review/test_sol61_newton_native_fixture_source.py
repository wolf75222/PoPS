"""Source authoring only: does not execute the Native fixture or mint evidence."""
from pathlib import Path
import sys
import pytest
import pops
from tests.python.integration.runtime.test_public_newton_typed_original import solver
from tests.python.integration.runtime import test_public_amr_original_field as amr
from tests.python.integration.runtime import test_nonlinear_mixed_field_runtime as uniform

@pytest.mark.parametrize("backend", ("uniform", "amr"))
@pytest.mark.parametrize("policy", ("relative", "floor", "absolute", "overflow"))
def test_real_public_authoring(backend, policy):
    root = Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
    assert "pops._pops" not in sys.modules
    selected = solver(backend, policy)
    if backend == "uniform":
        resolved, _, _ = uniform.prepared_case((0, 1), solver=selected)
    else:
        case, layout = amr.build(16, (2, 0, 1), solver=selected,
            physical_seed=2. if policy == "overflow" else None)
        resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved is not None
    assert "pops._pops" not in sys.modules

from tests.python.support.newton_typed_receipts import accepted_triplets, same_bits, uniform_original_norms
import numpy as np


def diagnostics(norm=1e-13, reference=.25):
    return {f"field_residual_{i}.{key}": value for i in (5, 13)
        for key, value in {"residual_norm": norm, "reference_residual_norm": reference,
                           "rel_residual": norm/(reference if reference > 0 else 1.)}.items()}


@pytest.mark.parametrize("policy", ("relative", "floor", "absolute"))
def test_triplet_receipt_admits_exact_typed_policy(policy):
    rows = accepted_triplets(diagnostics(), "uniform", solver("uniform", policy).convergence)
    assert len(rows) == 2
    assert all(row["cutoff"] == (2.5e-12 if policy == "relative" else 1e-11) for row in rows)


@pytest.mark.parametrize("attack", ("legacy-loose", "rawratio", "nan", "negative", "bool", "missing", "extra-solve", "overflow"))
def test_triplet_receipt_refuses_corruption(attack):
    data = diagnostics()
    policy = solver("uniform", "relative").convergence
    if attack == "legacy-loose":
        data = diagnostics(norm=5e-9)
    elif attack == "rawratio":
        data["field_residual_5.rel_residual"] = 0.
    elif attack == "nan":
        data["field_residual_5.reference_residual_norm"] = float("nan")
    elif attack == "negative":
        data["field_residual_5.residual_norm"] = -1.
    elif attack == "bool":
        data["field_residual_5.residual_norm"] = True
    elif attack == "missing":
        del data["field_residual_5.reference_residual_norm"]
    elif attack == "extra-solve":
        data.update({k.replace("_5.", "_29."): v for k,v in diagnostics().items() if "_5." in k})
    elif attack == "overflow":
        data = diagnostics(reference=2.)
        policy = solver("uniform", "overflow").convergence
    with pytest.raises(AssertionError):
        accepted_triplets(data, "uniform", policy)


def test_reference_zero_denominator_and_signedzero_bytes():
    absolute = solver("uniform", "absolute").convergence
    accepted_triplets(diagnostics(norm=0., reference=0.), "uniform", absolute)
    with pytest.raises(AssertionError):
        same_bits(np.array([0.]), np.array([-0.]))
    with pytest.raises(AssertionError):
        same_bits(np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float32))


def test_independent_full_original_norm_recompute():
    target = uniform.target_means(2); coefficient = uniform.parameter_means()
    loads = uniform.original_lhs(target, coefficient)
    references, norms = uniform_original_norms([target, target], loads, coefficient, uniform.original_lhs)
    assert references[0] == float(np.linalg.norm(loads)) and references[0] > 1
    assert references[1] > 0 and norms == [0., 0.]
