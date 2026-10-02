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
