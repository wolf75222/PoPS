"""Public Dim1 Gaussian/mixture qualification of nonzero far tails."""

import math
import os

import numpy as np
import pytest

import pops
from tests.python.integration.runtime.test_dsl_runtime_params import (
    _binary_fingerprints,
    _binary_paths,
)
from tests.python.integration.runtime.test_exact_analytic_initial import (
    _case,
    _evidence_metadata,
    _export_image,
    _image,
    _legacy_image,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def _tail_oracle(n, sign, center):
    """Independent CPython erfc bound difference, with no additive background."""
    lower, upper = (8.0, 8.1) if sign == 1 else (-8.1, -8.0)
    dx = (upper - lower) / n
    values = []
    for cell in range(n):
        lo, hi = lower + cell * dx - center, lower + (cell + 1) * dx - center
        # Reflect negative intervals before evaluating positive erfc arguments.
        if hi < 0:
            lo, hi = -hi, -lo
        values.append(math.sqrt(math.pi) / (2 * dx) * (math.erfc(lo) - math.erfc(hi)))
    return np.asarray(values)


@pytest.mark.parametrize("sign", [-1, 1], ids=["negative-tail", "positive-tail"])
@pytest.mark.parametrize("mixture", [False, True], ids=["gaussian", "mixture"])
def test_public_exact_initial_preserves_nonzero_far_tails(
    sign, mixture, isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    configured = os.environ.get("POPS_NATIVE_DIM")
    assert configured in {"1", "2", "3"}, "qualification requires explicit POPS_NATIVE_DIM"
    if configured != "1":
        pytest.skip("far-tail qualification is declared for native Dim=1")
    n = 8
    resolved, parameter = _case("system", n, 1, mixture, tail_sign=sign)
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    first, second = _tail_oracle(n, sign, 0.0), _tail_oracle(n, sign, 0.1 * sign)
    for weight in (0.25, 0.75) if mixture else (1.0, 0.5):
        context = artifact_execution_context(artifact)
        simulation = pops.bind(
            artifact, params={parameter: weight}, resources={"execution_context": context}
        )
        actual = np.asarray(simulation.get_state("renamed-density"), dtype=np.float64).reshape(n)
        expected = weight * first + ((1 - weight) * second if mixture else 0.0)
        metadata = _evidence_metadata(
            artifact,
            simulation,
            context,
            before,
            target="system",
            n=n,
            dim=1,
            mixture=mixture,
            weight=weight,
            tail_sign=sign,
        )
        _export_image(
            tmp_path,
            record_property,
            "initial",
            _image(simulation, "system", n, 1),
            {0: expected},
            metadata,
        )
        assert np.all(expected > 0) and np.all(actual > 0)
        # An erf(hi)-erf(lo) implementation returns zero on every one of these cells.
        # There is no absolute tolerance that could hide cancellation of this tiny signal.
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
        if not mixture:
            legacy, legacy_metadata = _legacy_image("system", n, 1, weight, tail_sign=sign)
            legacy_metadata.update(
                paired_generic_artifact_identity=metadata["artifact_identity"],
                classification="invalid-old-nonzero-cell-integral",
                invariant="a positive Gaussian has a strictly positive integral on every cell",
            )
            _export_image(
                tmp_path,
                record_property,
                "invalid-zero-tail",
                legacy,
                {0: expected},
                legacy_metadata,
            )
            # Classify the actually executed old failure, never equality to an invalid baseline.
            assert np.all(legacy[0][0] == 0)
        assert _binary_fingerprints(paths) == before
