"""Source/emission and spectral cross-check of the real Dim3 native fixture; no JIT."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


WORKER = r"""
import importlib.util, json, pathlib, sys
import numpy as np
root, output, order = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]), tuple(map(int, sys.argv[3]))
sys.path[:0] = [str(root / "python"), str(root)]
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
fixture = root / "tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py"
spec = importlib.util.spec_from_file_location("dim3_runtime_fixture", fixture)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.pops is pops
from pops._native_selector import selected_native_dimension
assert selected_native_dimension() is None
resolved, subject = module._case(order)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
source = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
assert selected_native_dimension() is None
assert "PreparedCoupledGradient<pops::kNativeDimension, 2>" in source
assert source.count(".stage_accepted_exchanges(") == 4
assert source.count("/occurrence:0") >= 2 and source.count("/occurrence:1") >= 2
output.write_text(source)
initial = module._initial()
variation = module._check_initial(initial)
assert initial.shape == (2, 3, 4, 6) and module.EXPECTED_RECORDS == 3456
predictor = initial + module.DT * module._rate(initial)
stencil = .5 * (initial + predictor + module.DT * module._rate(predictor))
# Independent Fourier-space SSPRK2 amplification polynomial of the discrete tensor Laplacian.
kx, ky, kz = (-4 * np.sin(np.pi * np.arange(n) / n)**2 / h**2
              for n, h in zip(module.CELLS, module.SPACING, strict=True))
symbol = kx[None,None,:] + ky[None,:,None] + kz[:,None,None]
field = np.fft.fftn(initial, axes=(1,2,3))
first = np.einsum("ab,bzyx->azyx", module.B, field)
second = np.einsum("ab,bzyx->azyx", module.B @ module.B, field)
factor = module.DT * sum(module.WEIGHTS) * symbol
spectral = np.fft.ifftn(field + factor*first + .5*factor**2*second, axes=(1,2,3))
np.testing.assert_allclose(spectral.imag, 0, rtol=0, atol=8.e-15)
np.testing.assert_allclose(stencil, spectral.real, rtol=0, atol=8.e-15)
np.testing.assert_allclose((stencil-initial).sum(axis=(1,2,3))*module.VOLUME, 0, rtol=0, atol=8.e-15)
initial_energy, final_energy = (float(np.sum(value**2)*module.VOLUME) for value in (initial,stencil))
assert final_energy < initial_energy - 1.e-9
print(json.dumps({"scope":"source/emission/math only; no native selected or JIT", "order":order,
    "source_import":pops.__file__, "stage_exchange_calls":4, "planned_native_records":3456,
    "axis_variation":variation, "spectral_max_error":float(np.max(np.abs(stencil-spectral.real))),
    "initial_energy":initial_energy, "final_energy":final_energy}))
"""


@pytest.mark.parametrize("order", ("01", "10"))
def test_dim3_public_fixture_emits_four_occurrences_and_has_independent_spectral_oracle(
    tmp_path, order
):
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("POPS_NATIVE_DIM", None)
    source = tmp_path / f"coupled-gradient-dim3-order-{order}.cpp"
    result = subprocess.run(
        [sys.executable, "-I", "-c", WORKER, str(root), str(source), order],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["order"] == list(map(int, order))
    assert receipt["stage_exchange_calls"] == 4
    assert receipt["spectral_max_error"] < 8.0e-15
    assert source.stat().st_size > 0
