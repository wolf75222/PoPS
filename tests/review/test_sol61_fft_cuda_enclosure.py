"""Source equivalence and actual negative receipt, without CUDA qualification."""
import hashlib
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "include/pops/numerics/elliptic/poisson"


def _calls(source, pattern):
    calls = []
    for match in re.finditer(pattern, source):
        cursor, depth = match.end(), 1
        while depth:
            if source[cursor] == "(": depth += 1
            if source[cursor] == ")": depth -= 1
            cursor += 1
        assert source[cursor] == ";"
        calls.append(source[match.start():cursor + 1])
    return calls


def test_device_lambda_enclosing_functions_are_named_public_detail_members():
    plan = (DIRECTORY / "poisson_fft.hpp").read_text()
    kernels = (DIRECTORY / "poisson_fft_device_kernels.hpp").read_text()
    assert "KOKKOS_LAMBDA" not in plan
    assert kernels.count("KOKKOS_LAMBDA") == 11
    assert "struct PoissonFFTDeviceKernels" in kernels
    assert "private:" not in kernels and "protected:" not in kernels
    assert len(re.findall(r"static void [a-z_]+\(", kernels)) == 11
    assert len(_calls(plan, r"device_kernels::[a-z_]+\(")) == 11
    assert "using device_view = Kokkos::View<complex_type*, MemorySpace>;" in kernels
    assert "KOKKOS_LAMBDA" not in plan[plan.index(" private:"):]


def test_all_eleven_launch_expressions_and_non_kernel_plan_bytes_are_preserved():
    filename = os.environ.get("POPS_FFT_NEGATIVE_HEADER")
    if not filename:
        pytest.skip("exact preserved b376 negative header must be explicitly supplied")
    raw = Path(filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "6594028d0c5eb8e7203d8935a16326aedf20b24da96a062aeb9e04308804c92a"
    old = raw.decode()
    plan = (DIRECTORY / "poisson_fft.hpp").read_text()
    kernels = (DIRECTORY / "poisson_fft_device_kernels.hpp").read_text()
    old_launches = _calls(old, r"Kokkos::parallel_for\(")
    new_launches = _calls(kernels, r"Kokkos::parallel_for\(")
    normalize = lambda text: re.sub(r"\s+", "", text.replace("local_count_", "local_count"))
    assert len(old_launches) == len(new_launches) == 11
    assert list(map(normalize, old_launches)) == list(map(normalize, new_launches))
    original_reverse = re.search(r"  static POPS_HD int reverse_bits_\(.*?\n  }\n", old, re.S).group()
    new_reverse = re.search(r"  static POPS_HD int reverse_bits_\(.*?\n  }\n", kernels, re.S).group()
    assert original_reverse == new_reverse
    new_calls = _calls(plan, r"device_kernels::[a-z_]+\(")
    restored = plan.replace("#include <pops/numerics/elliptic/poisson/poisson_fft_device_kernels.hpp>\n", "")
    restored = restored.replace("  using device_kernels = detail::PoissonFFTDeviceKernels<Dim, MemorySpace>;\n\n", "")
    for call, launch in zip(new_calls, old_launches): restored = restored.replace(call, launch)
    restored = restored.replace("  void distributed_last_radix2_", original_reverse + "\n  void distributed_last_radix2_")
    assert restored == old
    assert plan[plan.index("class PoissonFFT {"):plan.index(" private:")] == old[old.index("class PoissonFFT {"):old.index(" private:")]
