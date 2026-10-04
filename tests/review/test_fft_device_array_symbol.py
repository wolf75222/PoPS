"""Source mechanism equivalence; actual CUDA acceptance is a separate run."""
import re
import os
import hashlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT/'include/pops/numerics/elliptic/poisson/poisson_fft_device_kernels.hpp'


def negative():
    filename=os.environ.get('POPS_FFT_SYMBOL_NEGATIVE_HEADER')
    if not filename:pytest.skip('exact preserved d2db negative kernel header must be supplied')
    body=Path(filename).read_bytes()
    assert hashlib.sha256(body).hexdigest()=='d1c8e9d3227397b11a21a1800b5d6d5b02b4683f9fc9582f2060179db3e58870'
    return body.decode()


def body(text):
    start=text.index('  static void inverse_symbol(')
    return text[start:text.index('\n};',start)]


def tokens(text):
    return re.findall(r'\w+|[^\s]',text)


def test_symbol_arithmetic_and_operation_order_unchanged():
    old=body(negative());new=body(HEADER.read_text())
    old_lambda=old[old.index('KOKKOS_LAMBDA'):]
    new_lambda=new[new.index('KOKKOS_LAMBDA'):].replace('device_cells','cells').replace('device_spacing','spacing')
    assert tokens(old_lambda)==tokens(new_lambda)


def test_host_plan_interface_and_other_kernels_unchanged():
    old=negative();new=HEADER.read_text()
    assert old[:old.index('  static void inverse_symbol(')]==new[:new.index('  static void inverse_symbol(')].replace('#include <Kokkos_Array.hpp>\n','')
    assert old[old.index('  static void inverse_symbol('):old.index('    Kokkos::parallel_for(',old.index('  static void inverse_symbol('))].split('{',1)[0]==new[new.index('  static void inverse_symbol('):new.index('    Kokkos::parallel_for(',new.index('  static void inverse_symbol('))].split('{',1)[0]


def test_device_lambda_owns_device_callable_array_values():
    new=body(HEADER.read_text());lambda_body=new[new.index('KOKKOS_LAMBDA'):]
    assert 'Kokkos::Array<int, Dim> device_cells{};' in new
    assert 'Kokkos::Array<double, Dim> device_spacing{};' in new
    assert 'device_cells[axis] = cells[axis];' in new and 'device_spacing[axis] = spacing[axis];' in new
    assert not re.search(r'(?<!device_)\b(cells|spacing)\[',lambda_body)
    assert not any(word in new for word in ('malloc','new ','make_shared','create_mirror'))
