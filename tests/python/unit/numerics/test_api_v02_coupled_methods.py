import math
import numpy as np
import pops
import pytest

from examples.api_v02.methods_coupled_nonsymmetric import (
    build_t04_nonsymmetric, build_t06_selected_global, build_t02_imex_second_order)
from examples.api_v02.methods_patankar_network import build_t11_network
from tests.python.support import coupled_methods_oracle as reference


def test_global_block_matrix_is_nonsymmetric_and_both_diagonals_are_spatial():
    a = reference.block_matrix(5)
    assert np.max(abs(a-a.T)) > .1
    assert np.linalg.matrix_rank(a) == 10
    assert np.count_nonzero(a[:5, :5]-np.diag(np.diag(a[:5, :5]))) > 0
    assert np.count_nonzero(a[5:, 5:]-np.diag(np.diag(a[5:, 5:]))) > 0


def test_distinct_global_roots_and_off_diagonal_partitions():
    positive = reference.selected_root(7, "unknown", 1)
    negative = reference.selected_root(7, "unknown", -1)
    frozen = reference.selected_root(7, "frozen", 1)
    assert np.min(positive) > 0 and np.max(negative) < 0
    assert np.max(abs(positive+negative)) > 1.e-3  # spatial operator rules out pointwise square roots
    assert np.max(abs(positive-frozen)) > .4


def test_imex_all_additive_order_two_conditions_and_observed_order():
    gamma = 1-1/math.sqrt(2)
    delta = 1-1/(2*gamma)
    explicit = np.array([[0., 0., 0.], [gamma, 0., 0.], [delta, 1-delta, 0.]])
    implicit = np.array([[0., 0., 0.], [0., gamma, 0.], [0., 1-gamma, gamma]])
    for weights in (explicit[-1], implicit[-1]):
        assert abs(weights.sum()-1) < 1.e-15
        for matrix in (explicit, implicit):
            assert abs(weights@matrix.sum(axis=1)-.5) < 1.e-15
    exact = reference.imex_exact(7, .1)
    errors = [np.max(abs(reference.imex_reference(7, .1/n, n)-exact)) for n in (4, 8, 16)]
    assert all(3.5 < a/b < 4.5 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("count", (3, 5, 8))
@pytest.mark.parametrize("empty", (False, True))
def test_mprk_network_conservation_nonnegative_and_directed_presentation(count, empty):
    values = reference.patankar_reference(count, .1, 6, empty=empty)
    expected_total = sum(1+.2*np.arange(count))-(1 if empty else 0)
    assert abs(sum(values)-expected_total) < 2.e-13
    assert min(values) > 0
    changed = reference.patankar_reference(count, .1, 6, circulation=1.5, empty=empty)
    assert np.max(abs(changed-values)) > 1.e-7


def test_mprk_second_order_on_a_general_five_species_network():
    exact = reference.network_exact(5, .6)
    values = [reference.patankar_reference(5, .6/n, n) for n in (6, 12, 24)]
    errors = [np.max(abs(v-exact)) for v in values]
    assert all(3.5 < a/b < 4.5 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("builder,kwargs", (
    (build_t04_nonsymmetric, {}),
    (build_t06_selected_global, {"mode": "frozen"}),
    (build_t06_selected_global, {"branch": -1}),
    (build_t02_imex_second_order, {}),
    (build_t11_network, {})))
def test_method_source_admission(builder, kwargs):
    spec = builder(**kwargs)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    assert "continuation_publish" in str(spec.case._time._serialize(include_provenance=False))
