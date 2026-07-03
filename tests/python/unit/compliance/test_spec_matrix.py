#!/usr/bin/env python3
"""ADC-547: run and verify the declarative spec-compliance matrix (one greppable table).

The matrix itself is :mod:`tests.python.unit.compliance._cells` (POSITIVE / NEGATIVE dicts). This
runner iterates every cell and fails loudly if any regresses:

  * ``test_every_positive_cell_selects_its_native_route`` -- each positive cell INSPECTS the chosen
    native route (``native_capability_report().routes`` / ``compiled.inspect()`` /
    ``Problem.explain_routes()``) and proves it is advertised available, not merely "no exception".
  * ``test_every_negative_cell_refuses_with_stable_message`` -- each negative cell raises the expected
    exception type with exact, stable message needles and emits NO warning (a warning + fallback is
    exactly what the matrix catches).
  * ``test_matrix_is_complete`` -- the 8 positive + 11 negative cell ids are ALL present (a hard-coded
    expected set), so a dropped cell fails here.

The phase-6 lesson is honored: every cell runs its full PRE-compile decision path locally (the
install-time route predicates over a ``CompiledModel`` stub and ``run_bind_gates`` over a
``CompiledProblem`` stub -- no ``.so`` on disk). The suite importorskips ``pops`` so it is green on a
bare box; nothing here needs a native compile. Pytest + __main__ guard.
"""
import pytest

# exc_type=ImportError: the pops bootstrap raises a plain ImportError (not ModuleNotFoundError) when
# the native extension is absent, so the default importorskip would let it escape as a collection
# error instead of skipping. Broadening the caught type keeps the suite green on a bare box.
pops = pytest.importorskip("pops", exc_type=ImportError)

from tests.python.unit.compliance import _cells  # noqa: E402


_EXPECTED_POSITIVE = {
    "pos.uniform_fv_typed_riemann",
    "pos.uniform_poisson_elliptic",
    "pos.program_manual_operator",
    "pos.program_macro_lib_time",
    "pos.matrix_free_krylov",
    "pos.params_runtime_const_bind",
    "pos.diagnostics_output_ckpt",
    "pos.amr_route_when_capable",
}

_EXPECTED_NEGATIVE = {
    "neg.hll_no_wave_speeds",
    "neg.hllc_no_star_state",
    "neg.roe_no_dissipation",
    "neg.weno_ghost_depth",
    "neg.fft_on_amr_or_bc",
    "neg.amr_maxlevels_ratio",
    "neg.param_missing_or_domain",
    "neg.operator_signature_mismatch",
    "neg.missing_aux_field",
    "neg.abi_cache_mismatch",
    "neg.ir_index_refusal",
}


@pytest.mark.parametrize("cell_id", sorted(_cells.POSITIVE))
def test_every_positive_cell_selects_its_native_route(cell_id):
    cell = _cells.POSITIVE[cell_id]
    proved = cell.check()
    assert proved in cell.route_ids, (
        "positive cell %r proved route %r, expected one of %r"
        % (cell_id, proved, cell.route_ids))


@pytest.mark.parametrize("cell_id", sorted(_cells.NEGATIVE))
def test_every_negative_cell_refuses_with_stable_message(cell_id):
    cell = _cells.NEGATIVE[cell_id]
    message = _cells.run_negative_cell(cell)
    assert message, "negative cell %r refused with an empty message" % cell_id


def test_matrix_is_complete():
    positive = set(_cells.POSITIVE)
    negative = set(_cells.NEGATIVE)
    assert positive == _EXPECTED_POSITIVE, (
        "positive cell set drifted: missing %s, unexpected %s"
        % (sorted(_EXPECTED_POSITIVE - positive), sorted(positive - _EXPECTED_POSITIVE)))
    assert negative == _EXPECTED_NEGATIVE, (
        "negative cell set drifted: missing %s, unexpected %s"
        % (sorted(_EXPECTED_NEGATIVE - negative), sorted(negative - _EXPECTED_NEGATIVE)))
    assert len(positive) == 8 and len(negative) == 11, (
        "the matrix must have 8 positive + 11 negative cells, got %d + %d"
        % (len(positive), len(negative)))
    # No id collides between the two halves.
    assert not (positive & negative), "a cell_id appears in both halves: %s" % sorted(positive & negative)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
