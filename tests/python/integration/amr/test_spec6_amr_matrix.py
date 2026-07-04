#!/usr/bin/env python3
"""ADC-515: run + verify the declarative Spec 6 sec.20 Uniform x AMR matrix (one greppable table).

The matrix itself is :mod:`tests.python.integration.amr._spec6_matrix` (the ``MATRIX`` dict of
:class:`Cell` rows keyed ``"op.layout.blocks"``). This runner iterates every cell and dispatches by
its ``kind``, so one honest verdict per (operation x layout x block-count) cell:

  * ``green_inert`` / ``green_live`` -- run the check; it must return a proof token (an inert
    introspection result, or a real-``AmrSystem`` finite/conserved/clock-advanced run).
  * ``refuse``   -- the cell's ``_expect_refusal`` must raise the pinned exception with stable
    substrings and NO warning.
  * ``exists``   -- assert the CITED existing coverage still holds (a thin route-fact check), so an
    already-covered cell is pointed at, not duplicated.
  * ``pending``  -- the cell CONSTRUCTS its authoring object (proving the row is structurally real),
    then this runner ``pytest.skip``s it with the pending marker; it flips to a live cell when the
    named issue lands. No row is pending now: the multistep history rings flipped to exists with
    ADC-631, the clean-compile AMR explicit / SSPRK Program row to green_live with ADC-634, and the
    compiled condensed-Schur hierarchy Program row to green_live with ADC-633.

``test_matrix_is_complete`` pins ``set(MATRIX) == EXPECTED_KEYS`` so a dropped cell fails loud: a
future layout gap cannot hide. ``importorskip('pops')`` skips the whole suite on a bare box (this
Mac has no ``_pops``); the live cells step a real Kokkos-Serial engine on the CI runner. ``__main__``
runs pytest.
"""
import sys

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from tests.python.integration.amr import _spec6_matrix as matrix  # noqa: E402


@pytest.mark.parametrize("cell_id", sorted(matrix.MATRIX))
def test_spec6_cell(cell_id):
    cell = matrix.MATRIX[cell_id]
    if cell.kind == "pending":
        # Construct the authoring object (proves the row is real), then defer to the named issue. The
        # cell NEVER executes the deferred path and NEVER pins transitional behavior.
        marker = cell.run()
        pytest.skip("%s is %s (structurally inert; flips to live when the issue lands)"
                    % (cell_id, marker))
    token = cell.run()
    assert token, "cell %r (kind=%s) returned no proof token" % (cell_id, cell.kind)
    # The token names the verdict the cell proved, so a cell can never be a silent no-op: a green
    # cell's token carries its kind prefix (green_inert / green_live); a refuse cell returns the
    # pinned refusal message (the exact reject fired); an exists cell returns its citation.
    if cell.kind in ("green_inert", "green_live"):
        assert token.startswith(cell.kind), (
            "cell %r proved %r, inconsistent with kind=%s" % (cell_id, token, cell.kind))
    elif cell.kind == "exists":
        assert token.startswith("exists:"), (
            "cell %r must cite its coverage (exists:...), got %r" % (cell_id, token))


def test_matrix_is_complete():
    keys = set(matrix.MATRIX)
    expected = set(matrix.EXPECTED_KEYS)
    assert keys == expected, (
        "sec.20 matrix drifted: missing %s, unexpected %s"
        % (sorted(expected - keys), sorted(keys - expected)))
    # Every key is the "op.layout.blocks" shape (three dot-separated parts), so a malformed id fails.
    for key in keys:
        parts = key.split(".")
        assert len(parts) == 3, "cell id %r is not op.layout.blocks" % key
        cell = matrix.MATRIX[key]
        assert (cell.op, cell.layout, cell.blocks) == tuple(parts), (
            "cell %r fields %r disagree with its key" % (key, (cell.op, cell.layout, cell.blocks)))
        assert cell.kind in ("green_inert", "green_live", "refuse", "exists", "pending"), (
            "cell %r has an unknown kind %r" % (key, cell.kind))


def test_every_layout_is_amr_and_no_row_is_pending():
    # The AMR column is the ADC-515 focus: every cell targets the AMR layout (the Uniform baseline is
    # the shipping Spec 5 coverage the cited rows point at).
    for key, cell in matrix.MATRIX.items():
        assert cell.layout == "amr", "cell %r is not an AMR-layout cell" % key
    # NO row is pending: multistep-on-AMR (ab2/bdf2) flipped to exists when ADC-631 merged, the
    # clean-route explicit / SSPRK Program row to green_live with ADC-634, and the compiled
    # condensed-Schur hierarchy Program row to green_live with ADC-633.
    pending = {k for k, c in matrix.MATRIX.items() if c.kind == "pending"}
    assert pending == set(), pending


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
