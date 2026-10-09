"""Exact diagnostic provenance of this ALE shared-face/fixed-domain refusal.

The native ALE guard votes the mismatch before throwing on every rank. MPI
cadence/System phases choose the lowest failing rank and wrap as RuntimeError;
Python then records every receiver rank. An empty rank receives the same vote.
Serial execution preserves the native invalid_argument -> ValueError mapping.
"""


def require_exact_moving_interval_refusal(failures, *, size):
    if type(size) is not int or size < 1:
        raise ValueError("moving refusal requires a positive exact rank count")
    cause = "moving shared face or fixed-domain boundary is inconsistent"
    expected = ("ValueError", cause, False)
    if size > 1:
        native = (
            "System step failed collectively; rank 0: "
            "Program cadence phase failed collectively; rank 0: " + cause
        )
        message = "collective step attempt failed during solve: " + "; ".join(
            "rank %d RuntimeError: %s" % (rank, native) for rank in range(size)
        )
        expected = ("RuntimeError", message, True)
    assert isinstance(failures, tuple) and len(failures) == size, failures
    for failure in failures:
        assert type(failure) is tuple and len(failure) == 3, failures
        assert type(failure[0]) is str and type(failure[1]) is str and type(failure[2]) is bool, (
            failures
        )
        assert failure == expected, failures
