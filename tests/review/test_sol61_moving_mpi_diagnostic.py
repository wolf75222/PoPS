"""Independent source tests of exact nested MPI provenance, including empty ranks.

No native guard or MPI collective is executed. The actual Python vote consumes
recorded native diagnostics; only its world/allgather transport is substituted.
"""

import ast
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pops import _native_collectives
from tests.python.support.moving_interval_diagnostics import require_exact_moving_interval_refusal


# Execute unchanged production function bodies without importing the native bootstrap.
# Only ordinary fatal errors are exercised; this nominal typed-rejection class is never used.
class UnusedTypedRejection(RuntimeError):
    pass


_source = Path(__file__).resolve().parents[2] / "python/pops/runtime/_step_strategy.py"
_tree = ast.parse(_source.read_text())
_selected = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_phase", "_attempt_error_record", "_collective_attempt_error"}
]
_assert_names = {node.name for node in _selected}
assert _assert_names == {"_phase", "_attempt_error_record", "_collective_attempt_error"}
_step_strategy = ModuleType("isolated_actual_step_error_functions")
_step_strategy.__dict__.update(
    Mapping=Mapping,
    StepAttemptRejected=UnusedTypedRejection,
    _MISSING=object(),
    _attempt_world=lambda _engine: None,
)
_module = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *_selected,
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(_module), str(_source), "exec"), _step_strategy.__dict__)

CAUSE = "moving shared face or fixed-domain boundary is inconsistent"
NATIVE = (
    "System step failed collectively; rank 0: "
    "Program cadence phase failed collectively; rank 0: " + CAUSE
)


def independent_record(size):
    if size == 1:
        return ("ValueError", CAUSE, False)
    return (
        "RuntimeError",
        "collective step attempt failed during solve: "
        + "; ".join("rank %d RuntimeError: %s" % (rank, NATIVE) for rank in range(size)),
        True,
    )


@pytest.mark.parametrize("owned", ((16,), (8, 8), (0, 16), (16, 0), (0, 8, 8)))
def test_actual_python_vote_preserves_every_receiver_and_native_rank_zero(monkeypatch, owned):
    size = len(owned)
    results = []
    for rank, local_patches in enumerate(owned):
        # Native all_reduce mismatch causes every rank to throw, irrespective of local_size.
        # The zero-patch row is not omitted from the actual Python consensus algorithm.
        world = SimpleNamespace(rank=rank, size=size, local_size=local_patches)
        monkeypatch.setattr(
            _step_strategy, "_attempt_world", lambda _engine, selected=world: selected
        )
        local = ValueError(CAUSE) if size == 1 else RuntimeError(NATIVE)
        sent = []

        def gathered(actual_world, envelope, selected=world, recorded=sent):
            assert actual_world is selected
            recorded.append(envelope)
            return tuple(
                {"rank": peer, "error": _step_strategy._attempt_error_record(RuntimeError(NATIVE))}
                for peer in range(size)
            )

        monkeypatch.setattr(_native_collectives, "allgather_value", gathered)
        error = _step_strategy._collective_attempt_error(object(), local)
        if size == 1:
            assert error is local and sent == []
        else:
            assert sent == [{"rank": rank, "error": _step_strategy._attempt_error_record(local)}]
        results.append((type(error).__name__, str(error), isinstance(error, RuntimeError)))
    assert tuple(results) == (independent_record(size),) * size
    require_exact_moving_interval_refusal(tuple(results), size=size)


@pytest.mark.parametrize(
    "forged",
    (
        ("ValueError", CAUSE, False),
        (
            "RuntimeError",
            "collective step attempt failed during solve: rank 0 ValueError: "
            + CAUSE
            + "; rank 1 ValueError: "
            + CAUSE,
            True,
        ),
        (
            "RuntimeError",
            independent_record(2)[1].replace("Program cadence phase", "Program unrelated phase"),
            True,
        ),
        ("RuntimeError", independent_record(2)[1].replace("; rank 0: ", "; rank 1: "), True),
        (
            "RuntimeError",
            independent_record(2)[1].replace("System step failed collectively; rank 0: ", ""),
            True,
        ),
        ("RuntimeError", independent_record(2)[1] + "; unrelated failure", True),
        ("RuntimeError", independent_record(2)[1], False),
        ("RuntimeError", independent_record(2)[1], 1),
    ),
)
def test_exact_fixture_refuses_wrong_category_rank_layers_cause_or_envelope(forged):
    with pytest.raises(AssertionError):
        require_exact_moving_interval_refusal((forged, forged), size=2)


@pytest.mark.parametrize(
    "failures", ((), (None, None), (independent_record(2),), (independent_record(2), None))
)
def test_exact_fixture_refuses_missing_or_successful_rank(failures):
    with pytest.raises(AssertionError):
        require_exact_moving_interval_refusal(failures, size=2)


@pytest.mark.parametrize("size", (True, 0, -1, 2.0, "2"))
def test_rank_count_is_not_coerced(size):
    with pytest.raises(ValueError):
        require_exact_moving_interval_refusal((), size=size)
