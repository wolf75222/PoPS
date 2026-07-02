"""Shared assertion helpers for the Python test tree.

Importable both under pytest (REPO_ROOT is on sys.path via the rootdir) and from a
process-isolated script (conftest._process_pythonpath puts REPO_ROOT on PYTHONPATH), so a call site
uses ``from tests.python.support.assertions import _check`` in either mode.
"""

from __future__ import annotations


def _check(cond: object, msg: str) -> None:
    """Raise ``AssertionError(msg)`` when @p cond is falsy (the script-style check helper).

    The canonical copy of the ``_check(cond, msg)`` guard duplicated across the script-style tests.
    ``test_operator_introspection._check(obj)`` is a DIFFERENT function (a model validator) and is
    deliberately not replaced by this one.
    """
    if not cond:
        raise AssertionError(msg)
