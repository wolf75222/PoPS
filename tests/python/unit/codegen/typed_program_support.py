"""Typed Program fixtures shared by codegen tests.

The implementation is owned by the time-language tests.  Codegen tests are also executed as
standalone scripts, so their directory is the only unit-test directory guaranteed on ``sys.path``;
load that fixture module under an explicit private name and re-export its final-API helpers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PATH = Path(__file__).parents[1] / "time" / "typed_program_support.py"
_SPEC = importlib.util.spec_from_file_location("_pops_time_typed_program_support", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - impossible in a source checkout
    raise ImportError("cannot load typed Program fixtures from %s" % _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _MODULE)
_SPEC.loader.exec_module(_MODULE)

fresh_state_refs = _MODULE.fresh_state_refs
commits_by_block = _MODULE.commits_by_block
state_refs = _MODULE.state_refs
typed_field = _MODULE.typed_field
typed_state = _MODULE.typed_state

__all__ = [
    "commits_by_block", "fresh_state_refs", "state_refs", "typed_field", "typed_state",
]
