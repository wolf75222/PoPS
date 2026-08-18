"""Compatibility shim. TR-01 authority lives in ``run.py`` and ``analyze.py``.

This module does not compute orders, write reports, or invent native results.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_run = load_sibling_module(Path(__file__).with_name("run.py"))

NativeUnavailable = _run.NativeUnavailable
AuthoringPending = _run.AuthoringPending
Tr01Config = _run.Tr01Config
variant_catalog = _run.variant_catalog
resolve_config = _run.resolve_config
resolve_config_id = _run.resolve_config_id
catalog = _run.variant_catalog
