"""Redaction helper for the compile-command introspection string (Spec 5 sec.12.4, #49).

Extracted from :mod:`pops.codegen.compile_drivers` to keep that module under the Spec-4 500-line
budget. ``_redact_compile_command`` is a pure function (no codegen state) re-imported by
``compile_drivers`` and called once per fresh compile.
"""

from __future__ import annotations

from typing import Any


def _redact_compile_command(cmd: Any, *, tmp_cpp: Any, gen_src: Any) -> str:
    """Return a redacted compile-command STRING for introspection (Spec 5 sec.12.4, #49).

    The raw argv is safe to surface (it is a compiler invocation, not a credential), but two things
    are normalised so the string is stable and leaks nothing machine-specific: the ephemeral
    TemporaryDirectory .cpp path is replaced by the persistent generated-source path (or
    ``<generated>``), and any token that looks like a secret/credential (``*token*`` / ``*secret*``
    / ``*password*`` / ``*key*=value``) is masked. Header / include / library paths are KEPT (they
    are part of the reproducible toolchain), only obvious secrets are masked."""
    masked = []
    for tok in cmd:
        if tok == tmp_cpp:
            masked.append(gen_src)
            continue
        low = tok.lower()
        if (("token" in low or "secret" in low or "password" in low or "passwd" in low)
                and "=" in tok):
            masked.append(tok.split("=", 1)[0] + "=<redacted>")
        else:
            masked.append(tok)
    return " ".join(masked)
