"""pops.codegen.env -- the codegen / compile ``POPS_*`` environment resolver (Spec 5 sec.12.4).

The single, honest place that reads the user-facing codegen environment variables and turns them
into a typed :class:`CodegenEnv` snapshot. The contract (shared with ``POPS_THREADS`` /
``POPS_PROFILE``, criteria #47-48, epic ADC-479) is additive:

  * the env only supplies a DEFAULT -- an explicit Python argument to ``compile_problem`` ALWAYS
    wins (each resolver below takes the explicit value first and falls back to the env);
  * coercion is transparent and lenient (an unparseable value is ignored, never raised -- no
    stricter rejection than passing the argument directly);
  * only implemented controls are accepted and recorded.

The variables (sec.12.4):

  ``POPS_LOG`` / ``POPS_CODEGEN_LOG``  a log level for the compile/codegen path (a simple, honest
      level gate -- quiet by default). ``POPS_CODEGEN_LOG`` is the codegen-specific name and wins
      over the broader ``POPS_LOG`` when both are set.
  ``POPS_CODEGEN_DIR``  the directory dumps (and a kept generated source) are written to by default.
  ``POPS_KEEP_GENERATED``  keep the generated ``.cpp`` next to the ``.so`` instead of discarding the
      temp dir (the same effect ``compile_problem(debug=True)`` has; the env supplies the default).
  ``POPS_DUMP_IR`` / ``POPS_DUMP_CPP``  after a successful compile, dump the IR / the C++ via the
      ``CompiledProblem.dump_ir`` / ``dump_cpp`` handles into the codegen dir.
  ``POPS_CACHE_DIR``  the out-of-source ``.so`` cache directory (read in :mod:`pops.codegen.cache`;
      recorded here for inspection).
  ``POPS_PROFILE``  the runtime profiling default (read in :mod:`pops.runtime.profile`; recorded
      here for inspection).
  ``POPS_AUTOTUNE``  off / basic / aggressive. No autotune engine exists today, so any value other
      than ``off`` is an HONEST no-op stub: it is recorded + surfaced in ``inspect()`` but changes
      NOTHING in the emitted code, hence it does NOT enter the cache key (stated, not faked). If a
      future tuner ever changes codegen, it must then enter the cache key.
This module imports only the standard library at module scope (``os`` / ``sys``); it
references no other ``pops`` layer, so it adds no edge to the codegen import graph
(tests/python/architecture/test_import_graph.py).
"""
from __future__ import annotations

from typing import Any

import os
import sys


# --- recognised env var names (one literal, here) ----------------------------------------------
ENV_LOG = "POPS_LOG"
ENV_CODEGEN_LOG = "POPS_CODEGEN_LOG"
ENV_CODEGEN_DIR = "POPS_CODEGEN_DIR"
ENV_KEEP_GENERATED = "POPS_KEEP_GENERATED"
ENV_DUMP_IR = "POPS_DUMP_IR"
ENV_DUMP_CPP = "POPS_DUMP_CPP"
ENV_CACHE_DIR = "POPS_CACHE_DIR"
ENV_PROFILE = "POPS_PROFILE"
ENV_AUTOTUNE = "POPS_AUTOTUNE"

# Log levels, quiet-first. A bad value falls back to the quietest honest default.
_LOG_LEVELS = ("quiet", "info", "debug")
_LOG_QUIET = _LOG_LEVELS[0]
_LOG_ALIASES = {"": _LOG_QUIET, "0": _LOG_QUIET, "off": _LOG_QUIET, "none": _LOG_QUIET,
                "false": _LOG_QUIET, "no": _LOG_QUIET,
                "1": "info", "info": "info", "on": "info", "true": "info", "yes": "info",
                "2": "debug", "debug": "debug", "verbose": "debug", "trace": "debug"}

# Autotune levels. Only "off" exists today; the others are honest no-op stubs.
_AUTOTUNE_LEVELS = ("off", "basic", "aggressive")
_AUTOTUNE_OFF = _AUTOTUNE_LEVELS[0]
_AUTOTUNE_ALIASES = {"": _AUTOTUNE_OFF, "0": _AUTOTUNE_OFF, "off": _AUTOTUNE_OFF,
                     "none": _AUTOTUNE_OFF, "false": _AUTOTUNE_OFF, "no": _AUTOTUNE_OFF,
                     "1": "basic", "basic": "basic", "on": "basic", "true": "basic", "yes": "basic",
                     "2": "aggressive", "aggressive": "aggressive", "full": "aggressive"}

# Truthy tokens for boolean controls, mirroring _env_truthy in
# toolchain.py: anything else (unset / blank / "0" / "off" / ...) is False.
_TRUTHY = ("1", "on", "true", "yes", "y")


def _truthy(value: Any) -> bool:
    """True for an explicit truthy token; False for unset / blank / falsey (lenient, never raises)."""
    return str(value or "").strip().lower() in _TRUTHY


def _level(value: Any, aliases: Any, default: Any) -> Any:
    """Map a raw env value to a known level via @p aliases; an unknown value -> @p default."""
    return aliases.get(str(value or "").strip().lower(), default)


def resolve_log_level(env: Any = None) -> Any:
    """The codegen/compile log level: POPS_CODEGEN_LOG (specific) wins over POPS_LOG (broad).

    Quiet by default. Returns one of ``"quiet"`` / ``"info"`` / ``"debug"``; an unrecognised value
    falls back to quiet (lenient -- a bad level never raises).
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_CODEGEN_LOG)
    if raw is None:
        raw = env.get(ENV_LOG)
    return _level(raw, _LOG_ALIASES, _LOG_QUIET)


def resolve_autotune(env: Any = None) -> Any:
    """The autotune level from ``POPS_AUTOTUNE``: ``"off"`` (default) / ``"basic"`` / ``"aggressive"``.

    HONEST STUB: no autotune engine exists, so any non-``off`` value is recorded and surfaced but
    changes nothing in the emitted code (and therefore does not enter the cache key). An unrecognised
    value falls back to ``off``.
    """
    env = os.environ if env is None else env
    return _level(env.get(ENV_AUTOTUNE), _AUTOTUNE_ALIASES, _AUTOTUNE_OFF)


class CodegenEnv:
    """An inert, inspectable snapshot of the active codegen ``POPS_*`` settings (Spec 5 sec.12.4).

    Built by :meth:`from_env`, recorded on the :class:`pops.codegen.loader.CompiledProblem` and
    surfaced in :meth:`CompiledProblem.inspect`. It carries the RESOLVED values (the env default
    already overridden by any explicit argument), so reading it tells you exactly what governed a
    compile. It performs no compile and no I/O; :meth:`to_dict` is a JSON-ready view.
    """

    __slots__ = ("log_level", "codegen_dir", "keep_generated", "dump_ir", "dump_cpp",
                 "cache_dir", "profile", "autotune", "_frozen")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("CodegenEnv is immutable after artifact sealing")
        object.__setattr__(self, name, value)

    def freeze(self) -> Any:
        object.__setattr__(self, "_frozen", True)
        return self

    def __init__(self, *, log_level: Any = _LOG_QUIET, codegen_dir: Any = None,
                 keep_generated: Any = False, dump_ir: Any = False, dump_cpp: Any = False,
                 cache_dir: Any = None, profile: Any = None,
                 autotune: Any = _AUTOTUNE_OFF) -> None:
        self.log_level = log_level
        self.codegen_dir = codegen_dir
        self.keep_generated = bool(keep_generated)
        self.dump_ir = bool(dump_ir)
        self.dump_cpp = bool(dump_cpp)
        self.cache_dir = cache_dir
        self.profile = profile
        self.autotune = autotune

    @classmethod
    def from_env(cls, *, codegen_dir: Any = None, keep_generated: Any = None,
                 env: Any = None) -> Any:
        """Resolve the snapshot, EXPLICIT arguments winning over the environment (sec.12.4).

        @p codegen_dir  an explicit codegen directory (wins over ``POPS_CODEGEN_DIR``);
        @p keep_generated  an explicit keep flag, e.g. ``compile_problem(debug=True)`` (wins over
            ``POPS_KEEP_GENERATED`` -- ``True`` forces keep, ``False``/``None`` defers to the env);
        @p env an env mapping (defaults to ``os.environ``; injected by tests).
        """
        env = os.environ if env is None else env

        # POPS_CODEGEN_DIR: explicit wins; the env supplies the default dump/keep directory.
        eff_dir = codegen_dir if codegen_dir is not None else (env.get(ENV_CODEGEN_DIR) or None)

        # POPS_KEEP_GENERATED: an explicit True forces keep; otherwise the env decides.
        eff_keep = bool(keep_generated) if keep_generated else _truthy(env.get(ENV_KEEP_GENERATED))

        return cls(
            log_level=resolve_log_level(env),
            codegen_dir=eff_dir,
            keep_generated=eff_keep,
            dump_ir=_truthy(env.get(ENV_DUMP_IR)),
            dump_cpp=_truthy(env.get(ENV_DUMP_CPP)),
            cache_dir=env.get(ENV_CACHE_DIR) or None,
            profile=env.get(ENV_PROFILE) or None,
            autotune=resolve_autotune(env))

    @property
    def verbose(self) -> bool:
        """True when the log level asks for at least ``info`` output."""
        return self.log_level != _LOG_QUIET

    def log(self, message: Any, *, level: str = "info", stream: Any = None) -> None:
        """Emit @p message to stderr iff the active level is at least @p level (else a no-op).

        An honest, dependency-free level gate: ``quiet`` < ``info`` < ``debug``. Used by the compile
        path to trace the steps when ``POPS_CODEGEN_LOG`` / ``POPS_LOG`` ask for it; silent otherwise.
        """
        want = _LOG_LEVELS.index(level) if level in _LOG_LEVELS else len(_LOG_LEVELS)
        have = _LOG_LEVELS.index(self.log_level) if self.log_level in _LOG_LEVELS else 0
        if have >= want and want < len(_LOG_LEVELS):
            print("pops.codegen: %s" % message, file=stream or sys.stderr, flush=True)

    def run_dumps(self, compiled: Any) -> None:
        """Honor POPS_DUMP_IR / POPS_DUMP_CPP on a freshly resolved handle (Spec 5 sec.12.4, #47).

        After a successful compile (or a cache hit), dump the IR / the C++ via the EXISTING
        ``CompiledProblem.dump_ir`` / ``dump_cpp`` into the codegen directory (``codegen_dir``, which
        already reflects ``POPS_CODEGEN_DIR``). A failed dump is logged and swallowed -- a dump is a
        diagnostic convenience, never a reason to fail an otherwise-successful compile. A no-op unless
        the corresponding env flag is set.
        """
        if not (self.dump_ir or self.dump_cpp):
            return
        out_dir = compiled.codegen_dir or "."
        name = compiled.program_name or "problem"
        if self.dump_ir:
            try:
                path = compiled.dump_ir(os.path.join(out_dir, "%s.ir.json" % name))
                self.log("compile_problem: POPS_DUMP_IR wrote %s" % path)
            except (OSError, ValueError) as exc:
                self.log("compile_problem: POPS_DUMP_IR skipped (%s)" % exc)
        if self.dump_cpp:
            try:
                path = compiled.dump_cpp(out_dir)
                self.log("compile_problem: POPS_DUMP_CPP wrote %s" % path)
            except (OSError, ValueError, NotImplementedError) as exc:
                self.log("compile_problem: POPS_DUMP_CPP skipped (%s)" % exc)

    def to_dict(self) -> dict:
        """A plain-dict, JSON-ready view of the resolved settings (inspectable, never an array)."""
        return {"log_level": self.log_level, "codegen_dir": self.codegen_dir,
                "keep_generated": self.keep_generated, "dump_ir": self.dump_ir,
                "dump_cpp": self.dump_cpp, "cache_dir": self.cache_dir, "profile": self.profile,
                "autotune": self.autotune}

    def __repr__(self) -> str:
        return ("CodegenEnv(log_level=%r, codegen_dir=%r, keep_generated=%s, dump_ir=%s, "
                "dump_cpp=%s, autotune=%r)"
                % (self.log_level, self.codegen_dir, self.keep_generated, self.dump_ir,
                   self.dump_cpp, self.autotune))


__all__ = ["CodegenEnv", "resolve_log_level", "resolve_autotune",
           "ENV_LOG", "ENV_CODEGEN_LOG", "ENV_CODEGEN_DIR", "ENV_KEEP_GENERATED", "ENV_DUMP_IR",
           "ENV_DUMP_CPP", "ENV_CACHE_DIR", "ENV_PROFILE", "ENV_AUTOTUNE"]
