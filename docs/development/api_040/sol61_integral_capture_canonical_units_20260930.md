# Native canonical units for integral candidate captures

Date: 30 September 2026. Author: GPT-6.1 Sol. Exact source base:
`bb87174d82d95422c3086eea7b0c28cf8390fb15`. Worktree:
`PoPS-sol61-integral-units`, branch `codex/api040-sol61-integral-units`.
The independent T3 review remains frozen separately at `3e962835`.

## Defect and correction

`PreparedIntegralCapture::require_units_` authenticated a digest of the supplied
string but did not authenticate its physical type. Replacing the units with
`not-json` or `{"powers":[],"kind":"physical_dimension"}`, then redigesting
the v2 identity, reached a live capture. The latter is valid JSON with a
noncanonical key order. The digest proves byte identity, not typed canonicality.

The new targeted native reader authenticates precisely the compact, sorted-key,
ASCII-escaped `PhysicalDimension.to_data()` JSON image used by
`integral_units_bytes`. It requires:

- Exact `kind`/`powers` structure, key order and compact separators, with no extra
  keys, whitespace, suffix, duplicate keys or alternate JSON value types.
- Nonempty JSON string bases, with Python `ensure_ascii=True` escaping. Short
  control escapes, lowercase Unicode escapes and surrogate pairs are supported.
  Bases compare by decoded Unicode codepoints, not UTF-16 escape bytes; names
  must be unique and strictly sorted. U+E000 therefore precedes U+10000.
- Three entries per power: base, nonzero signed integer numerator, positive
  integer denominator. No negative zero, leading zero, boolean, quoted integer,
  exponent or floating spelling is admitted. Zero powers must be omitted.
- Reduced fractions. Decimal long division and Euclid's algorithm validate the
  GCD without conversion to floating point, int64, or another bounded integer
  type. There is no artificial bound on exponent metadata digits or bases.

The reader is intentionally specific to this physical type. Repository search
found a flat external-brick manifest reader, but no general native JSON codec
that authenticates this schema and spelling. No JSON or bigint dependency was
introduced. Exponents remain metadata; this code does not convert numerical units.

`require_units_` now calls the canonical reader before hashing. Preparation and
consumption retain their existing try/catch, exception vote and exact ordered
contract agreement. A rank-local parsing/allocation failure therefore converges
before a candidate POD is returned. No controller, ledger, cache ownership,
attempt, point, consumer or rollback algorithm changed. Canonical units retain
their original bytes, digest and v2 identity; the contract already required
exact typed units, so this closes its missing validation rather than changing
the successful serialization. Untyped v1 integral declarations are unchanged
and cannot authorize a typed candidate capture.

The new transitive implementation header is classified `sdk-support` in
`include/pops_headers.manifest`; it participates in installation and header
authentication. Central SDK rebuilding belongs to the parent task. Neither MAIN
nor the installed `pops-api040` environment was changed by this work.

## Executed evidence

The actual dependency-free native reader was compiled with C++20 and
`-Wall -Wextra -Werror` and run from a source-only pytest fixture:

- 92 images emitted by the real Python physical type pass: dimensionless,
  negative powers, Unicode/control/quote/backslash bases, non-BMP ordering,
  80 deterministic random rational powers, and a 2049-digit numerator.
- 48 malformed/noncanonical images refuse, including redigestable structural,
  escaping, arity, ordering, zero, sign, integer-type and fraction failures.
- 60 additional large reduced fractions pass and their 60 common-factor
  variants refuse.

The separate C++ host probe exercises the actual `PreparedIntegralCapture`,
`PreparedResourceCache` and `AcceptedExchangeLedger`, using an explicitly
documented private access shim. Its **38 assertions pass**. Five redigested
invalid images refuse before attempt/ledger getters; three valid Unicode/large
rational images capture and consume the exact .7 value with unchanged ledger.
Noncanonical consumption units also refuse before getters. Compiling this same
probe against an authenticated `git archive` of the exact base header produces
exit 1: `redigested units reached live capture authority`. The base replay loads
the old production capture class; the new reader included by the probe is not
called by that class.

Source/host regression command:

```sh
rtk proxy env -u PYTHONPATH PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_integral_units_native_canonical.py \
  tests/python/unit/codegen/test_integral_candidate_capture.py \
  tests/python/unit/codegen/test_integral_state_independent.py \
  tests/python/unit/codegen/test_integral_state_names.py \
  tests/python/unit/codegen/test_program_v8_contract.py
```

Result: **25 PASS**, including five new source/host tests. Python canonical-unit
emission and the digest in a real authored IntegralState identity are checked.
Ruff and diff whitespace validation pass.

Actual class/cache/ledger host command:

```sh
rtk proxy clang++ -std=c++20 -O0 -Wall -Wextra -Werror -Iinclude \
  tests/review/sol61_integral_units_capture_host.cpp \
  -o /tmp/sol61-integral-units-capture-host
rtk proxy /tmp/sol61-integral-units-capture-host
```

The full `test_program_runtime.cpp` translation unit, including the new public
fixtures, passes genuine header syntax checking, Dim1, MPI and Kokkos enabled:

```sh
rtk proxy clang++ -std=c++20 -fsyntax-only -DPOPS_NATIVE_DIM=1 \
  -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE \
  -DPOPS_HAS_MPI -DPOPS_HAS_PARALLEL_HDF5 -Iinclude -Itests/cpp/support \
  -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
  -I/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS/build-mpi/_deps/googletest-src/googletest/include \
  -Xpreprocessor -fopenmp tests/cpp/integration/runtime/test_program_runtime.cpp
```

There is one existing external gtest char8_t conversion warning. The first
manual attempt omitted the repository's `tests/cpp/support` include directory;
the corrected command above succeeds.

## Public fixtures reserved for central execution

Two tests were added to the existing genuine System/Program provider fixture:

- `IntegralCaptureRejectsRedigestedUntypedOrNoncanonicalUnits`: five invalid
  variants, injected on all ranks and then on rank zero only. Every rank
  declares the same two ledger identities before the snapshot, so refusal is
  not accidentally qualified by differing declarations. Real
  `declare_integral_state`, `capture_integral_candidate` and
  `integral_candidate_value` are invoked in a live System step. Consumption
  must remain unreached; fields, ledger, time and macrostep must roll back.
  Serial requires invalid_argument; multi-rank lanes require runtime_error from
  the existing collective exception boundary.
- `IntegralCapturePreservesCanonicalUnicodeAndLargeRationals`: dimensionless,
  Unicode rational and greater-than-uint64 metadata cases must consume the exact
  value and accept the real step, retaining the ledger and advancing clocks.

These extend the public counterexample frozen independently by Hypatia at
`18680869`, with no private access in these public tests. The fixtures are
syntax-received but **not executed here**. No public System native run, MPI
rank-local failure, GPU, scientific feedback, checkpoint codec or M14/M27
qualification is claimed by the host results. Parent owns the rebuilt SDK and
actual native/collective reception.
