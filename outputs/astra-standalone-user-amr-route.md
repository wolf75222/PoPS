# Standalone User AMR route admission

Base: `3bd6ea9b143145dafda13e9d8758e6416e69d273`, isolated branch
`codex/api040-standalone-user-amr`. No MAIN edits, package rebuild or installation.

## Real failure and cause

The installed reception log `outputs/installed-3bd6ea9-t2-amr/pytest.log` at the
workspace root records the failure of
`test_user_numerical_bodies_runtime.py::test_two_block_captures_and_rebind_match_independent_fv_oracle[2-amr1]`:

```
compiled AMR block: unknown limiter route 'source_stencil:7d46f416483194c77961d6099a65be3a50fbea7690d0638df1cd575adee6b240'
```

`AmrSystem::add_native_block` called the exact generated-policy validator before
loading the authenticated DSO, without a compiled policy identity. An absent
identity correctly means a catalogue-only policy in that validator. The host
therefore rejected the User route before reaching its actual typed installer.
The analogous `source_face:<sha>` admission defect follows the same call path;
the observed installed failure was specifically the reconstruction route.

## Correction

The loader now calls `validate_compiled_amr_system_block_route_syntax`. It admits
only the canonical `source_stencil:` and `source_face:` lowercase 64-hex payloads,
and retains all existing route/time/positivity/WENO/cache checks. This preflight
authenticates no body and publishes no prepared block.

The four generated `prepare_compiled_amr_system_block` overloads still use the
unchanged exact validator. Their compiled reconstruction/face identities must
equal the requested complete strings; a missing typed policy still refuses a
source route. The loaded model and binary identities are authenticated before
the installer, and the full limiter/Riemann strings, captures/parameter values,
model identity and binary identity remain in the existing collective package
contract before installation. No enum substitution, model dispatch or fallback
was introduced. Source policy callbacks and their ownership are unchanged.

## Checks performed

Two new tests in `test_generated_amr_system_block.cpp`:

- `SourceRoutePreflightDefersAuthenticationToCompiledPolicies`: all four
  catalogue/source combinations pass syntax; exact policies must match; missing
  policies and one-hex-digit identity changes refuse.
- `SourceRoutePreflightRefusesMalformedIdentityAndInvalidControls`: empty,
  short/long, uppercase/nonhex, suffixed and embedded-NUL identities refuse;
  reversed route kinds, partial storage, invalid reconstruction/time, negative
  positivity, unrelated WENO epsilon and unsupported wave cache still refuse.

The exact two gtest bodies were extracted into a small host TU and compiled
against this checkout's real production header with Apple clang C++20, the
existing Kokkos/MPI headers/libraries and gtest. **2/2 passed**. Receipt:
`outputs/standalone-user-amr-route-host.log`. The initial fixture assumed all
refusals were `invalid_argument`; catalogue parsers actually throw `runtime_error`.
Those expectations were corrected without changing production exception behavior.
This fixture failure is not presented as the original product regression.

`git diff --check` passed. Sol6Native independently reviewed the three-file diff
and verified the codegen→four exact overloads→materialization route. No additional
authentication bypass was found in that bounded review.

## Central reception still required

Rebuild the native runtime and generated artifacts because the installed loader
and SDK header both change. Run target `test_generated_amr_system_block` and the
two new gtests, then rerun the actual installed two-block User/capture/rebind AMR
case above (and the same family at other available AMR levels). The lightweight
checks validate admission/authentication, **not** a successful AMR bind or flux.
No new full TU build, MPI2 execution, GPU execution or native Python run was made
in this isolated checkout. Existing package collective/rollback code was read,
not requalified by these validator tests.
