# C++ local symbol contract

`cpp-local-symbols@2` separates accepted public names from C++ local identifiers.
The real common emitter is `pops.codegen.cpp_writer`; public Module names,
component strings, ProviderPack keys, operator handles and Case/Model owners remain
exact. No declaration or provider identity is renamed to make a formula compile.

A Source/Host counterexample uses the accepted Aux component `difference gain`.
The previous Program emitter declared `const pops::Real difference gain`, while
its common expression writer referenced `difference_gain`. Clang rejected the
actual Program translation unit. A sanitizer alone is insufficient: `a b` and
`a_b`, or two different Greek names, can share the same sanitized identifier.

The additive `@2` revision also reserves the generated State/Prim/Schema types
and Axis template parameter. A non-author full TU demonstrated that a valid
public component named `State` otherwise hid the required type in conversion.
`cpp-public-text@2` is the separate canonical C++ string boundary. Quotes,
backslashes and line breaks are escaped; invalid JSON-only low Unicode escapes
and surrogate pairs are emitted as valid C++ UTF-8 encodings. Existing ordinary
JSON/C++ spellings remain unchanged. Length-aware std::string positions preserve
embedded NUL. Module comments escape line breaks too. Structured contract JSON
continues to use the existing serialization; only the surrounding C++ literal
changes. This covers component labels, provider keys/contracts/owners, publication
keys, public metadata and names, and the JSON metadata exports.

Each Model or whole-Program emission now installs a deterministic typed table
`(Var.kind, Var.name) -> identifier`. Valid noncolliding spellings keep their
legacy spelling. A collision uses the UTF-8 bytes of its kind and exact name;
all authored base spellings are reserved before choosing an escape. An authored
name that itself resembles an escape stays distinct. Existing generated CSE
locals and the common Program provider/index/runtime bindings are also kept
distinct. Tables are restored on every exit, including exceptions; concurrent
emitters use independent ContextVar contexts.

Conservative, primitive and Aux declarations, conversion methods, recovery
predicates, and common expression/Roe references use this same table. Primitive
state slots distinguish defined primitives from pass-through conservative
coordinates. Opt-in reciprocal locals use typed `hoist` identities; the existing
reciprocal transformation and opt-in rounding policy are unchanged. CSE scalar
bindings accept typed keys while retaining the existing plain-string protocol.
The local residual writer still materializes every referenced scalar leaf, so
an Aux NaN cannot disappear behind IEEE `fmin` before the existing finite guard.

This is a private emission contract, with no C++ ABI, field storage, registry,
rollback, equation or tolerance change. The existing compiler artifact identity
receives the complete generated source. A new installed build and C25 authority
are required before any Native execution; no old compiled artifact is qualified
by the Source/Host checks below.

## Bounded reproduction

From the candidate checkout, using the existing read-only local environment:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python \
 tests/review/sol61_cpp_local_symbol_replay.py /absolute/new/evidence-directory \
 /Users/romaindespoulain/dev/tmp/sol61-provider-instance-independent-20261005/case_aux_space_name_red.py
```

The script emits the original non-author public counterexample plus independent
collision/Unicode, reversed-provider, component/recovery and derived-primitive
profiles. It compiles twenty-five complete actual Program/Model translation units
against the real candidate headers. A compiled extraction of their actual variable
metadata checks every declared UTF-8 byte, including quotes, backslashes, newline,
BMP/astral Unicode and reserved type/template spellings. A compiled scalar control distinguishes six
Aux values with the exact result 91. A second real common-emitter control rejects
a nonfinite Aux leaf even when the final IEEE min result is finite. Three complete
legacy Program/Model C++ files are compared byte for byte against actual base Git
sources. These controls do not execute a PoPS runtime or qualify physical states.

Source tests also retain provider ownership, conditional evaluation boundaries,
primitive inverses, coupled-rate and consumed-publication structural checks.
The two installed Native tests in the selected existing suite are explicitly
deselected; they need ROOT's fresh build/install and C25 before bind. There is no
Native, MPI, CUDA, HIP, AMR, restart or scientific qualification from this packet.
The original non-author RED and all intermediate failed Source/Host receipts are
preserved externally alongside the final evidence.

The second text revision also protects analytic and derived launcher identities.
PreparedProviderIdentity borrows a static literal with explicit UTF-8 length when
NUL is present; trusted_extension receives an owned exact-parameter string.
There is no borrowed view of a temporary std::string. Non-NUL legacy spellings
remain unchanged. The non-author NUL launcher mismatch and the actual astral Aux
Clang refusal remain preserved; these are text/identity controls, no Native run.
