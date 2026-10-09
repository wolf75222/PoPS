# Independent review: standalone User AMR route admission

Reviewed commit: `bf4f1751b138eb71ee135ef8964c5864069e8b31` (based on
`3bd6ea9b143145dafda13e9d8758e6416e69d273`). This review made no
production edits and did not rebuild the installed package.

The installed `3bd6ea9` AMR case
`test_two_block_captures_and_rebind_match_independent_fv_oracle[2-amr1]`
reached `AmrSystem::add_native_block`, then rejected
`source_stencil:<64-hex>` as an unknown catalogue limiter before loading the
authenticated native image. The analogous `source_face:<64-hex>` route would
have encountered the same preflight call after the first refusal was removed.

The patch introduces a syntax-only host check and changes just the host
preflight call to use it. I checked all four generated
`prepare_compiled_amr_system_block` overloads: they still use the exact
compiled-policy validator and require the full requested source identity to
match the typed policy before materializing or publishing a block. The
generated AMR installer passes the authored reconstruction and face policies,
including declaration RuntimeParams, to those overloads. The source+source
path captures both policies in the actual flux callbacks; it does not enter
the catalogue selectors. Model/binary identity and the full route/parameter
contract remain in the existing collective loader preflight. I found no
additional catalogue parse or identity bypass on that path.

The two new C++ tests exercise all four builtin/source combinations,
missing/mismatched exact policies, malformed source hashes and invalid
metadata. The author's host harness ran the exact two test bodies against the
real header: **2/2 passed**. I independently checked the final commit's diff
for whitespace errors. These checks validate route admission and exact-policy
guarding; they do not establish successful AMR bind, flux, rollback, MPI or
scientific agreement. The installed two-block User AMR case must be rerun on
the centrally rebuilt native artifact. Verdict: favorable within that
boundary.
