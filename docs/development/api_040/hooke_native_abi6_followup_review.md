# Independent ABI6 follow-up review

Actual GPT-6.1 Sol reviewer Hooke; frozen root fix76adc71c, separate docs-only review WT. This supplements e0d8311f without rewriting its bounded receipt. Expanded ROOT Source170 batch exposed two failures: native module kAbiVersion remained5 and architecture expectation remained5 despite generated ABI6. The existing C++ static_assert would also have rejected a real native build. My earlier bounded review missed this consumer drift; the corrected source is now independently checked.

Exact fix: module_capabilities.hpp40 changes kAbiVersion5→6; architecture test_release_contract.py95 changes expected native_abi_version5→6. The static_assert at41 comparing generated kReleaseNativeAbiVersion is unchanged. Release preflight's regex/version check at132–134 remains strict. Manifest/capability factories and runtime report use kAbiVersion at90/134/417; init_core.cpp708 exposes that same constant as __abi_version__. Generated JSON/Python/C++ all specify6. No acceptance check is weakened and older binaries are not masked by a compatibility fallback.

Scoped searches of runtime headers, native selector files, core bindings, runtime Python and architecture tests found no remaining native module ABI5 check. _bricks_scheme.py415 and descriptors.py16 still intentionally use5 for a separately named pops.external-riemann/v5 contract; that is not the release/module ABI and must not be blindly changed.

Actually executed3 architecture Source tests in0.57s: static release preflight, exact version/matrix, and duration-catalog preflight ordering. Explicit checkout/python import path, no _pops loaded. Static preflight has no --metadata-dir option in this frozen version; used supported test harness/default static path. No native/JIT/C++ build or environment mutation. ROOT owns expanded full suite rerun and rebuilt Native01c5b654 qualification; this receipt does not invent their status.

Reproduction: env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python, sys.path.insert checkout/python, pytest nodes test_generated_release_contract_is_current_and_preflight_passes_static_checks, test_release_contract_versions_every_protocol_and_declares_exact_matrix, test_final_and_release_preflights_verify_cpp_duration_catalogs_before_build in tests/python/architecture/test_release_contract.py; assert _pops absent. Result3passed.

SHA256:

```text
f30115aea0d7774fa2526c63145ce8eff96ae9d435fbe8c79617a80891d37dea  include/pops/runtime/module_capabilities.hpp
03b18b13ca7702f3a38f8e8c23ed7a13560dcc976bff7b73e42aa5ed1182fa8a  include/pops/runtime/config/generated_release_contract.hpp
7eee4390dac4c9d8c81d84de9663b8c8c9967dd465f69745de333e97c48a9acc  python/pops/_generated_release_contract.py
1ae0da07a6a8d611f6c011f1c37081195a804947b9a6390b8c0e2932a9a1c35f  schemas/release_contract.v2.json
51196b4bc06affc1c41a0c3371671b6f64fc2f9e869e1b58e13b1888f7feec59  tests/python/architecture/test_release_contract.py
4c643f72cf9e58a48a25353d656afd2a47a7aa1fbc4082e38c8041730b606dbd  scripts/release_preflight.py
```
