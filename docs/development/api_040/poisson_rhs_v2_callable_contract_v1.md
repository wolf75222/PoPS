# Provider-aware Poisson RHS callable: portability contract 1

The generic `make_poisson_rhs_v2(Model)` adapter materializes a Field RHS from
separately authenticated State and readonly auxiliary inputs. Its original
anonymous host closure contained an extended CUDA device lambda. Job 736186
failed during NVCC compilation because the enclosing factory had a deduced
return type; that job produced no qualified native Python module or GPU solve.

Contract 1 names both callables. `make_poisson_rhs_v2` returns
`PoissonRhsV2Callable<Model>` explicitly. The callable owns its Model by value,
accepts a moved factory parameter and exposes a const call operator returning
`void`. `generated_system_detail::PublishPoissonRhsV2<Dim>` carries the writable
target and readonly candidate views and performs `target(index) += candidate(index)`
with a const device call operator returning `void`.

The existing preflight votes, allocation handling, selected execution space,
valid-cell boxes, component-zero publication, fences, collective error handling
and view-owner lifetimes remain in place. The sole body substitution is the
anonymous publication lambda becoming the named publication functor. The entire
host body is therefore not byte-identical. Source review establishes no measured
floating-point, binary or performance equivalence.

No model formula, model name, provider-count specialization or new numerical
policy selects this path. Provider arity retains the existing generic declared
count and fallback zero. The adapter adds no Python per-cell execution.

The public factory name and call arguments remain available. Its concrete C++
return type changes. The version marker is
`kPoissonRhsV2CallableContractVersion = 1`. Native ABI 13, native system-package
schema 8 and Field-input/read contract 2 retain their numeric versions. The
signed SDK changes from `967e6afd549028368716cf484593d8424775d415daacba937e03822565da93e8`
to `af8d3a678a42f9d28abedc386bad5af8d987eb694a9042439414de1ac9c278e1`.
The existing header-signature guard consequently requires rebuilding and
relinking the actual Native and affected generated, fixture and component DSOs.
An unchanged numeric ABI does not authorize loading older SDK artifacts.

Source admission is recorded by the immutable patch
`c3f73502238474ee90b92ff04e13cc92a16344dcd48b5fd93a8b6df0e04ea09d`,
the independent review `f0d3fa70bb9143b06b3b111888b0d9e3742ecc6dfbf5b55dd981d071e1b6f45e`,
and Root admission `3bfc59ee663246b608a3a9d0de9727e5d791d4bf1569b4321eca3903c5116b39`.
The recorded Host syntax probe checks actual Count-0 and Count-2 model interfaces
and callable bodies; it does not establish NVCC compilation or device execution.

## Actual Source integration - 2026-10-08

The exact header is now integrated in Main
`637c3220da44102ff7dbe3468c99086d141f6981`, SHA
`ec50b10fdf582f51515fce07fb622dde32a13edba0009102ae29f8c0d5817231`.
The current383-header Source signature is
`af8d3a678a42f9d28abedc386bad5af8d987eb694a9042439414de1ac9c278e1`.
Coref3 and test3e98 remain unchanged. The earlier private carrier-bindings@1
integration initially preserved SDK967; this public callable@1 change is separate.

The CPP17/51 and first STATE world1 receipts were executed on Source2cb/SDK967.
NativeD154, binarye625 and fixture DSOs967 are historical artifacts: no af8
compilation, import, Native/DSO/scientific non-regression or CUDA/GPU qualification
is received. Recorded Host syntax and Source counter-review remain their own scope.
Root's exact-wheel/install/prove/codeSign/verify/doctor Nativeflowv2 is prepared,
NOT launched, pending its final documentation HEAD/GO. The source-admission receipt
3bfc remains unchanged and dated; its former Main_integrated=false is not rewritten.
Later runtime reception must authenticate actual source/head, signed SDK, module,
DSOs, backend and original scientific oracles after real rebuild/relink.
