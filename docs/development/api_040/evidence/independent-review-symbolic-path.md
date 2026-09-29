# Independent review: SymbolicPath (GPT-6 Sol)

Scope: source review of `symbolic_path.py`, `module_emit_symbolic_path.py`, `nonconservative_lowering.py`, `module_emit_path.py`, `nonconservative.py`, the path interface in `include/pops/numerics/fv/path_flux.hpp`, and the existing source/runtime tests. No production code or shared build/environment was changed by this reviewer.

## Confirmed finding, initial API

The first `SymbolicPath(product, integral=..., speed=...)` API allowed an integral callback to return zero without ever calling the supplied physical `action`. The resulting path erased a nonzero declared matrix `B`, passed authoring validation, and was eligible for native emission. The first independent test reproduced this as `Failed: DID NOT RAISE ValueError`. Root replaced that API with `quadrature=((node, weight), ...), bend=...`; the kernel now constructs every term from the retained matrix `B(Ψ)` and its differentiated tangent. An `integral=` callback is rejected with `TypeError`.

The authored `speed` expression is a mathematical obligation to bound `DF+B` over the entire path. Nonnegative, finite runtime checks cannot prove the inequality. It should remain clearly described as an author assertion and be tested with independent oracles for each model.

A subsequent review found that an additive bend independent of `right-left` could produce a nonzero loop integral for `left==right`, violating constant-state preservation. Root changed the path to `Psi=L+s(R-L)+s(1-s) K(L,R,s,axis)(R-L)`. The independent curved-path test now checks the zero-jump limit directly.

## Independent positive probe

`tests/python/unit/numerics/test_symbolic_path_adversarial.py` declares a three-component triangular model with distinct x/y matrices and a separate analytic oracle. Its x path transfers `1.7 beta d(alpha)`, `-0.4 gamma d(gamma)`, and `0.3 alpha d(beta)`; its y path transfers `2.1 alpha d(alpha)`. The straight-path test evaluates both vectors for an asymmetric jump, resolves the case, and checks generated native C++ contains three path coordinates and both direction branches. This is evidence of generic source lowering, not native execution.

The curved-path test uses the jump-linear bend matrix `K[beta,alpha]=2`, giving `beta(s)=3+4s+6s(1-s)` for the probe from `(2,3,4)` to `(5,7,9)`. The closed-form oracle for `1.7 beta d(alpha)` is `1.7*3*(3+2+1)=30.6`. The independent oracle for `0.3 alpha d(beta)` is `0.3*(3.5*4-3)=3.3`, exercising the differentiated bend rather than merely the curved state value. Simpson nodes integrate these polynomials exactly. The same test checks all path integrals vanish for `left==right`; the bend is proportional to the jump. Free Var with the same printed name but different identity, an obsolete free integral callback, and five malformed quadrature rules are rejected.

## Pending verification

Checkout-source command: `env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python3 -m pytest -q -o pythonpath=python tests/python/unit/numerics/test_symbolic_path.py tests/python/unit/numerics/test_symbolic_path_adversarial.py` returned 15 passed after the jump-linear bend and malformed-matrix checks. These are source and C++ generation checks, not native runtime checks.

`tests/python/integration/runtime/test_symbolic_path_param_runtime.py` is prepared for the native lane: one artifact, two binds with owner-qualified `RuntimeParam` used only by `speed`, independent NumPy Rusanov/DLM oracle, and an assertion that changing the bind value changes dissipation. Checkout-source `pops.validate` and `pops.resolve` passed. Native execution waits for the root-owned `pops-api040` rebuild and installed-module identity check; generated C++ alone cannot establish runtime behavior.

`tests/python/integration/runtime/test_symbolic_path_adversarial_runtime.py` adds native straight and jump-linear-curved variants of the independent three-component model. Its oracle computes the conservative flux and separate DLM side terms directly from arrays; the curved formulas include the path-state correction `Delta(alpha)/3` and tangent correction `-Delta(alpha)^2/3`. The root-owned installed runner will execute it after environment authentication.

An additional test was created **after the path-core freeze at commit `75aa789`**, outside the implementation design loop: `tests/python/integration/runtime/test_symbolic_path_four_state_runtime.py`. It uses four state components in two permutations, active x and y fluxes, noncommuting state-dependent x/y matrices, a conservative fourth row, and s-dependent jump-linear bend matrices. Its authored Gauss-4 rule integrates polynomial path actions of degree at most five; a separate Gauss-12 NumPy oracle computes both directional Rusanov fluxes and DLM side terms. With `M=max(|L_i|,|R_i|)` and `D=max|R_i-L_i|`, each path component obeys `|Psi_i|<=M+D/8`, giving whole-path infinity row-sum bounds `.25+.4*(M+D/8)` in x and `.3+.35*(M+D/8)` in y. Checkout-source validate→resolve→C++ emission passed for both permutations (generated strings 64,552 and 60,764 characters, four path coordinates and y branch present). Native qualification awaits the root-owned installed runner and RAM scheduling; this witness is one-step 8×8 evidence, not a full scientific corpus.
