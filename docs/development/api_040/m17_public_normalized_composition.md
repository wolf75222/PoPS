# M17 explicit normalized arithmetic composition

This tranche replaces the compiler-selected Hermite recipe. It does not turn
Source or isolated host checks into a new Native, MPI, GPU or scientific receipt.
The historical generic-core audit and its baseline stdout remain historical.

## Separation of equations and realization

`pops.moments.fan_li.fan_li15_native_plan` now **constructs** the scalar flux,
Hermite coefficients, declared fifth-degree closure, regularization one-form,
and spectral majorant in Python. Its output is an arithmetic DAG, not an order,
closure selector and list of regularized rows interpreted by the compiler.
The five products and their factorials are visible in this library. The
compiler has no Hermite coefficient, edge-closure, regularization, Gaussian
formula, physical component name or Fan–Li branch.

`NormalizedPathInputs` exposes exact normalized monomial coordinates and raw
endpoint coordinates, density, direction and the certified second-moment norm.
`Arithmetic` expresses ordered addition/product/integer power, literal division,
scalar FMA, compensated sum, absolute value, square root and polynomial
derivation. These are common arithmetic operations. The nodes have explicit
scalar/polynomial types; references must be exact integers into preceding nodes.
Native polynomial multiplication, certified recovery, density weighting and
compensated integration are reused. Python runs only during authoring/codegen.

Contract `normalized_polynomial_path@1` authenticates the complete basis, all
outputs, graph type/degree, factors and integration mode. Polynomial capacity
is the maximum degree of **all intermediate polynomial nodes**, distinct from
the moment order. A degree-two basis with the one-form `u**6 * derivative(u)`
uses `Polynomial<6>` and seven density weights; no coefficients are truncated.
The native coefficient/exponent index width is checked, rather than imposing
a physical order ceiling. Legacy C++ `integrate_normalized_moment_path<Order>`
retains its default `Degree=Order`; Fan–Li still uses degree four/five weights.

The second mode, `analytic_endpoint`, compiles a scalar endpoint formula from
qualified `EndpointPathInputs.left/right` coordinates. This is required by the
distinct law `B_g=(g_x-g_y/2) rho I` on a **raw** straight path:

```
integral[k] = (g_x-g_y/2) * (rho_left+rho_right)/2 * (U_right[k]-U_left[k])
```

That law is not converted into `rho dW`. The formula, its physical product and
its whole-path bound remain the author's mathematical obligations, just as the
whole-path speed of `SymbolicPath` is an author's obligation. The core checks
the retained physical expressions/owners and the complete typed arithmetic;
it does not claim to prove a symbolic equivalence theorem.

## Public binding and method choice

`CartesianMonomialBasis` is the shared immutable M16/M17 basis. Storage slots
bind to its explicit multi-index order, independently of component spelling.
`NormalizedPolynomialPath` binds the graph to an exact declared flux/product.
It retains the qualified state separately and relays it to the selected native
state by exact handle/index substitution, preserving rejection of foreign
quantities. Its graph is detached/deeply immutable, and native lowering receives
a plain detached snapshot.

`pops.moments.fan_li15_path(product, frame=..., covectors=..., basis=...)` is the
public library composition. The old `FanLi15RawMomentPath` import remains a
compatibility facade into the scientific library; its old component-name
contract is not a compiler restriction. Its descriptor is version two and
includes the authored arithmetic in its identity. The new public descriptor is
version one. No runtime class layout, accepted state/clock contract, checkpoint
codec or numerical wire representation changes. Changed SDK templates and
Python/compiler identities still require Root's genuine rebuilt artifacts.

The linear M17 script preserves its historical `SymbolicPath/Gauss4` default.
`POPS_API040_M17_PATH=normalized-analytic` explicitly selects the new Python
composition; `POPS_API040_M17_ORDER=reverse` preserves its permutation variant.
The equations, N=16, dt=1e-4, eight SSPRK2 steps and all scientific criteria are
unchanged. Old receipts do not qualify this separately selected realization.

## FP and scope

FMA covariance differences and ordered compensated scalar central sums are
explicit. Polynomial operations and density-logarithm integration retain their
existing implementation. Numerical canonical orientation uses the declared
basis binding, so a physical storage permutation cannot change density ordering
or floating evaluation orientation. Flux and analytic-integral outputs are
checked in full before any result component is published.

The current certified normalized covariance realization has two velocity
coordinates and a complete total-degree basis of order at least two. This is
the existing mathematical realization's support, not an inferred model order
or formula. SPD certification is not a full moment cone certificate. The new
host checks do not qualify production kernels or device performance. Code size,
register/storage cost and comparable backend timing still require Root's
measurements; the explicit DAG is compact in identity but expands into native
arithmetic statements.

## Reproduction and gate

From this private worktree, use the existing `pops` interpreter with
`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1`, pytest `--noconftest`,
`-p no:cacheprovider -o pythonpath=python`. The affected cohort is:

```
tests/python/unit/codegen/test_fan_li15_path_program.py
tests/python/unit/numerics/test_fan_li15_path_contract.py
tests/python/unit/moments/test_fan_li15_constitutive.py
tests/python/unit/numerics/test_symbolic_path.py
tests/review/test_m17_normalized_path_composition.py
tests/review/test_generic_core_composition_source_only.py
```

The host witness compiles actual emitted kernels and SDK arithmetic headers
with C++20, strict FP and no PoPS DSO. It checks storage-permutation equality,
historical arithmetic comparison and the degree-two/degree-six integral. The
degree-six reference is independently `integral_0^1 u^6 du=1/7`; its FP bound
comes from the seven exact integer binomial coefficients, their weighted
absolute sum `127/7`, and `gamma_42` for divisions/products/compensated reduction.
Malformed references, operation/type/degree claims, capacities, output supports
and endpoint qualification are refused. The old plan that selected a closure
without expressions is now refused by the independent Source-only witness.

Non-author structural composition, final Source results and Root's rebuilt
Native/scientific/performance evidence remain separate gates.

## Contrôle strict du graphe et des entrées de vitesse

La borne de vitesse reçoit explicitement le même état brut que la récupération normalisée, avec la liaison exacte des composantes. Le noyau commun conserve le protocole historique à deux arguments pour les lois existantes. Aucun nœud non atteignable depuis les sorties déclarées n'est accepté : une opération inconnue, cyclique ou falsifiée ne peut donc être cachée dans une capacité supplémentaire.

Delta Source/host : 12 tests passent en 2.93 s ; les neuf adversaires indépendants de Galileo passent en 0.72 s, dont la compilation réelle de la vitesse dépendant d'un moment brut. La précédente cohorte au gel7541 est incomplète : ENOSPC aux probes finales et à l'écriture XML. Elle ne constitue pas un reçu positif. La cohorte finale complète est relancée sur ce correctif. Aucun résultat Native, GPU, MPI ou performance n'est déduit de ces contrôles.

## Gel Source final

Le code au gel `45e746abff861781cd78e881d3572d8952afe742` ferme la cohorte complète : **39 PASS en 355.84 s**, sans échec ni skip. XML externe `/tmp/sol61-m17-final-source.xml`, SHA256 `6974f62223e6672eaa1704ee405e506e965d2c79eeabd6489ae9fdcdc52ad6eb`. Le reçu JSON compagnon lie les fichiers de production réellement inspectés. Un import frontal depuis le chemin absolu du WT confirme zéro extension `_pops` chargée.

Commande exécutée depuis ce WT :

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/python/unit/codegen/test_fan_li15_path_program.py tests/python/unit/numerics/test_fan_li15_path_contract.py tests/python/unit/moments/test_fan_li15_constitutive.py tests/python/unit/numerics/test_symbolic_path.py tests/review/test_m17_normalized_path_composition.py tests/review/test_generic_core_composition_source_only.py -q --tb=short --junitxml=/tmp/sol61-m17-final-source.xml
```

Contre-revue indépendante Galileo du delta : dix tests Source/host passent en 1.55 s ; pins SHA256 `8832efba522c3da3e987efc03d022e4c3a4cbafa79cd965fd0ddce62d0c59161`. Cela comprend une densité brute au slot4 et le protocole historique à deux arguments. La compilation hôte de code réellement émis ne qualifie aucune DSO PoPS, exécution MPI/GPU, stabilité globale ou performance.

Le corpus original a effectivement été lu dans la copie authentifiée fournie par ROOT : `context/CORPUS_ORIGINAL.md`135–174 et `reference/PoPS_API_v0.4.0/sources/mathematical_original.tex`547–551. La composition conserve dix équations de degrés0–3 conservatives et cinq terminales non conservatives ; elle ne transforme pas le produit différentiel en source. Le port arithmétique accepte une base2 et une capacité6 indépendante ; la bibliothèque FanLi15 conserve son modèle degré4. Les nouvelles réceptions scientifiques, cas non-auteur et mesures FP/performance Native appartiennent à ROOT.
