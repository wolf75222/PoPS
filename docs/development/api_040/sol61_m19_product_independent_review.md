# M19 independent finite product contract - 30 September 2026

Scope: source/math reception at `9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`,
private branch `codex/api040-sol61-m19-product-review`. Production, installed
Python, native SDK and root artifacts were not edited. No build, JIT, install,
MPI or native execution was performed.

The M19 entry in `corpus.json` specifies a product support, velocity reduction,
spatial field and lift, explicitly excluding a single-domain array shortcut.
Its initial 4×3 → 4 → 4×3 witness is a first reception size, not an API extent
restriction. `mission_gap_audit_a458113.md` records existing generic support maps
and the separate open Vlasov/BGK campaign. Hypatia confirmed that the current
route is `PhysicalSupportMap` → `Program.map`/`require_mapping` → native
transfers/sessions. This review adds no competing realization.

## Mathematical contract

For each declared vector component c and retained physical coordinate x:

`R_w f[c,x] = Σ_v w[v] f[c,x,v]`, and `L g[c,x,v] = g[c,x]`.

More than one eliminated coordinate uses the tensor product of its explicitly
declared quadratures; retained coordinates may be reordered. Component names,
coordinate identities, physical domain identities and native axis embeddings
are distinct. The lift preserves every component and every retained value.
It is independently authored constant extension. In particular,
`R_w L = (Σ_v w[v]) I`; it is not an inverse. With signed zero-sum weights this
composition is zero even though the lifted field is nonzero.

Weights include physical measure and any authored moment factor. Nonuniform,
signed and zero moment weights are meaningful; they must not be confused with
the positive measure defining an inner product. For positive dx/dv and the
corresponding measure reduction, `<f,Lg>_(dx⊗dv) = <R_dv f,g>_dx`. Replacing dv
by a signed moment changes this adjoint equation. Tensor mass equals retained
mass under the declared measure. Dropping weights, permuting components alone,
or dropping peer contributions changes that original functional.

`sol61_m19_product_oracle.py` evaluates explicit Cartesian coordinates and uses
`Fraction` for products and sums of the input binary64 values, then rounds once.
It does not import PoPS, author oracles, emitted operators, or use an opaque
matrix inverse. Its arrays are mathematical test inputs, never synthetic native
saved-state evidence. Exact rational weights and binary64 weight admission are
tested separately; no bitwise native sum accuracy or reproducibility is claimed.

## Ownership and distribution obligations

Each physical source cell must contribute once across patches/ranks. A velocity
fibre can span owners; a rank-local partial reduction cannot replace its global
sum. Destination owners can differ from source owners. An empty source rank can
own destination cells. Replicas require every participant to validate its copy,
agreement of copies, and one physical contribution rather than rank multiplicity.
The same obligations apply to the spatial field lifted back into independently
owned product cells, including vector-valued fields.

The independent record oracle checks complete source and destination owner
tables, duplicates, missing cells, foreign ranks/coordinates, physical domains,
subject, point, generation, ordered components, units and nonfinite values before
constructing its immutable publication result. Lift requires unchanged retained
domains, point, generation, components and units. These records model consistency
obligations. They do **not** mint or authenticate PoPS Case/registry authority;
real binding must still validate exact owned declarations and prepared resources.
The oracle retains complete inputs to establish mathematics; it is not a proposal
to gather runtime fields or flatten the product into one native domain.

The Python descriptor probes independently construct signed quadratures with
permuted storage embeddings and widths 1/3/5. They receive reduction ABI 2,
extension ABI 3 and `support-reduction@2`/`support-extension@2`; extension declares
`inverse_closure=False`. Reduction multiplies each component dimension by the
quadrature dimension; constant extension preserves units. Wrong support domain,
width, missing units, unchanged reduction units, sampling and representation are
refused. Descriptor ports are deliberately small structural test doubles; these
checks do not establish native port ownership.

## Results and reproduction

Run from the private checkout with
`PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`:

```sh
rtk proxy env -u PYTHONPATH "$PY" -m pytest -q tests/review/test_sol61_m19_product_oracle.py
rtk proxy env PYTHONPATH=python "$PY" -m pytest -q tests/review/test_sol61_m19_source_contract.py
rtk proxy "$PY" -m ruff check tests/review/sol61_m19_product_oracle.py tests/review/test_sol61_m19_product_oracle.py tests/review/test_sol61_m19_source_contract.py
rtk git diff --check
```

Observed: **58 autonomous math/record checks PASS**, **15 Python descriptor
checks PASS**, Ruff PASS. Source import identity was explicitly
`PoPS-sol61-m19-product-review/python/pops/__init__.py` under this checkout.
Shapes include 2×5, 7×3, 3×2×4 and 5×3×2; vector widths are 1/3/5; source
partitions split x, v or both over 1/2/3 ranks; lifts include 4 ranks with an empty
source rank. Tests vary axis order, retained order, two eliminated axes,
nonuniform signed/zero weights, components and destination ownership. They also
reject 25 malformed owner/authority/shape/admission cases. No production defect
is demonstrated by this bounded descriptor review.

## Native and scientific acceptance still required

Root owns the compiled/native reception. A public witness must actually compile,
bind and execute the existing generic chain with separate product/retained
supports and subjects, exact embeddings and declared layouts. Saved fields must
show each original component at reduction, spatial-field use and lift, including
permutation and a second size. Serial/MPI multi-patch cases must exercise fibres
split across ranks, different destination owners, empty contributors and replicas;
all-rank failures must preserve accepted states, time and step. Real reduction
accuracy must be assessed against original quadrature and numeric admission.
AMR reception additionally needs active-owned coverage and hierarchy authority.

This finite map contract leaves the physical field equation, its forcing and
boundary conditions independently declared. It does not supply a Poisson solve,
BGK relaxation, Vlasov transport, Landau damping, Nx/Nv convergence or a completed
M19 scientific campaign. No historical native map result is reassigned to this
review's SHA.
