# Independent offline reception of the original nonlinear field equations

Date: 30 September 2026. GPT-6.1 Sol. Review worktree
`PoPS-sol61-t3-saved-state-oracle`, base `45cb04ea53745f24b5df51cd60454657d59d78fc`.
No production, installed environment, SDK or author checkout is modified.

**Received against external owner pins.** Root independently recalculated 70
files and correlated 2210 source-manifest entries against Git archives of the
two historical commits. The independent worker then ran the frozen oracle with
the owner pins SHA supplied by Root, and obtained `status: received` for all six
actual saved datasets. The resulting JSON is byte-identical to Root's separate
execution. No installed PoPS code or new native solver execution is used by the
offline verifier.

External owner file:
`outputs/native-original-field-sdk7b-root-owner-pins-20260930.json`, workspace
relative, SHA
`a5496e2c2d8313918b17fe0602d580118023b113a8af67f3558e95ab2745c07c`.
Frozen actual reception JSON:
`sol61_t3_saved_state_received_20260930.json`, adjacent to this report, SHA
`8af5d25dd84a6278b8cfca07601e4cd9549fbc0e0ace973ace2358970e8450d1`.
Executing oracle SHA:
`46bbb326c5b98b3d2376a93034331d0101d67bc16ce5ad0534b6c1dc2f0353df`.

## Inputs and authority

The proposed inventory remains deliberately **unsealed**. It is not owner authority
and cannot produce a positive authenticated reception through `verify`.
Root must independently recalculate/correlate its file pins, change authority to
`root-reviewed`, and supply the resulting pins SHA externally. The oracle
requires that SHA before loading scientific inputs. It authenticates its own
executing bytes, the physical fixture, result/identity/source manifests, JUnit
and logs, every NPZ and receipt, and all three compiled component DSOs and their
manifests per dataset. A receipt's refreshed NPZ digest cannot update the
external owner pins. No PoPS module or author oracle is imported.

Inventory schema: `sol61.t3.saved-state-pins@1`. File records contain exact
absolute `path`, `sha256` and byte size. Runs contain directory, source commit,
native DSO SHA and ABI, rank count, run evidence files, and three datasets with
unknown order, NPZ/receipt pins, artifact identity and compiled component pins.
The inventory is under `outputs/sol61-t3-saved-state-independent/` in the private
review checkout and remains untracked. No owner pin or synthetic positive
saved state is created by this agent.

Root's actual run directories, under the workspace outputs, are:

- `installed-spatial-original-nonlinear-integral-feedback-dim2-sdk7b-20260930`:
  source `634cba3511fef957e65a21fcae3ab257f164b360`, one rank.
- `installed-original-nonlinear-integral-feedback-mpi2-dim2-sdk7b-20260930`:
  before identity source `45cb04ea53745f24b5df51cd60454657d59d78fc`, two ranks.

Both report native DSO SHA
`8a217a0fff5a131729a842dd222e2e08346034eb53b4f461fc17fc8cc291b563`,
header signature
`7b503163f41cc33040c296679e3b3530a9716c91ecfe4fdce6f69dca6f07dad0`,
and physical fixture SHA
`54fc68f192616aa1a93db9c0d695e2f51f61a283226c5c8ccbf2aa0784efe346`.
The source manifest file digest is
`498eb240f2b9f016b5f75dbfd71c896a481f30c5e7cc255f79505706eb284844`.
These discovered identities were received through the separate external owner
pin above. This agent did not manufacture the authority or seal its own inventory.

Every rank JUnit has eight passed tests: five T3 and three T5. Only T3 is received
here. Three successful T3 cases write actual global observations on rank zero;
the other ranks do not fabricate local NPZ duplicates. Exact dataset files are
listed in the inventory. Serial and MPI2 NPZ bytes match for each case:

| Unknown declaration order | NPZ SHA256 |
| --- | --- |
| `(0,1)` | `4121f0b99be433a14992ca569814edc3fa7a6595b0a95223c44c1daa24652e8e` |
| `(1,0)` | `181c84422453ee51d7fae8d42cd10f48128c53f53ccbc57e6646b3a96cec1e2d` |
| `(2,0,1)` | `d82efb6b228755d8a1040c7ad4fc10d320d46155859274fb287c1f409cee5f39` |

## Independent physical reconstruction

The finite grid is periodic, five physical x cells over length 1 and four y
cells over length 1.5. Saved component arrays use `(component,y,x)` storage;
two solves are observed as `(solve,component,y,x)`. History routes in the pinned
fixture store original q/v/z handles, independently of the declaration order.

The oracle integrates the declared trigonometric data by antiderivatives over
physical cells. It does not import the author's `np.sinc` target constructor.
The periodic Laplacian uses explicit one-dimensional stencil matrices on each
physical axis. It does not import or reproduce the author's `np.roll/einsum`
implementation. With captured forcing f and spatial coefficient a, all original
equations are evaluated directly:

```
Fq = a*q + q^3 + .03*q*v + .02*v^2 - .04*Δh(q) - .006*Δh(v) - fq
Fv = 1.3*v + .4*v^3 + .02*v*q + .01*q^2 + .003*Δh(q) - .05*Δh(v) - fv
Fz = 1.4*z + z^3 - .03*Δh(z) - fz                 (three-component case)
```

No inverse, local condensation, Newton solve or expected-solution generator is
used to replace actual observed native solutions. The third field is nonlinear
but diffusion-decoupled from q/v in this specific witness. The two-field
subsystem has nonsymmetric cross diffusion; this native witness does not test
arbitrary indefinite coefficient matrices. The wider generic source/math
obligations were scoped separately in review `3e962835`.

The second initialization is `.8 * actual_first_solution`. Its original
residual is 0.07294921079353184, so it is distinct and does not already solve the
equations. Both actual solutions satisfy the same captured equations. Captures
are checked against the physically declared cell averages and forcing generated
from those discrete equations. Captured fields are not replaced by seeds.

## Numerical inspection and counter-probes

Independent numerical verification of the six externally pinned actual NPZs
finds maximum state error `4.218847493575595e-15` and maximum original residual
`5.467848396278896e-15`. Maximum parameter and forcing reconstruction differences
are `1.1102230246251565e-16` and `1.6653345369377348e-16`. The original fixture's
state/residual admission remains `2e-8`; input reconstruction uses `2e-13` solely
for differing antiderivative/stencil arithmetic. Native guards are not relaxed.

Counter-equation residuals on the actual second solution are:

| Deliberate wrong interpretation | Minimum residual |
| --- | --- |
| Transposed D | 0.005817664773379971 |
| First solution used as captured parameter | 0.14336132446828306 |
| Stale forcing scaled with the seed | 0.07079834343938507 |
| Seed used as RHS | 0.1463522799285094 |
| Local-only equations, diffusion dropped | 0.026671161190325726 |

Ten independent offline tests pass: actual solutions and these contrary
equations; four route/capture/seed faults; serial/MPI exact-byte observations;
unsealed-owner refusal; external-SHA refusal; receipt order self-reseal refusal;
and an import/inverse prohibition. Deliberately tainted negative in-memory
copies are never saved or reported as actual native states. Ruff and diff checks
pass. Command:

```sh
rtk proxy env -u PYTHONPATH \
  SOL61_T3_INVENTORY=outputs/sol61-t3-saved-state-independent/inventory-unsealed.json \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_t3_saved_state_oracle.py
```

The independent worker executed:

```sh
rtk proxy env -u PYTHONPATH \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_t3_saved_state_oracle.py verify \
  --pins /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/native-original-field-sdk7b-root-owner-pins-20260930.json \
  --pins-sha256 a5496e2c2d8313918b17fe0602d580118023b113a8af67f3558e95ab2745c07c \
  --output outputs/sol61-t3-saved-state-independent/received-owner-pinned.json
```

Result: `received`, six datasets. Exact JSON and byte comparison against
Root's `outputs/native-original-field-sdk7b-root-independent-reception-20260930.json`
passes. The oracle, scientific criteria and tests remain byte-identical to
preparation commit `507982d`; only this reception report and frozen numerical
receipt are added by the final report update.

## Scope

This is offline verification of actual solutions of the declared discrete
spatial residual at one grid. The supplied forcing is manufactured from the
discrete operator, not a spatial-convergence proof for a continuum PDE. Time .01
encloses two steady spatial solves and unchanged capture commits; it is not a
qualified nonlinear evolution of M27. The original Newton/GMRES and explicit
central-FD selection are source/provenance facts from the pinned native fixture
and previous lowering review; arrays alone do not prove which algorithm ran.
The two transactional refusal cases have genuine passing native JUnit evidence,
but no dedicated before/after saved NPZs for independent offline rollback
verification. T5, checkpoint codecs, GPU, AMR nonlinear solves and scientific
M27 closure are not qualified by this lot. No new native execution occurred.
