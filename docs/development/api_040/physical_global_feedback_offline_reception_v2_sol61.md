# T5 physical global source: independent offline receipt @2

This is a distinct reception of the SDK375f physical_global=True witness. The
historical SDK7b ProgramFalse reception, its source contract, and its oracle are
unchanged. The new wrapper imports neither PoPS nor a native module; it does not
execute or rebuild the simulation. It receives actual ROOT-sealed files only.

## External authority and results

ROOT supplied these immutable files and hashes, independently of this checker:

| Campaign | ROOT inputs | External SHA256 |
|---|---|---|
| Serial | `outputs/physical-global-serial-root-owner-20261001/root-inputs-associated.json` | `8a38ae1f13f6fe90027c3fa488fe155b81b93a4226d32aeda7499bd6d41d2efe` |
| MPI2 | `outputs/physical-global-mpi2-root-owner-20261001/root-inputs-associated.json` | `383252dd21ca867173de787e3084867251b78ebd4b4d0a6d8fca47b472d8080e` |

Each contains a separate externally pinned mathematical inventory: Serial
`8f4aeca8ad6f37d95aeeacea5a6ff14d516a9c38a8745761f9dc5e60f126b753`, MPI2
`d8d7dd753fb514baa7878d5758a645c246632b1ddcec91bf4a75ff6232e756a8`.
Both actual campaigns were received here. Each contains N8/N16 restart and N8
rejection/retry witnesses: 13 phases, 39 state/receipt/checkpoint files plus
runtime identity. The unchanged mathematical core receives the real arrays,
native all-rank wire ledgers, exact restart state, and rejected publication.
Maximum field error is `2.220446049250313e-16`; maximum physical inventory
residual is `8.304988641238964e-17`; maximum native ledger residual is
`1.951563910473908e-18` in both campaigns. The pinned native JUnit files contain
8 successful cases Serial and 11 per rank MPI2, without skips; these counts are
historical ROOT executions, not executions performed by this reader.

Runtime source is `180afdc5b294788f370e3720c71772d70805919c`, distinct from native
build source `32235b93296fe6d4337194e17d0d45569dc63a08`. Native DSO hash is
`8dcdfacab20fffd876128c2ee4b06bd26f4c952f2d7ed6ce76fb028815bb6310`; SDK hash is
`375fbdebb4b68bb43164ba3fdfa32ead3beaa8d89606925609e96507eeed3eff`.

## Source and executable evidence

The wrapper receives a closed ROOT input schema, 20 unique additional leaves,
and three explicit case associations. It authenticates five actual source
leaves against Git180: public fixture, capture unit fixture, physical global
source adapter, model kernel emitter, and ProgramContext. Four identity/compiled
artifact source blobs are authenticated against the pinned production catalog
and Git180. The 1114-file production catalog is ROOT-attested and its own bytes
are checked against runtime identity; this reader does not claim independently
to reread all 1114 installed production files.

For each actual retained C++ file, the reader checks the embedded ProgramIRv8
capture and source DAG, exact point n, candidate q units and original
circuit_quantity port, integral initial value .7, the unique external x+ trace
with scale -1, and the source-rate/candidate composition. In the executable
source scope it requires:

* one `integral_candidate_value` evaluation outside patch and cell loops;
* original source `(-0.3 * physical_global_2_1) * mass`, with mass from u0;
* source balance coefficient one, with no extra q or duration factor;
* reaction candidate `u0 + dt*u3`, then the authored transport and publication.

The exact ProgramContext overload passes factor one to axpy; its `dt` argument
is coefficient-authority metadata and does not multiply the source balance
again. The resulting recurrence is reaction
`v_i = u_i*(1 - .01*.3*q_before)` followed by the unchanged original transport.
The common numerical oracle is reused explicitly for this recurrence and the
physical inventory/ledger checks. Its historical source label is not silently
reinterpreted as original physical source proof: the distinct @2 wrapper adds
the typed IR, executable C++, and source evidence.

Program and System component binaries and their actual sidecars are read from
each explicit execution association. Independent deterministic CBOR recomposes
their binary identities from actual SO bytes and component artifact identities
from typed spec and binary references. C++ semantic provenance is compared to
the actual Program sidecar. File names alone establish no association.

ROOT attested each case's isolated execution/cache association to its retained
Program and System components. The original aggregate payload was not retained.
Therefore this receipt reports `cryptographic_aggregate_binding=false` and
qualifies mathematical receipt plus authentic source/C++/component bytes plus
ROOT-attested execution association. It cannot recompose the aggregate artifact
identity from missing payload, and does not invent that payload. Retry has no
JUnit directory property; its directory remains explicitly ROOT-attested.

## Independent controls

`test_sol61_physical_global_feedback_offline.py` receives the byte-exact retained
N8 C++ sample, SHA256
`df50a7d1dd56edd3414e1f3b9360f4c4e93e81c5192d282580133a5dc376ea66`.
The sample was copied read-only from the ROOT-pinned Serial case and stored as
base64 so its original whitespace is preserved exactly; decoding must reproduce
the externally pinned digest. It is never compiled by this review. The 35
source/protocol controls pass, including extra
q, absent q, wrong source sign/state, doubled dt, foreign quantity, changed
capture point/units/port, wrong external incidence/scale, source-rate mistakes,
duplicate IR keys, altered binary/sidecar, alias paths, missing external seal,
and canonical byte-codec vectors. Opaque binary test bytes exercise identity
protocol only; they are not represented as a native artifact. These are decoder
and executable-source countermodels, not new fully resealed saved-state
campaigns. No positive scientific arrays were manufactured.

An isolated `python -I -B` subprocess with an import hook that rejects every
`pops` module name verifies that the reader and its dependencies never import
PoPS/native. The test runs in an unrelated temporary working directory, so
already imported PoPS modules in a broader pytest process cannot fake or fail
that property. Ruff and `git diff --check` also pass.

## Reproduction

From the private checkout, with WS as shown below, run these exact offline
commands. Output files are new review receipts; ROOT inputs and donors are not
rewritten or resealed.

```sh
WS=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -B tests/review/sol61_physical_global_feedback_offline.py --root-inputs "$WS/outputs/physical-global-serial-root-owner-20261001/root-inputs-associated.json" --owner-sha256 8a38ae1f13f6fe90027c3fa488fe155b81b93a4226d32aeda7499bd6d41d2efe --output outputs/physical-global-serial-reception.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -B tests/review/sol61_physical_global_feedback_offline.py --root-inputs "$WS/outputs/physical-global-mpi2-root-owner-20261001/root-inputs-associated.json" --owner-sha256 383252dd21ca867173de787e3084867251b78ebd4b4d0a6d8fca47b472d8080e --output outputs/physical-global-mpi2-reception.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_physical_global_feedback_offline.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/sol61_physical_global_feedback_offline.py tests/review/test_sol61_physical_global_feedback_offline.py
```

This witness is the actual scalar Cartesian Dim2 N8/N16 realization. It adds no
AMR, vector, Dim3, new MPI execution, or cryptographic aggregate-payload
qualification. No production/header, MAIN, installation, environment, cache,
native library, or donor receipt was modified by this review.
