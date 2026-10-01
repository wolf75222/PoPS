# Artifact-issued AMR output history projection @2

The private capacity `pops.program.scalar-output-history-projection@2` extends
`scalar-output-field-v1` for **store-only** cell fields. It covers the exact IR16
`pops.program.global-field-history-storage@1` observation descriptor, deeper rings,
and legal multicomponent scalar-field stores. `depth` in the frozen DSO tuple is
the physical ring size (`maximum lag + 1`), not the maximum lag.

The complete frozen tuple is `(name, program_owner, state_identity,
space_identity, clock_identity, interpolation_identity, depth, components)`.
The existing installed DSO accessors supply all eight values; no accessor, wire
schema, Program IR revision or native ABI is added. The space URI is the private
**transfer capacity**, not a physical FieldSpace. `state_identity` continues to
contain the exact global observation descriptor. The descriptor's physical
source, original problem/unknown, point and clock remain distinct from the State
used to allocate storage. In particular, storing a temperature observation in an
accumulation carrier Q never authorizes copying Q into temperature.

The codegen classifier traverses inputs and recorded regions/results. A Program
history consumer keeps the ordinary history identity; output-only capacities
cannot be read through `ctx.history`. Ordinary width-one, two-slot, non-global
output profiles retain v1 and their original generated metadata/CPP. IR16 global
stores always select @2. Width and maximum lag must be exact positive integers.
The existing history authoring contracts continue to decide which field types
and widths are legal; this capacity does not introduce a new type conversion.

## Current spatial realization

AMR regrid authenticates @2 against `ProgramRuntimeState`'s checkpoint shape from
the installed artifact. It requires an artifact-backed installation, nonempty
installed identity, a unique frozen row, the exact Program-to-runtime owner map,
and equality of every tuple field. A live JSON prefix or matching live metadata
alone is insufficient. This check runs in the existing preparation/vote before
source transfers for cold, retained and removed rings alike. The context callback
rechecks the frozen row before publishing its remapped provenance; global history
stores also check it before writing a ring.

The current @2 realization projects actual parent ring slots under an aligned
`IntegralOnly` **1:1** parent/child clock relation. It transfers every slot and
component using the existing prepared regrid/Ghosting transfer. Parent samples
must be authenticated; sample windows, outgoing durations, initialization,
pending flag, fill count, owners and descriptor fields are checked. A retained
child must carry identical metadata (durations are compared bit-exactly).
Uninitialized slots remain uninitialized and transfer their actual registered
zero-start images; no publication/sample/fill count is synthesized. Initialized
aligned samples retain their actual IDs; this is not temporal interpolation or
CopyCurrent initialization of an earned lag.

A missing, unequal or nonintegral clock relation is an explicit @2 protocol
refusal, including a cold or retained ring. This qualifies this spatial
projection realization only. It does not claim that all global history storage
supports arbitrary temporal subcycling. A future time-sampling/remap protocol
must authenticate different-clock sample windows separately. Ordinary state and
AB2 history paths retain their existing contracts and 1:1/2:1 realization.

Allocation, preparation and exact-contract consensus use the existing collective
error votes and detached history candidate. Transfer failures are converged
before publication. The surrounding accepted AMR transaction remains rollback
authority; no new mini-runtime or ring publication protocol is introduced.

## Validation and limits

The author source tests add deeper output rings (maximum lags 2, 4, 7), an actual
public IR16 observation allocated on a distinct five-component State, and the
actual C++ frozen-tuple consumer with all eight field mutations, foreign runtime
owner, missing artifact identity, duplicate frozen row and changed owner map.
The C++ fixture's trusted-loader image setup tests the consumer, not DSO loading
or a numerical AMR solve. Those native tests remain pending central execution.

Local source checks use the real Python classifier with an explicitly substituted
descriptor lookup; they qualify its decision path only, not original issuance or
public Case/codegen execution. A single syntax-only Dim2 translation unit includes
the real `src/runtime/amr/amr_system.cpp` and instantiates the real
`AmrProgramContext<2>`, with private source headers first and root-authorized
NEWENV dependency headers read-only. It passed exit 0 without warnings; its exact argv, source before/after hashes,
TU hash (`da0cec0082b348735cb275d256f11225ff69b86f71ccc64de6b746ad8f0dac9f`) and result are retained under `outputs/amr-history-capability/`.
No DSO, MPI numerical execution, installation or scientific positive archive is
produced here. The two actual four-case native refusal campaigns are preserved;
root will rebuild and receive the installed IR16 AMR witnesses independently.

Reproduce the source selections after root authorizes the source/SDK association:

```sh
env -u PYTHONPATH PYTHONPATH="$PWD/python" /path/to/root-owned/python -m pytest -q \
  tests/python/unit/codegen/test_scalar_output_history_identity.py \
  tests/python/unit/time/test_global_history_issuance.py
```

This selection was prepared but not executed by the author against the shared
installation. The local classifier checks were nine SOURCE_ONLY host decisions,
with `global_history_storage.descriptor` replaced by a named lookup adapter. The
C++ test target `AmrOutputHistoryCapability.*` and the retained v1 AMR expansion
tests must be executed after root rebuilds. No local result substitutes for them.
