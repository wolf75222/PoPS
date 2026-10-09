# C38: retained pending history across consecutive hierarchy publications

On production `90da6e8`, the native three-level restart test fails during the
second parent replacement. The first replacement creates an unconsumed deferred
child-1 lag at epoch E+1; replacing child 2 moves the hierarchy to E+2 while
retaining child 1. Its old marker is then rejected by the accepted-state writer,
which required the creation epoch to equal the current hierarchy epoch. The
refusal is real and must not be removed by weakening the live checkpoint guard.

The repair separates two authorities:

- `prior_*` and `published_*` remain the immutable, immediately adjacent
  topology/materialization edge that created the deferred lag.
- `qualified_*` names the current hierarchy on which that exact lag is still
  admissible. It may advance only through one engine-prepared publication at a
  time; topology and materialization distances from the origin must agree.

At creation, the marker captures the exact source `HistorySampleIdentity` and
an encoded ring contract. This contract includes logical name, Program owner,
state/space/clock/interpolation identities, depth, component count, both slot
indices, intervals, fill/initialized flags, publication identities, and accepted
clock. It deliberately excludes global topology/materialization generations.
The source identity and captured contract never change when qualification moves.
An unknown legacy sample is not converted into earned source authority.

Requalification admits only a ring at or below the new descriptor's parent
level, absent from every affected-ring decision. The old qualification must
equal the exact descriptor prior edge; its live ring contract and source sample
must still equal the captured origin. A pending store, changed clock or sample,
affected/removed/reprojected ring, stale generation, or superseding lag is a
refusal. No marker is silently erased to make a regrid succeed. The staged map
is private until collective validation and accepted-state publication succeed.
The existing outer regrid transaction owns rollback of the hierarchy and all
artifact context state if a later resource refresh fails.

The callback uses engine-prepared live ring metadata for comparison against the
captured contract. It does **not** read the facade's prior accepted image after
topology publication; that getter correctly refuses an image from another live
generation. The deferred reader checks current qualification and exact slot-1
source identity, while allowing the required fresh slot-0 store.

## Native wire and resource capacity

The native accepted-state magic changes from `POPSAND8` to `POPSAND9`. The Python
checkpoint envelope schema is unchanged. Every pending marker adds two u64
qualification words, four source-sample words and a length-prefixed binary ring
contract. The minimum record is 20 encoded words, plus key and contract bytes.
Writer and CountingWriter use the same field encoder. Capacity derives the
largest possible contract from the artifact's frozen history descriptors using
that same contract encoder, before any scientific history allocation. Repeated
counts and each variable length are overflow-checked independently.

AND8 native images are refused rather than upgraded by inventing qualification
or a captured source. Existing legacy AND4–7 inspection/decoding without pending
markers remains historical; pending markers from those formats are refused.
Test-only Python prefix inspectors explicitly distinguish the new layout from
each older layout; they do not authorize native restore.

## Evidence boundary

Three new production-helper host tests at O2 pass: two consecutive valid
qualifications preserve origin through wire round trips and exact capacity;
forged qualification/source/ring identity fails; and every rejected transition
leaves its candidate unchanged. Sixteen Python wire-inspector tests pass,
including truncated AND9 fields and preserved legacy layouts. An instantiated
`AmrProgramContext<2>::install` TU, including its actual remap/import callbacks,
passes syntax-only compilation with the matching main SDK/MPI flags.

These host/source checks do not claim a native hierarchy acceptance. The existing
three-level synthetic loader test is extended independently by GPT-6 Sol with a
second-resource-refresh failure, all-rank rejection, complete pre-regrid image
comparison and successful retry. The central build must receive that real
serial/MPI test and the original red restart case on the rebuilt artifact.
