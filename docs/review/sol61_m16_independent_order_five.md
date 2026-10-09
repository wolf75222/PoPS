# Independent M16 review - Source / actual-header CPU only

Reviewed author 251c51099696beac9393078ba1c519e2acd6755d and basis 78d9624,
against base 3f5a5502, materialized on the private cubature preparation f9685154.
No production files were altered by this review. No Native, MPI, GPU, runtime
package, scientific reception or global mission completion is asserted.

The public descriptor proves completeness by distinct nonnegative indices and
simplex cardinality, retains supplied ordering, and refuses mutation. Python
maps every monomial bijectively to the actual State components. The compiler
requires the descriptor and authenticates complete names, dimension and order;
it gathers/scatters to its mathematical ordering rather than guessing a species
or slot. Compatibility authoring now explicitly creates the descriptor and uses
IR23; historical affine IR bytes are **not** claimed unchanged. The Cayley and
exponential arithmetic, polynomial recurrence, compensated contraction and
candidate-before-output publication are unchanged in the reviewed C++ diff.

The degree ceiling removal widens cardinality arithmetic to int64. Positive int
template degrees cannot overflow that product before the signed-int cardinality
check. The last admitted cardinality boundary (degree65534,2147450880 components)
is checked without allocating storage; degree65535 is compile-refused. This bound
is representation admission, not a practical stack/resource guarantee. Large
admitted templates may exhaust compile/runtime resources; no high-degree success
is implied. Order-five overflow is explicitly refused without changing output.

Independent case: translated, weighted three-atom measure with density6,
nonzero mean, degree-five odd/even mixed moments. A cyclic basis order and a
coprime component-name permutation put density away from slot0. Exact Fraction
particle integration validates the public polynomial push-forward, independently
of the native recurrence. Genuine public Program authoring/lowering verifies
IR23 and each canonical gather in emitted C++. This supplements rather than
replaces the author's actual-header float/double particle tests.

Measured CPU cost, one process/thread, same compiler/source/input construction,
20000 calls each, Apple clang21.0.0, C++20 -O2, binary64:

|degree|components|ns/call|checksum|
|---|---|---|---|
|2|6|177.11|127.679|
|4|15|1186.26|81.4057|
|5|21|2452.73|65.0221|

Probe retained at tests/review/sol61_m16_cost_probe.cpp. Inputs vary per iteration
and output checksum is consumed. Timings are one local sample, not a statistically
stable speedup or performance receipt; component counts differ as degree requires.
The comparable workload is the same number of affine cell calls. Degree5 is about
2.07× degree4 here. Candidate and two polynomial buffers alone contain
N+2(d+1)^2 Real entries (93 at degree5), excluding other arrays/caller/compiler
stack choices. Runtime memory was not measured.

Reproduce Source/host checks with read-only ir17 Python, PYTHONPATH unset,
--noconftest -p no:cacheprovider -o pythonpath=python. Independent+author-header+
explicit-basis cohort initially11PASS4.56s; final added cardinality rejection and
full affected-area results recorded in the handoff. Compile cost probe with
c++ -std=c++20 -O2 -Iinclude tests/review/sol61_m16_cost_probe.cpp -o /tmp/sol61-m16-cost.

Principles→decision→file→oracle→status: 1.1/1.3 explicit monomial/component binding
in public affine authoring/compiler, tested emitted slots; 1.4/1.5 same common
native affine mathematics, actual-header particle oracle; 1.6 Source identity and
_pops absence only, Native unreceived; 1.7 local measured cost above; 1.8 new
translated degree-five atomic measure plus arbitrary binding, not only renaming
fifteen Fan–Li moments. The M17 Raw B_rho preparation remains a separate unfinished
composition slice, not certified by these tests.

Final results: full affected author Source cohort45PASS67.90s on251c; independent+
author-header+cubature7PASS3.91s. Corrective author e943942c adds cardinality
preflight before compatibility names/ranges; exact delta materialized and reviewed.
Final explicit-basis+independent cohort12PASS2.66s, including huge-order no-expansion
spies and actual-header overflow/cardinality rejection. No correctness blocker
remains for this bounded Source/CPU slice. Practical large-order resource limits,
MPI/GPU/native package behavior and full scientific M16 remain unreceived.
