# Independent review of Uniform Fan–Li15 eight-step preparation

Author freeze: 02cdcb8e9169b7ecf4f1271ea2c7856d92b49513.
Independent math reader remains 0ad0dabecfce1b36084cfa31db3d735e53e9bacc.
No Native execution or scientific receipt is claimed by this preparation.

The public case retains fifteen degree-four moments (ten conservative and five
regularized terminal rows), the signed periodic FV/Rusanov path, N=16, dt=1e-4
and eight explicitly authored SSPRK2 steps. Complete explicit basis reversal is
used for the second node; density/covariance lookups first canonicalize by the
authenticated basis. There is no model-specific compiler branch. Uniform remains
the original target; full grown storage, checkpoint and restart are explicitly
unavailable and must not be inferred from the nine valid snapshots.

Each initial/accepted phase is captured and saved before the next run or guard.
The existing collective capture/run helpers preserve the original local error
and annotate persistence failures. Rejected attempts are separately labelled.
Every model is required to provide public C25 complete provenance and actual
source before bind; exact layout/program verification precedes those exports.

Two review findings on the author freeze are recorded without silently changing
its claim: Program dump_cpp has an advanced regeneration fallback and should be
preceded by the existing dump_retained_program guard; the fixture itself does
not compare Gauss4/GL48 on saved faces. The independent reader already enforces
the latter original guard, together with Wick/Gram reconstruction, SPD,
GL24/48, SSPRK2, ten inventories and final DOP853. Cross-node permutation 1e-12
is a reception operation (compare_permutation), not proved by two independent
fixture nodes each passing the looser oracle threshold.

Wire compatibility is direct: initial and accepted1..8 NPY arrays with explicit
ordered shape (15,16,16), and exact JSON [float time,int tick] clocks. Independent
frozen save_phase execution proves reverse component order, dtype/shape and
signed-zero bytes survive serialization. These synthetic wire tests are not
Native snapshots. Missing phases, Boolean ticks and x/y-transposed nonconstant
images are refused by the unchanged independent reader.

Commands: pure reader plus wire review tests with pythonpath='.' (17 PASS in
14.97s); author preparation tests excluding resolves_and_emits with its own
Source pythonpath='python .' (3 PASS in 3.75s). Both commands unset PYTHONPATH,
disable bytecode/cache, use the existing ir17 interpreter, and import no Native.
The two genuine public validate/resolve/emit Source nodes are separately replayed;
final results and any author corrective delta are recorded in the subsequent
review update. Main/Native/ENV/ROMEO and raw receipts are untouched.
