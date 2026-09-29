# Correct the negative FixedDt witness, keep native acceptance unchanged

The installed 5d11268 reception (Dim2 native `73b3bfaa`, SDK `242e849c`)
executes nineteen tests and reports nine passes and ten failures. Eight failures
are the separate missing storage carrier for states without a physical flux.
The two path failures instead expose a wrong negative-test input: the fixture
expected a rejection at `dt=0.1`, but this authored path has speed 0.59 in x and
exactly zero in y. Its two-incident-face frequency on N=8 is 9.44. FixedDt and
the external grid use unit numerical budget, so `0.1*9.44 = 0.944` is admissible.
The older scalar CFL proposal remains conservative; it is not the authority for
the explicit consumer's evaluated directional numerical bound.

The native implementation and all acceptance thresholds are unchanged. The
negative FixedDt and guarded external-grid inputs now use `dt=0.2`, giving
budget 1.888. The guarded external-grid test keeps `dt=0.1` as a positive control
on the same artifact. The dedicated unit-budget test also uses `dt=0.1`, which
discriminates the real unit budget from an incorrectly inherited CFL=0.25.
The hot-stage test, fractional consumer, diagnostic-only consumer, independent
FV state oracle and exact rejected-state/clock assertions are unchanged.

The failed receipt is retained under the parent workspace directory
`outputs/installed-5d11268-dim2-face-products/`. These changes need installed
reception; a mathematical input check or source review is not a native pass.
