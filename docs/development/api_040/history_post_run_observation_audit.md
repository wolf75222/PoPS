# Post-run history observation audit

Four scientific examples declare `store_history(..., depth=1)` before their
commits and read `history_global(..., 0)` after `pops.run`. The generated Program
rotates every history ring after the commit. With depth 1, the ring has two
physical slots: the store writes working slot 0, then the accepted sample moves
to slot 1 while slot 0 recycles the preceding buffer. The M06 two-step
[installed probe](evidence/m06_history_two_step_receipt.json) demonstrated this
behavior for both a varying temperature and liquid fraction on the authentic
Dim=2 native build. The four post-run readers now use slot 1.

The meaning differs by workflow:

| Example | History content | Consequence of old slot-0 read |
| --- | --- | --- |
| `euler_poisson.py` | Poisson gradient from an explicitly frozen driver State | Previous step's gradient; numerically the same for this fixed driver. |
| `field_transport.py` | Poisson gradient from a `driver` component with zero physical flux on every axis | Previous step's gradient; numerically the same for this stationary driver. |
| `variable_coefficient_field.py` | Potential from load/reference States with zero flux and explicit unchanged commits | Previous step's potential; numerically the same for this stationary problem. |
| `api040_m08_guiding_center.py` | Charge and field observations at the two SSPRK2 stages, plus same-time probe | **Actual stale diagnostic**: step 1 fields were saved with the step 2 final State. |

M08's retained original `outputs/m08-eac92bb-openmp1/states/state_32.npz`
has `time=0.04` with two `dt=0.02` steps. Its saved `stage0_q` equals the
initial charge bit-for-bit, whereas the independent one-step SSPRK2 reference
at the start of the final accepted step differs by `5.235496359987102e-6`.
The [archive audit](evidence/m08_history_stage_audit.py) and its
[read-only receipt](evidence/m08_history_stage_old_receipt.json) reproduce this
comparison without compiling PoPS; the archive SHA-256 is
`7abc44f77e22e4485b4accfbb4aee5946698620268a2d48dcd41a149753e5e1b`.
The original archive and its passing
receipt remain untouched: its final State matched the full reference, but its
field diagnostic sampled the earlier step. Internal Poisson/gradient tests
could still pass because all three observations were consistently stale.

M08 now archives the independent charge reference at the **start of its final
accepted step**, time `(STEPS-1)*DT`, and compares saved `stage0_q` against it.
This test discriminates both ring slots; it uses the already declared
`reference_fv_max_error` threshold. The example's equations, SSPRK2 calendar,
resolutions, stability checks, oracles for final State, and numerical limits
are unchanged. Full M08 native reception at both declared resolutions remains
necessary after integration. The other three examples are corrected for
semantic provenance; their frozen-field calculations do not constitute an
observed numerical failure.
