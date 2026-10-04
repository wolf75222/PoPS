# VP public NumPy array protocol correction

Real SDK31 job 733012 failed before bind: all three supplied arrays used mathematical axes rather than component-major reversed Native axes. Compilation completed; this job supplies no runtime, callback, checkpoint, or physics acceptance. The closed ROOT receipt is `54f329ad2c0b8c1efb4419903694cb9f2bcbab75e1f46f678ad013e99ab0c8f3`.

The Native grid remains velocity/position `(4,8)` and density `(8,1)`. Public NumPy valid arrays are respectively `(component,position,velocity)` and `(component,inactive,position)`. Fixture profile and phase envelope @2 state this representation. Mathematical references retain `(velocity,position)`; the binding and comparison boundaries transpose explicitly. Captured NPY, checkpoints, local boxes, cursors and clocks remain direct public outputs. No equation, force, Field derivative axis, SSPRK2 association, mesh, or 2e-11 tolerance changes.

The AMR consumed case already declares native cells `(3,4)` with valid arrays `(4,3)`, and destination native `(1,3)` with arrays `(3,1)`. Its different convention is already consistent; no AMR changes are justified by this failure. Composite CP9 is opaque here: no reconstructed carrier payload or Field-image claim.

`tests/review/test_sol61_vp_public_array_order.py` executes actual fixture assignments, reproduces all three parent refusals through the production initial-state checker, accepts corrected non-square arrays, and checks the actual post-step comparison expressions against the unchanged oracle. These are Source checks, not a Native execution. The previous packet math reader @1 must remain immutable; a new explicit @2 reader must accept the corrected public array shape.
