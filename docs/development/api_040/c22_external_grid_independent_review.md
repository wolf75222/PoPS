# C22 external-grid independent review

Reviewed `b4e3ea2` and follow-up `8bf15fe` in the isolated
`PoPS-resource-lifetime` checkout. No native build or runtime test was run
by this reviewer.

The first patch correctly replaced the four-ulp approximation with exact
grid membership and an exact requested frontier. It checks that the interval
is finite, positive and representable as `now + dt == next`, and checks the
reached native time and macro step inside `_native_attempt` before temporal
acceptance. The new public native test retains real `bind`/`run` and makes a
rank-local wrong landing observable after a genuine native step.

Source call-graph inspection found one MPI gap in that first patch:
`RuntimeInstance._accepted_controller_step` calls `prepare_attempts` before
`_accepted_step_transaction`. A rank with an undeclared local `native.time()`
could throw in `prepare_attempts` while another rank entered `native.step`
and its collective vote. This is a source-level counterexample, not a
deadlock reproduced by a native run. The existing wrong-landing test only
covered errors after native step.

The follow-up `8bf15fe` captures local preparation errors and gathers the
exact controls, `now`, macro step, grid index, next point, `dt` and requested
frontier on the authenticated communicator before any rank returns an
attempt. All ranks refuse an error or mismatched contract before calling
`native.step`. The new rank-local wrong-entry native witness checks no step
was entered; synthetic source tests discriminate each of the seven contract
fields. The post-step landing remains under `_native_attempt`'s collective
decision and precedes temporal acceptance. I found no remaining issue in
this bounded preparation/publication window.

The author's source suite reports 87 passes. I did not run it independently.
The public MPI/native witness still needs the central rebuilt package and
its receipt. Errors in controller construction or invalid execution
resources before this preparation vote are outside this bounded patch and
must retain their upstream authority; this review does not claim a general
MPI agreement protocol for all step strategies.
