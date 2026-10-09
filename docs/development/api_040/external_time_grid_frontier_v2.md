# C22 ExternalTimeGrid computed binary64 frontier, version 2

`ExternalTimeGrid("grid")` retains the exact version 1 contract. The opt-in
`ExternalTimeGrid("grid", frontier="computed", endpoint_ulps=1)` permits a reached
endpoint within the explicitly authored number of binary64 neighbours of the
requested grid point. The count is an integer in [0,1024], not a hidden epsilon.
The controller computes `duration = requested - native_entry` and
`reached = native_entry + duration`; it never resets or relabels the native clock.
The version 2 strategy participates in artifact/run/restart identity. Preflight
collectively compares that policy and the exact reached endpoint before stepping.

An accepted controller receipt retains start, requested, reached, numerical
duration and grid index as canonical hexadecimal coordinates. Strict restart
checks the addition/subtraction, grid membership, policy bound and actual clock.
Continuation advances the next requested grid coordinate and stops on the last
requested coordinate while reporting the genuine reached native time. A distinct
future requested grid point may not cross `t_end`, even when it is a neighbour.
Numerical stages and native exchange measures use the actual duration argument;
the shared cadence continuation version 2 agreement also includes its effective
duration. Existing RuntimeInstance rollback owns all provisional receipt changes.

This is a bounded scheduler rule, not a Program Scalar-expression endpoint or a
relaxation RK acceptance proof. General C22 and T5 remain open in this tranche.
Source unit evidence: 30 frontier cases pass with selected installed Dim2 native
extension and Python sources from this checkout; the existing 83-case exact grid
and step strategy selection passed before adding the six version 2 cases. Native
rebuilt reception remains the integration owner's obligation.
