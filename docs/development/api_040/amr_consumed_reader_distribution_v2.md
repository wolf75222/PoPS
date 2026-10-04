# AMR consumed-field reader distribution, version 2

`sol61.m19-amr-consumed-offline@2` corrects the independent saved-data reader.
The PoPS CP12 payload and production API are unchanged. A replicated level
records an empty `int64` owner map, and each full state carrier has owner `-1`.
A partitioned level records one owner per patch, in patch order, with each
owner in the recorded communicator. Both forms remain authenticated alongside
the carrier geometry, components, complete active coverage, time and clocks.

The real SDK32 job 733094 (Source `f6c28edc`, Native `57d41201`, CPU/OpenMP/
MPICH world1) failed at its initial checkpoint read. The earlier reader
incorrectly required `len(dmap) == len(patches)` in replicated mode. The raw
checkpoint has a replicated laminate level, a replicated reservoir coarse
level, and a partitioned reservoir fine level. Its SHA-256 is
`6934c15e747a94ac2b3e934db8eb85a515794e1aa3ad998fb9bffb5212ce5a7d`.
The corrected reader reconstructs the three saved states. This is a read of
the existing initial capture; the failed job has no accepted steps or complete
phase receipt and receives no scientific or rollback acceptance.

`tests/review/test_sol61_m19_amr_distribution_authority.py` exercises both
ownership forms across different patch/rank counts and rejects malformed
types, maps, modes and ranks. The existing public declaration and mathematical
oracle tests keep their equations, tolerances and geometry unchanged. A fresh
installed run must still complete the original accepted phases, regrid and
nonfinite rollback. Complete finest coverage is this independent oracle's
scope; it is not a production restriction on AMR.
