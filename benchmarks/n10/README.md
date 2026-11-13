# N10 DM-T2: time per step against b6

`dmt2.py` measures the DM-T2 gate of spec 0.3 (§10.8) for N10 phase 1 (ROM-85): M01 and M03, at
the PF10 sizes, 1 and 4 threads, each head against b6 (`b6e64c1d`) on b6's native face route, in
interleaved A B B A blocks across two worker processes, final states compared bit for bit. The gate
is 1.02 on the median time per step of every case; `dmt2.py gate` prints the table and fails above
it. See the module docstring for the protocol and the command lines.

The first campaign (08/11, integration `2dcb8ce1`) found M01 2048² at 1.19 and bisected it to #699:
a generated package compiled for its one resolved route is a small translation unit, GCC's
inlining budget (`inline-unit-growth`) ran out before the metric accessors, and each face called
them through the PLT. The ROM-18 follow-up forces those accessors inline (`POPS_HD_FORCEINLINE`).
