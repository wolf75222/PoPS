# Analytic endpoint return: proven Source defect

The generic `analytic_endpoint` emission in `moment_path_kernel.py` populated its integral result but fell off the end of the non-void C++ `path_integral` function. This is undefined behavior. The default integral status is overwritten by successful recovery before the identical/zero-direction branches, so those branches are not a separate status bug.

The correction adds the missing `return result` after the finite-checked integral assignments. Contract@1, physics, basis, endpoint orientation, bounds and refusal guards are unchanged; no schema extension is needed for implementing the already declared return contract.

A newly generated, distinct degree2 six-component law with density in storage slot1 is compiled with C++20, O2, no fast math, FP contraction off and `-Werror=return-type`. Before correction it genuinely failed compilation at the instantiated method. After correction it runs and checks independent analytic Raw-density integral, exact reversal, identical/zero cases and nonfinite-input rejection. Combined Source/host cohort: 15 PASS in13.63s (endpoint host, normalized-path composition, parameter phase).

Job731816's retained `program.cpp` is the Program wrapper, not the physical brick body. Its finite/SPD initial active cells and exact failed-step rollback do not by themselves localize status1031. The proven emission defect is applicable to the selected analytic endpoint route, but exclusive runtime causality requires a newly emitted artifact and actual Native run by ROOT. No Native, SDK, ENV or original evidence was modified or executed here. No ghost-fill diagnosis is inferred from initially zero grown cells.
