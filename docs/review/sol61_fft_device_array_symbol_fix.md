# Device-callable parameter snapshots for the FFT symbol

Actual CUDA diagnostic job 733030 compiled and ran both configurations on ARM.
In each configuration all 18 input-copy/forward stages passed; all 48 remaining
stages failed. The independent array-capture control read zero for every cell
extent and spacing. The symbol output was byte-identical to its last forward
input for all six shapes; Engine solutions reproduced RHS to roundoff, and the
constant zero mode was not eliminated. Exported references agree with independent
NumPy. These are observed values, not a cause inferred solely from warnings.

The affected device closures call `std::array::operator[]`, declared host-only
by the supported NVCC toolchain. The symbol now copies the host arrays into owning
`Kokkos::Array` values before launching. Its accessors are device-callable; no
pointer to host storage is captured, no allocation is introduced, and the plan's
interface, fields, collective stages and lifetime remain unchanged.

The discrete symbol's operation tokens and order are preserved after substituting
the captured array names. Transform kernels, FP flags, `1e-14` zero cutoff and
`1e-11` numerical oracle remain unchanged. The diagnostic array snapshot uses the
same owning conversion. A separate control binary keeps the original standard
array capture alongside Kokkos array and scalar captures. The standard capture
is an observed negative control; both corrected controls require exact values.
All Engine and direct-pipeline comparisons retain their original strict oracle.

Source checks pass 5/5. Actual Host compilation/execution passes both 66-stage
configurations; their four complete IEEE/JSONL exports equal the preserved
pre-correction Host bytes. All six Host capture-control rows agree exactly across
the three representations. These checks support Host behavior and arithmetic
preservation; they do not establish CUDA acceptance.

The separate direct pipeline and capture controls are not internal Engine traces.
The exact compiler lowering responsible for the observed no-op remains unproven.
The next actual CUDA comparison must retain the original header/result and
compare original/corrected Engine runs plus the three parameter controls with
unchanged compiler flags. No installed Native PoPS, distributed FFT, MPI or
physical-model qualification follows from this preparation.
