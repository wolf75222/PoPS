# Corrected Field/Ghost fixture independent review — SOURCE only

Exact8179f04ecc18b04475b2b871380e1c3078ec27e2 reviewed in separate materialized WT. Earlier dbf finding remains historical. No remaining Source blocker found for this uniform specimen.

Original -Delta(phi)+8phi-8m now has separate Linf1e-10 guard, unchanged phi absolute1e-10 and FE/ghost guards. Captured fullcarrier supplies actual m; coarse valid data and fine stored valid data are reconstructed from recorded boxes. Potential shape derives from authoritative captured mask before reshape; Cartesian fullgrid extents n/2n and fine coverage exactly ratio2 are checked. Covered parent phi is restricted from its four children; tensor quadratic parent sampling matches native composite_fac_nlevel QuadraticInterpolationTransfer, and coarse/fine interface flux uses two fine transverse faces. Physical Neumann flux0 is explicit. Original guard is an offline operator, not a solver report or guessed scaled ratio.

Independent additional support probe poisons covered parent potential to12345: restriction removes it, uniform residual0. A9e-12 active fine interface perturbation below phierror threshold is rejected by OriginalF. Author9e-11 uniform perturbation also rejects. This qualifies no general nonuniform/GPU/MPI solver equivalence.

Raw Ghost carrier capture remains before potential getters. Flat native potential is reshaped to recorded geometry. Getter may materialize Field; this limitation is declared in provenance and separate from raw Ghost freshness/value evidence. Initial tick0/time0/dt0 and acceptedtimeDT are unchanged; all archive/state captures precede guards. Restart compares carrier/state/provider arrays exact, CP12accepted9 authenticated by fixture. Fresh SDK Native reception still required; no Root SCI or runtime claim.

Coherent independent run32PASS15.54s zeroSkip: new support probe +exact author preparation/public callback suites. No Native/JIT/build/ENV/Main mutation.
