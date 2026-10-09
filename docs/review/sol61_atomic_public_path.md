# Independent six-atom public M17 composition preparation

Base preparation f9685154; independent M16 review0261ab38 retained. M17 author
7541f8f3 plus corrective45e746ab are materialized exactly. This case uses its public EndpointPathInputs,
endpoint_polynomial_path and NormalizedPolynomialPath, not the older SymbolicPath
route or a library-specific backend. No production author files were changed.

The six explicit atoms and freely ordered degree-two monomials are defined in
AtomicCubature. Component density is slot1. The physical flux is the same atomic
measure's next moment, explicitly composed in Python. Bx=rho I and By=-rho I/2
are visible physical equations. The arithmetic DAG separately records that flux
and the exact analytic integral along U(s)=UL+s(UR-UL) in RAW coordinates:
(gx-gy/2)(rhoL+rhoR)(UR-UL)/2. This is not asserted for a normalized straight path.

The declared endpoint speed is |gx|+|gy|+|gx-gy/2|rho. Since every atom has
coordinate absolute value<=1, it bounds each eigenvalue g·v+(gx-gy/2)rho.
Positive endpoint rho makes the raw interpolation positive and its maximum is
attained at an endpoint, proving the complete-path bound. This bound is explicit
Python arithmetic (sqrt of squares), not inferred from physical names/formulas.
Positive atomic weights for the host specimens imply positive definite covariance:
they include independent x/y atoms. Native common SPD recovery remains mandatory.
No realizability repair, corner case floor or arbitrary-measure admission is added.

B=0 is a separate conservative case. It uses the same atomic flux and has zero
nonconservative integral; its independent local flux reference is six scalar
advections weighted by monomials. No full PDE/Fourier/time-convergence claim is
made here, especially for active B. Reconstruction, Rusanov and ForwardEuler are
ordinary public method declarations. Layout is rectangular8x4 on physical2x1.

Public declarations->validate->resolve->emit runs use actual Source classes.
The emitted module contains no FanLi/Hermite recipe and reaches the same common
PathConservativeFiniteVolume machinery. The host witness compiles the actual
compiler-generated arithmetic kernel against real headers with C++20 -O2,
-fno-fast-math -ffp-contract=off. It compares directional flux to independent
particle summation and the nonconservative integral to the analytic raw formula,
for signed directions and zero direction; reversal is bitwise antisymmetric,
identical endpoints give zero, and negative density refuses. This is CPU/header
preparation, not a compiled PoPS package or Native/AMR/MPI/device reception.

Commands use read-only ir17 Python, env -u PYTHONPATH,
PYTHONDONTWRITEBYTECODE=1, --noconftest -p no:cacheprovider -o 'pythonpath=python .'.
Tests: tests/review/test_sol61_atomic_cubature.py and
 tests/review/test_sol61_atomic_public_path.py. Public pipeline initially2P15.80s;
actual generated-header witness2P2.42s. Final coherent results in handoff.

Author7541 raw-speed and dead-node defects are corrected in45e746ab. The added
equivalent raw-density speed variant is actually compiled and evaluated here;
unused graph rejection is covered by the coherent author cohort. Final coherent
case+reference+author-graph/header cohort:18PASS20.25s, zero skips. ROOT retains
independent central review, full author results and Native rebuild gates. No global
M17, scientific, performance or mission94 completion is claimed by this case.
