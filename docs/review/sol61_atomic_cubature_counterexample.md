# Independent preparation: finite-atomic degree-two closure

SOURCE_ONLY, base 3f5a55026f39de3429f3bbb552d043a2a65a9d09. No native execution,
performance qualification, or scientific receipt.

The six nodes (0,0), (+1,0), (-1,0), (0,+1), (0,-1), (+1,+1) define a
non-Gaussian atomic closure of the complete degree-two raw basis. Storage is
(11,00,02,10,20,01); density is not the first component. Fluxes are the next
monomials of this same measure. In particular m30=m10, m03=m01 and
m21=m12=m11. These identities follow from these explicitly declared nodes,
and are not inferred by the compiler from names or model recognition.

AtomicCubature is a Python authoring-time Vandermonde inversion with exact
rational coefficients, followed by ordinary scalar arithmetic. It is not a
cell callback. It admits other unisolvent node/basis systems, rejects singular
ones, and does not repair signed weights. Binary64 coefficient conversion and
ordered ordinary sums are explicit; no compensated-sum or stability claim is
made by this preparation.

The proposed nonconservative product is B_g(U)dU=(gx-gy/2) rho dU. The chosen
path is U(s)=UL+s(UR-UL) in RAW coordinates. Its integral is
(gx-gy/2)(rhoL+rhoR)(UR-UL)/2. This formula is not asserted for an affine path
in normalized variables. Positive endpoint densities imply positivity along
this path. Eigenvalues are g·v_atom+(gx-gy/2)rho(s), so the bound
max_atom|g·v_atom|+|gx-gy/2|max(rhoL,rhoR) holds on the whole specified path,
including signed directions. No Fourier/free-streaming claim is made for
active B. B=0 must remain a separate conservative specimen.

The M17 prototype currently integrates rho dW for an affine normalized path.
Its public raccord for an explicit raw endpoint integral is pending; no fake
adapter or old path route is counted as exercising that port. The M16
explicit-basis affine update is a distinct subsequent test against transformed
particles, without claiming atomic-weight positivity after general rotation.

Command: env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1
/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest
--noconftest -p no:cacheprovider -o pythonpath=python
 tests/review/test_sol61_atomic_cubature.py -q

Result: 2 passed in 0.46s. This is a preparation, not completion of the public
M16/M17 composition case requested by ROOT.
