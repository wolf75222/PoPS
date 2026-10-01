# Generic ConservativeCellAverage constant reproduction

Private Source patch based on a2408d6f. The actual Tag diagnostic reported
valid constant `0x1.ffffffffffffep-1` already at bind, distinct from the separately
observed zero ghost storage. This patch addresses only the valid projection.
No equation, threshold, transfer, Tag guard or ghost materialization is changed.

Cause: AnalyticCellAverage used tensor four-point Gauss values with rounded
binary weights, accumulated sum(w*f), then multiplied by2^-Dim. That sequence
perturbs even an expression constant at every quadrature point. The existing
single-literal optimization cannot handle the public expression1+0*cos(x).

The first author gel ab026d4 used a fixed affine reference. Independent review
found an avoidable overflow: f=A*x with A=1.5e308 on [-1,1] has finite samples
and exact mean0, but subtracting opposite-sign samples overflowed. That gel
must not be integrated alone.

The corrected rule forms a progressive weighted convex mean. At each sample,
t=weight/(prior_weight+weight). Same-sign values use mean+t*(value-mean), whose
difference cannot overflow for finite operands; opposite signs use
(1-t)*mean+t*value, avoiding the dangerous subtraction. Identical values skip
interpolation entirely and retain their exact bit pattern, including signed zero.
Nodes, weights, sample count, exact-integral and Gaussian paths remain unchanged.
In exact arithmetic this is the same normalized Gauss average, and no tolerance,
rounding, model/field name or particular physical value enters the algorithm.

Source-only host validation compiles the exact source quadrature body with POD
geometry/evaluator interfaces in a temporary directory, without PoPS/Kokkos/DSO
builds or native binding. Arbitrary positive/negative, tiny/large and signed-zero
constants reproduce exactly in dimensions1/2/3. Degree1..7 antiderivative controls
pass; varying the observed coordinate changes tensor traversal order. The same
probe against the unmodified a2408d6f header fails `constant mode changed`.

The real C++ AnalyticExpression suite gains a host test using genuine compiled
analytic programs and Geometry: arbitrary expression constants in both operand
orders and degree-six antiderivative controls, dimensions1/2/3. That complete
repository test has been prepared for ROOT but not built/run here. The existing
literal, explicit-integral, Gaussian and invalid-input tests remain intact.

Command executed in the existing pops interpreter:
`rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest -q tests/review/test_sol61_constant_projection_host.py`
Result:1PASS1.27s. This is Source host evidence, not a Native receipt. No setup_env,
installed ENV mutation or native/JIT build was performed. Public projection and
receipt schemas need no semantic version change, but header bytes and numerical
initialization bits change: ROOT must rebuild/requalify the SDK/DSO and actual
installed Tag run, with fresh identity receipts. Ghost freshness remains an
independent obligation assigned elsewhere.

Coherent Source host plus actual public Tag validate/resolve/emission suite:
11PASS13.02s, true private WT/python path asserted and _pops absent. No installed
Native module selected. Required ROOT C++ node:
AnalyticExpression.ExpressionCellAveragesPreserveConstantsAndPolynomialMomentsOnHost.

Correction validation also adopts the independently authored opposed-sample
probe (Galileo/ROMEO inventory review), changing its regression assertion to
finite exact-body reception. Dimensions1/2/3 and both equivalent linear
expressions now stay finite and near the independent odd antiderivative0.
Max-magnitude/subnormal/signed-zero constants and degree1..7 moments also pass.
The actual C++ host suite receives the same opposed-sign antiderivative control;
ROOT still owns its complete build/execution. Tiny Source host pair:2PASS2.15s.
