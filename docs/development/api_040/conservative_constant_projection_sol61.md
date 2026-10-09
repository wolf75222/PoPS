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


## Real repository C++ gtest reception

The true source file tests/cpp/unit/runtime/test_analytic_expression.cpp was
compiled and linked by the repository CMake target, not by the extracted-body
probe. Frozen e75ea17 header and test bytes were verified identical to Git blobs
before recording evidence. Build/output is a new private directory:
`/Users/romaindespoulain/dev/tmp/PoPS-sol61-constant-projection/build-gtest-projection-e75ea17`.
The worktree remained clean after configure/build/run, before this evidence note.

Successful configure (the repository requires explicit POPS_NATIVE_DIM):

```sh
/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/cmake -S . -B build-gtest-projection-e75ea17 \
  -DPOPS_NATIVE_DIM=2 -DPOPS_BUILD_TESTS=ON -DPOPS_BUILD_PYTHON=OFF \
  -DPOPS_USE_KOKKOS=ON -DPOPS_USE_MPI=OFF -DPOPS_USE_HDF5=OFF \
  -DPOPS_TESTS_FAST_O0=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17 \
  -DKokkos_DIR=/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/lib/cmake/Kokkos \
  -DOpenMP_CXX_FLAGS='-Xpreprocessor -fopenmp -I/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/include' \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/lib/libomp.dylib \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/Users/romaindespoulain/dev/tmp/pops-audit-20260907/build-mpi/_deps/googletest-src
/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/cmake --build build-gtest-projection-e75ea17 --target test_analytic_expression -j1
env OMP_NUM_THREADS=2 build-gtest-projection-e75ea17/bin/test_analytic_expression \
  --gtest_filter='AnalyticExpression.*' \
  --gtest_output=xml:build-gtest-projection-e75ea17/analytic-expression.xml
```

Result:12PASS,0failures/errors/skips,0.100s gtest total. The new host test
ExpressionCellAveragesPreserveConstantsAndPolynomialMomentsOnHost executes real
compiled AnalyticProgram/Geometry in dimensions1/2/3. The entire suite also
executes existing actual ranked Kokkos materialization, exact polynomial
integrals, Gaussian neutrality, literals and invalid-input refusals. C++build
rc0; two existing nodiscard warnings from EXPECT_THROW uses remain in the log.
No PoPS runtime-library, Python extension or installed SDK/DSO was built/installed.
Only the dedicated executable, test main and private GoogleTest archives were
built. j1 limited simultaneous translation-unit compilation to one.

The actual runtime log shows Kokkos::OpenMP::initialize; enabled devices are
OPENMP/SERIAL. Mach-O dependency/RPATH inspection resolves Kokkos5.2 and libomp
through the existing pops-api040-ir17/lib directory. This is CPU OpenMP C++
header conformance, not a new installed-package Python lifecycle receipt, MPI
qualification, GPU/device execution or a ghost-materialization result.
conservative_projection_real_cpp_reception_sol61.json pins source, executable,
raw XML/log/CMake cache, configure/build logs and actual dependency bytes.
The NativeWT, ENV/site-packages and installed DSO received no writes from this
agent. ROOT's separate SDK/DSO rebuild and Tag lifecycle rerun remain necessary.
