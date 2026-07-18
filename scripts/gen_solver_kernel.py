#!/usr/bin/env python3
"""Emit a GENERATED custom-solver C++ kernel from the solver-gen IR-DSL (ADC-462).

Lowers a textbook Richardson solver authored in the ``@pops.codegen.solvers.solver`` IR-DSL
to a self-contained C++ kernel (``pops.codegen.solvers.generate_solver_cpp``) and writes it
to a header the C++ validation test (``tests/cpp/unit/elliptic/test_solver_codegen_generated.cpp``) includes.
This is the build-time half of the codegen->compile->run validation: the test compiles the
emitted kernel against the real ``pops::pops`` runtime and runs it on a known linear system,
comparing to the native prepared-affine Richardson route.

The solver-gen DSL is internal / experimental (Spec 5 criterion 19) and lives in
``pops.codegen.solvers``. Run standalone (no ``_pops`` extension, no numpy needed -- the
IR-authoring + codegen-text layers are pure Python): ``python3 scripts/gen_solver_kernel.py
<out_header>``.
"""
import importlib
import os
import sys
import types


def _load_dsl():
    """Load the solver-gen DSL (``pops.codegen.solvers.dsl`` + ``.solver_cpp``) from source
    WITHOUT importing the heavy ``pops`` / ``pops.codegen`` package ``__init__`` (which pull the
    compiled ``_pops`` extension and numpy). The IR-authoring + C++-emission layers are pure
    Python, so we pre-seed lightweight package shims for ``pops`` and ``pops.codegen`` /
    ``pops.codegen.solvers`` (each just a ``__path__`` into the source tree) and then import the
    two leaf modules directly -- the leaf import never runs any real ``__init__`` file. Returns a
    namespace exposing ``solver`` (the decorator) and ``generate_solver_cpp`` (the lowering)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pops_dir = os.path.join(root, "python", "pops")
    for name, sub in (("pops", ""), ("pops.codegen", "codegen"),
                      ("pops.codegen.solvers", os.path.join("codegen", "solvers"))):
        shim = types.ModuleType(name)
        shim.__path__ = [os.path.join(pops_dir, sub) if sub else pops_dir]
        sys.modules[name] = shim
    dsl = importlib.import_module("pops.codegen.solvers.dsl")
    solver_cpp = importlib.import_module("pops.codegen.solvers.solver_cpp")
    return types.SimpleNamespace(solver=dsl.solver,
                                 generate_solver_cpp=solver_cpp.generate_solver_cpp)


# omega / tol of the generated Richardson solver. They MUST match the constants the C++ validation
# test (tests/cpp/unit/elliptic/test_solver_codegen_generated.cpp) feeds the native prepared
# Richardson controls so the two trace the same iterates and stop at the same residual level
# (parity). omega = 1e-3 under-relaxes the SPD
# Helmholtz operator A = I - 0.1*Lap on the 32x32 grid (lambda_max ~ 820, stable for omega < ~2.4e-3);
# tol is the ABSOLUTE residual L2 norm the loop breaks on.
GEN_OMEGA = 2.0e-3
GEN_ABS_TOL = 1.0e-8


def _build_richardson(dsl):
    """Register the Richardson solver IR (the spec example): x <- x + omega*(b - A x),
    looping while ||b - A x|| > tol and it < max_iter. omega / tol are IR literals."""
    @dsl.solver(name="richardson_gen", signature="(A, b)")
    def richardson(ctx, a, b):  # noqa: D401 - the IR builder
        x = ctx.zeros_like(b)
        it = ctx.scalar_int(0)

        def converging():
            return ctx.logical_and(
                ctx.norm2(ctx.residual(a, x, b)) > GEN_ABS_TOL,
                it < ctx.scalar_int(500000))

        with ctx.while_(converging):
            r = ctx.residual(a, x, b)
            x = ctx.combine(x + GEN_OMEGA * r)
            it = it + ctx.scalar_int(1)
        return x

    return richardson


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: gen_solver_kernel.py <out_header>\n")
        return 2
    dsl = _load_dsl()
    solver = _build_richardson(dsl)
    src = dsl.generate_solver_cpp(solver)
    out = argv[1]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="ascii") as handle:
        handle.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
