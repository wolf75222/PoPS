"""Environment doctor + official capability matrix (Spec-4 PR-F).

``doctor()`` diagnoses the links the module AND the runtime DSL compilation depend on (the class
of "build environment != execution environment" bugs). ``capabilities()`` is the single
introspectable matrix of what each facade / geometry / backend supports. Both read ``_pops`` and
the codegen / physics layers, so they live in the runtime layer.
"""

from __future__ import annotations

from typing import Any

from pops._generated_release_contract import (
    AMR_CHECKPOINT_PAYLOAD_VERSION,
    UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
)
from pops.runtime import _threading
from pops.runtime._threading import has_kokkos

# Canonical token orders for the matrix the doctor prints. The token SET is derived from the
# descriptor catalogs (see _descriptor_tokens); this only pins the display order so the audit
# table reads the same every run (and the test_capabilities contract keeps its ordered lists).
_RIEMANN_ORDER = ("rusanov", "hll", "hllc", "roe")
_LIMITER_ORDER = ("none", "minmod", "vanleer", "weno5", "mc", "superbee")


def _ordered(tokens: Any, order: Any) -> Any:
    """Tokens kept in canonical ``order`` first, then any extras sorted (deterministic display)."""
    present = set(tokens)
    ranked = [t for t in order if t in present]
    extra = sorted(t for t in present if t not in order)
    return ranked + extra


def _descriptor_tokens() -> Any:
    """Available brick tokens per category, sourced from the descriptor catalogs (Spec 5 sec.12).

    Single source of truth: this reads the SAME inert catalogs that
    the internal descriptor catalog report walks (riemann / limiter / reconstruction /
    elliptic solvers), so adding or retiring a descriptor cannot silently desync the doctor matrix
    from the introspectable capability matrix. Only descriptors that declare themselves available
    are reported; MC and Superbee are ordinary native limiter descriptors and therefore appear
    through this same path without a doctor-specific allowlist. Pure: no ``_pops`` import, no
    numeric loop.
    """
    from pops.numerics.reconstruction import reconstruction
    from pops.numerics.reconstruction.limiters import limiters
    from pops.numerics.riemann import riemann
    from pops.solvers.elliptic import CartesianCG, FFT, GeometricMG

    def _available(namespace: Any) -> Any:
        names = []
        for attr in sorted(vars(namespace)):
            factory = getattr(namespace, attr)
            if not callable(factory):
                continue
            try:
                descriptor: Any = factory()
            except TypeError:
                continue  # a selector that needs an argument (User(...)); not a standing entry.
            if getattr(descriptor, "available", False):
                names.append(descriptor.name)
        return names

    riemann_tokens = _ordered(_available(riemann), _RIEMANN_ORDER)
    limiter_tokens = _available(limiters)
    recon_tokens = _available(reconstruction)
    # The DSL "limiter" options are the slope limiters plus the high-order WENO5 reconstruction a
    # backend exposes by token; "none" is the no-limiter sentinel (not a catalog brick). Only the
    # canonical "weno5" token is surfaced (its "weno5z" alias selects the same pops::Weno5), so the
    # printed list stays the historical set while still being sourced from the catalog.
    high_order = ["weno5"] if "weno5" in recon_tokens else []
    dsl_limiters = _ordered(["none", *limiter_tokens, *high_order], _LIMITER_ORDER)
    # Elliptic field-solver tokens (the Poisson row), sourced from the elliptic descriptors.
    poisson = {
        "cartesian_cg": CartesianCG().name,
        "geometric_mg": GeometricMG().name,
        "fft": FFT().scheme,
    }

    return {
        "riemann": riemann_tokens,
        "dsl_limiters": dsl_limiters,
        "poisson": poisson,
    }


def doctor(verbose: bool = True) -> Any:
    """Diagnose the installed runtime environment and native toolchain.

    Checks each link on which the module AND the runtime compilation of the DSL depend (the class of
    bugs "build environment != execution environment", e.g. the `which c++` of a conda env
    that rejects -std=c++23). Returns a dict {check: (ok, detail)} ; verbose=True prints it."""
    import os
    import sys

    checks = {}

    # 1. interpreter + extension (cpython-3XY ABI trap)
    from pops._native_selector import selected_native_module

    _pops = selected_native_module(required=True)
    so = getattr(_pops, "__file__", "?")
    checks["interpreteur"] = (
        True,
        "%s (%d.%d) ; extension %s"
        % (sys.executable, sys.version_info[0], sys.version_info[1], so),
    )

    # 2. numpy (required by the codegen IR host evaluator)
    try:
        import numpy

        checks["numpy"] = (True, numpy.__version__)
    except Exception as e:
        checks["numpy"] = (
            False,
            "ABSENT from this interpreter (%s) -> compiling a model will fail. "
            "Install numpy in THIS python." % e,
        )

    # 3. compiled compute backend
    hk = has_kokkos()
    checks["kokkos"] = (
        hk is not False,
        {
            True: "Kokkos module (thread/device resources selected before initialization)",
            False: "SERIAL module (rebuild preset python-parallel for threaded execution)",
            None: "undetermined (old module without __has_kokkos__)",
        }[hk],
    )

    # 4. runtime DSL compiler (the link of the -std=c++23 bug)
    try:
        from pops.codegen import toolchain as _tc
        from pops.codegen import abi as _abi
    except Exception as e:
        checks["dsl"] = (False, "import pops.codegen failed (%s)" % e)
        _tc = None
    if _tc is not None:
        baked = _tc.loader_cxx_compiler()
        cc = _tc._default_cxx(None)
        if not cc:
            checks["compilateur"] = (
                False,
                "NO C++ compiler found (POPS_CXX, module, PATH). "
                "Install Xcode CLT (macOS) or `conda install cxx-compiler`.",
            )
        else:
            origin = (
                "$POPS_CXX"
                if os.environ.get("POPS_CXX") == cc
                else "baked by the _pops build"
                if cc == baked
                else "PATH (which)"
            )
            try:
                std = _tc._probe_cxx_std(cc, _tc.loader_cxx_std())
                checks["compilateur"] = (True, "%s [%s] ; -std=%s accepted" % (cc, origin, std))
            except RuntimeError as e:
                checks["compilateur"] = (False, str(e).splitlines()[0])
            if baked and cc != baked:
                checks["compilateur_abi"] = (
                    False,
                    "runtime compiler (%s) != build (%s) -> risk "
                    "of 'incompatible ABI' rejection on production "
                    "backend. export POPS_CXX=%r to force the one "
                    "from the build." % (cc, baked, baked),
                )

        # 5. pops headers (production DSL : the signature must match the one baked into _pops)
        try:
            inc = _tc.pops_include()
            checks["include"] = (True, inc)
            # 5b. SYNCHRONIZATION headers <-> module (real bug : module built BEFORE a git pull ->
            # the DSL loader references C++ signatures absent from the old .so -> dlopen 'symbol
            # not found' cryptic). We compare the baked signature to the one of the current tree.
            baked_sig = _abi.module_header_signature()
            if baked_sig is not None:
                cur_sig = _tc.pops_header_signature(inc)
                if cur_sig == baked_sig:
                    checks["headers_sync"] = (
                        True,
                        "headers == module build (sig %s...)" % baked_sig[:12],
                    )
                else:
                    checks["headers_sync"] = (
                        False,
                        "headers MODIFIED since the _pops build "
                        "(stale module) -> rebuild : cmake --build "
                        "build-py --target _pops (otherwise : dlopen "
                        "'symbol not found' on production backend)",
                    )
        except RuntimeError as e:
            checks["include"] = (False, "pops headers not found (set POPS_INCLUDE) : %s" % e)

        # 5c. Kokkos root for the production package compiler.
        # PoPS is Kokkos-only : every DSL .so that includes the pops headers MUST compile against an
        # installed Kokkos (Serial is enough on CPU), found via POPS_KOKKOS_ROOT / Kokkos_ROOT.
        kroot = _tc._native_kokkos_root()
        if kroot is None:
            checks["kokkos_root"] = (
                False,
                "POPS_KOKKOS_ROOT / Kokkos_ROOT not set -> production package cannot compile "
                "(the tutorial dead-ends on 'no DSL backend'). Fix (conda) :\n"
                '      conda env config vars set POPS_KOKKOS_ROOT="$CONDA_PREFIX"\n'
                '      conda env config vars set Kokkos_ROOT="$CONDA_PREFIX"\n'
                "      conda deactivate && conda activate pops",
            )
        else:
            checks["kokkos_root"] = (True, kroot)
            # 5d. A CUDA Kokkos on a host without nvcc breaks BOTH `pip install .` (find_package picks it
            # -> nvcc) AND the production .so. On a CPU host, install the CPU Kokkos variant instead.
            import shutil

            cuda = False
            try:
                with open(os.path.join(kroot, "include", "KokkosCore_config.h")) as _f:
                    cuda = any(line.startswith("#define KOKKOS_ENABLE_CUDA") for line in _f)
            except OSError:
                pass
            if cuda and shutil.which("nvcc") is None:
                checks["kokkos_cuda"] = (
                    False,
                    "Kokkos at %s is a CUDA build but nvcc is not on PATH -> `pip install .` fails "
                    "'Could not find nvcc'. Fix for a CPU host, recreate the env with CPU Kokkos :\n"
                    '      CONDA_OVERRIDE_CUDA="" bash scripts/setup_env.sh' % kroot,
                )

    # 6. current threads
    checks["threads"] = (
        True,
        "OMP_NUM_THREADS=%s ; first System created=%s"
        % (os.environ.get("OMP_NUM_THREADS", "(default)"), _threading._first_system_built),
    )

    if verbose:
        for cname, (ok, detail) in checks.items():
            print("[%s] %-16s %s" % ("OK " if ok else "FAIL", cname, detail))
        if all(ok for ok, _ in checks.values()):
            print("=> healthy environment : module importable, DSL compilable, ABI coherent.")
        else:
            print("=> fix the FAILs above before using the DSL backend='production'.")
    return checks


def capabilities() -> Any:
    """OFFICIAL MATRIX of capabilities by facade / geometry / backend (audit 2026-06, wave 2).

    SINGLE source of truth consultable by scripts and docs (the audits showed that System,
    uniform System, AMR and the DSL backends diverged silently). The entries reflect the gates
    actually coded by the exact-ranked runtime and ``dsl._BACKENDS``; the
    combinations outside the matrix raise an explicit error on the C++ side (never a silent ignore).

    Sec 12: the riemann / limiter / reconstruction / Poisson token lists are DERIVED from the
    descriptor catalogs via :func:`_descriptor_tokens` (the same single source
    the internal descriptor report reads), not hardcoded, so adding or retiring a
    brick cannot silently desync this matrix from the introspectable one.
    """
    from pops._native_selector import selected_native_module

    selected_native_module(required=True)
    # Sec 12: derive the riemann / limiter / reconstruction / Poisson token lists from the descriptor
    # catalogs (the same source as the internal descriptor report) instead of hardcoding them, so a
    # new descriptor cannot silently desync the doctor matrix from the introspectable one.
    tok = _descriptor_tokens()
    riemann_all = list(tok["riemann"])
    poisson_cg = tok["poisson"]["cartesian_cg"]
    poisson_mg = tok["poisson"]["geometric_mg"]
    poisson_fft = tok["poisson"]["fft"]
    dsl_limiters = list(tok["dsl_limiters"])
    from pops.runtime_environment import runtime_environment_report

    runtime_env = runtime_environment_report()
    poisson_system = [
        "%s (exact-ranked Cartesian Poisson; 1D/2D/3D)" % poisson_cg,
        "%s (exact-ranked Cartesian Dim=1/2/3, periodic, constant eps, ordered MPI slabs)"
        % poisson_fft,
    ]
    return {
        # Immutable rank selected by the loaded native artifact before runtime construction.
        "dimension": runtime_env["dimension"],
        "precision": {
            "real": runtime_env["precision"],
            "real_bytes": runtime_env["real_bytes"],
            "supports_single_precision": runtime_env["supports_single_precision"],
            "supports_mixed_precision": runtime_env["supports_mixed_precision"],
        },
        "runtime_environment": runtime_env,
        "riemann": {
            "system_cartesian": riemann_all,
            "amr": list(riemann_all),
            "notes": {
                "rusanov": "minimal generic (physical flux + exact provider pack + declared stability bound)",
                "hll": "generic with signed waves (typed Model.wave_speeds(...) "
                "explicit WITHOUT primitive 'p', or historical path eigenvalues + 'p')",
                "hllc": "model capability HasHLLCStructure required -- "
                "emitted by the DSL via m.enable_hllc() (roles + 'p', including 3-var non "
                "Euler, passive advected scalars) ; native Euler and isothermal bricks provide it. "
                "No component-count/layout inference and no fallback.",
                "roe": "model capability HasRoeDissipation required "
                "-- TWO DSL paths : (a) m.enable_roe() generated from the roles (roles + "
                "'p' : with Energy = transcribed canonical algebra, without Energy = "
                "c=sqrt(p/rho) Roe average, passive scalars on the entropy wave) ; (b) "
                "m.roe_dissipation(x=, y=) PROVIDED by the user (own eigenstructure, "
                "left()/right() of the two states, helper m.flux_jacobian auto-derived). Paths "
                "exclusive (a single provider of the hook). has_roe covers both ; native Euler "
                "and isothermal bricks provide the hook. "
                "No component-count/layout inference and no fallback.",
            },
        },
        "time": {
            "system": [
                "explicit (ssprk2|ssprk3)",
                "imex (= SourceImplicitBE)",
                "imexrk_ars222 (IMEX-RK family, ARS(2,2,2) scheme, order 2 ; cartesian only ; "
                "fully implicit source)",
                "Program factories Lie|Strang + explicit Program.solve",
            ],
            "amr": [
                "explicit (SSPRK2/Heun, order 2 + effective reflux flux)",
                "euler (Forward Euler)",
                "ssprk3 (order 3 + effective reflux flux)",
                "coarse/fine SSP stages sample the parent window at RK abscissae",
                "IMEX execution requires a typed implicit Program primitive "
                "(no AMR spatial-runtime fallback)",
                "Program factories Lie|Strang + hierarchy-scoped Program.solve",
            ],
            "newton_options": "options (max_iters/tol/fd_eps/damping) : System route "
            "or typed Program solve; AMR rejects non-default block options and "
            "diagnostics until its typed implicit primitive exists (.so block "
            "loaders: explicit rejection) ; "
            "analytic jacobian via m.source_jacobian ; newton_diagnostics/"
            "newton_report requires the executing typed solve",
        },
        "stability_policy": {
            "system": [
                "transport (max_wave_speed | stability_speed)",
                "source_frequency",
                "stability_dt",
                "coupled_source.frequency",
                "add_dt_bound (global, all_reduce_min)",
                "last_dt_bound",
            ],
            "amr": [
                "transport (max_wave_speed | stability_speed)",
                "source_frequency",
                "stability_dt",
                "coupled_source.frequency (multi-block)",
                "add_dt_bound",
                "last_dt_bound",
            ],
        },
        "poisson": {
            "system_cartesian": poisson_system,
            "amr": ["%s only ; rhs charge_density|composite" % poisson_mg],
        },
        "geometry": {
            "system_cartesian": "ranked n cells (1D/2D/3D, rectangular allowed) ; "
            "mono-box (multi-box = AmrSystem or MPI mono-box)",
            "amr": "hierarchy of levels (BoxArray per level, dynamic regrid) ; "
            "transition ratios selected by the authenticated hierarchy (exact native-rank "
            "values; no process-global AMR ratio is advertised)",
        },
        "schur": {
            "system_cartesian": "explicit Program.solve(LinearProblem(..., nullspace=None), "
            "solver=GMRES/BiCGStab) ; "
            "authored roles/fields ; generic matrix-free operator",
            "amr": "hierarchy-scoped Program.solve with CompositeTensorFAC ; gather-all-levels, "
            "one composite tensor solve, then reconstruct-all-levels through the Program",
        },
        "backends_dsl": {
            "default": "production package; compiler, headers and module ABI must match",
            "production": {
                "tier": "production",
                "riemann": list(riemann_all),
                "limiter": dsl_limiters,
                "stride": True,
                "evolve_false": True,
                "mpi": True,
                "amr": "target='amr_system'",
                "stability_hooks": True,
                "bind_params": "fixed at install",
            },
        },
        "io": {
            "scientific_output": (
                "typed NPZ/ParaView/HDF5 providers with explicit SERIAL, ROOT, COLLECTIVE or "
                "PER_RANK topology; collective HDF5 requires the native C++ parallel-HDF5 route"
            ),
            "checkpoint_restart": (
                "strict accepted-state Uniform v%d / AMR v%d, including multi-block, active "
                "regridding, fields, histories, clocks, tagging hysteresis and consumer cursors; "
                "exact MPI_COMM_WORLD captures collectively and publishes one rank-0 NPZ artifact"
                % (UNIFORM_CHECKPOINT_PAYLOAD_VERSION, AMR_CHECKPOINT_PAYLOAD_VERSION)
            ),
        },
        "amr_layout": {
            "set_conservative_state": "mono-block, native multi-block, and compiled multi-block "
            "(.so loaders ; complete block-qualified conservative state)",
        },
        "regrid": {
            "authority": "resolved AMRTagging graph installed as one prepared native program",
            "state_and_field_leaves": "block-qualified exact Handle identities",
            "logical_nodes": "prepared not/any/all bytecode; no scalar threshold fallback",
            "phi_gradient": "ordinary prepared field-gradient leaf",
        },
        "aux": {
            "authority": "owner-qualified ProviderPack ComponentKey routes; no reserved names or fixed width",
            "inputs": "stage_auxiliary_input(ComponentKey(...), values) only for declared InputAux",
            "derived": "native DerivedAux launchers publish transactionally from their exact dependency DAG",
            "field_outputs": "FieldOperator/solve producers own their typed routes and storage groups",
            "boundary": "per-output AuxiliaryBoundary: topology inheritance, first-order extrapolation, or dirichlet",
        },
    }
