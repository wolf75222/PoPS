"""adc.lib -- a catalog of typed brick descriptors and IR macros (Spec 3).

adc.lib is NOT a Python numerics library. Every entry is one of:

* a NATIVE brick -- a descriptor naming a C++ type already in ``include/adc``
  (``adc.lib.riemann.HLLC()`` -> ``adc::numerics::fv::HLLCFlux``);
* a GENERATED brick -- a descriptor of a DSL-authored brick compiled to C++;
* a MACRO brick -- a Python function that builds Program IR
  (``adc.lib.time.predictor_corrector`` delegates to :mod:`adc.time` ``std``);
* an EXTERNAL C++ brick -- a descriptor of a user ``.so`` registered by id
  (``adc.lib.riemann.User("my_hllc")``).

A descriptor carries metadata only -- a native id, a runtime scheme string,
requirements and capabilities. It computes nothing; the codegen and runtime
consume it. The namespaces mirror the Spec 3 catalog (riemann, reconstruction,
limiters, spatial, fields, solvers, preconditioners, diagnostics, projections,
invariants, time).
"""
from types import SimpleNamespace

__all__ = ["BrickDescriptor", "riemann", "reconstruction", "limiters", "spatial",
           "fields", "solvers", "preconditioners", "diagnostics", "projections",
           "invariants", "time"]

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name, brick_type, *, category="brick", native_id="",
                 scheme=None, requirements=None, capabilities=None, options=None,
                 available=True, expression=None):
        if brick_type not in BRICK_TYPES:
            raise ValueError("brick_type %r must be one of %s"
                             % (brick_type, ", ".join(BRICK_TYPES)))
        self.name = str(name)
        self.brick_type = str(brick_type)
        self.category = str(category)
        self.native_id = str(native_id)
        self.scheme = scheme
        self.requirements = dict(requirements or {})
        self.capabilities = dict(capabilities or {})
        self.options = dict(options or {})
        self.available = bool(available)
        # Optional board value carried by a generated/macro brick; kept OFF the
        # identity key (it may be an unhashable board node).
        self.expression = expression

    def _key(self):
        return (self.category, self.name, self.brick_type, self.native_id,
                self.scheme, tuple(sorted(self.options.items())))

    def __eq__(self, other):
        return isinstance(other, BrickDescriptor) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "BrickDescriptor(%r, %r, scheme=%r)" % (
            self.name, self.brick_type, self.scheme)


def _native(name, native_id, scheme, *, category, caps=None, **options):
    """A native-brick descriptor; ``caps`` lists required model capabilities."""
    req = {"capabilities": list(caps)} if caps is not None else {}
    return BrickDescriptor(name, "native", category=category, native_id=native_id,
                           scheme=scheme, requirements=req, options=options or None)


# --- riemann ---------------------------------------------------------------
def _riemann(name, native_id, caps):
    return _native(name, native_id, name, category="riemann", caps=caps)


riemann = SimpleNamespace(
    Rusanov=lambda: _riemann("rusanov", "adc::numerics::fv::RusanovFlux",
                             ["max_wave_speed"]),
    HLL=lambda: _riemann("hll", "adc::numerics::fv::HLLFlux",
                         ["physical_flux", "wave_speeds"]),
    HLLC=lambda: _riemann("hllc", "adc::numerics::fv::HLLCFlux",
                          ["physical_flux", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann("roe", "adc::numerics::fv::RoeFlux",
                         ["physical_flux", "roe_average"]),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="riemann", native_id=native_id,
        scheme="user", options=opts or None),
)


# --- reconstruction --------------------------------------------------------
reconstruction = SimpleNamespace(
    FirstOrder=lambda: _native("firstorder", "adc::numerics::fv::NoSlope",
                               "firstorder", category="reconstruction"),
    MUSCL=lambda limiter="minmod": _native(
        "muscl", "adc::numerics::fv::Minmod", limiter,
        category="reconstruction", limiter=limiter),
    WENO5=lambda: _native("weno5", "adc::numerics::fv::Weno5", "weno5",
                          category="reconstruction"),
    WENO5Z=lambda: _native("weno5z", "adc::numerics::fv::Weno5", "weno5",
                           category="reconstruction"),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="reconstruction", native_id=native_id,
        scheme="user", options=opts or None),
)


# --- limiters --------------------------------------------------------------
limiters = SimpleNamespace(
    Minmod=lambda: _native("minmod", "adc::numerics::fv::Minmod", "minmod",
                           category="limiter"),
    VanLeer=lambda: _native("vanleer", "adc::numerics::fv::VanLeer", "vanleer",
                            category="limiter"),
    # MC / Superbee are catalogued but not yet wired natively (available=False).
    MC=lambda: BrickDescriptor("mc", "native", category="limiter", scheme="mc",
                               available=False),
    Superbee=lambda: BrickDescriptor("superbee", "native", category="limiter",
                                     scheme="superbee", available=False),
)


# --- spatial ---------------------------------------------------------------
spatial = SimpleNamespace(
    FiniteVolumeResidual=lambda **o: _native(
        "fv_residual", "adc::numerics::fv::SpatialOperator", "fv",
        category="spatial", **o),
    FluxDivergence=lambda **o: _native(
        "flux_divergence", "adc::numerics::fv::SpatialOperator", "fv",
        category="spatial", **o),
    SourceAssembly=lambda **o: _native(
        "source_assembly", "adc::numerics::fv::SpatialOperator", "fv",
        category="spatial", **o),
)


# --- fields (elliptic) -----------------------------------------------------
fields = SimpleNamespace(
    Poisson=lambda **o: _native("poisson", "adc::numerics::elliptic::Poisson",
                                "poisson", category="field", **o),
    Helmholtz=lambda **o: _native("helmholtz", "adc::numerics::elliptic::Helmholtz",
                                  "helmholtz", category="field", **o),
    EllipticSolve=lambda **o: _native(
        "elliptic_solve", "adc::numerics::elliptic::FieldSolver", "elliptic",
        category="field", **o),
    GeometricMG=lambda **o: _native(
        "geometric_mg", "adc::numerics::elliptic::GeometricMG", "geometric_mg",
        category="field", **o),
)


# --- solvers (linear / nonlinear) ------------------------------------------
def _solver(name, native_id, **o):
    return _native(name, native_id, name, category="solver", **o)


solvers = SimpleNamespace(
    CG=lambda **o: _solver("cg", "adc::numerics::linear::CG", **o),
    BiCGStab=lambda **o: _solver("bicgstab", "adc::numerics::linear::BiCGStab", **o),
    GMRES=lambda **o: _solver("gmres", "adc::numerics::linear::GMRES", **o),
    Richardson=lambda **o: _solver("richardson", "adc::numerics::linear::Richardson", **o),
    Newton=lambda **o: _solver("newton", "adc::numerics::local::Newton", **o),
    FixedPoint=lambda **o: _solver("fixed_point", "adc::numerics::local::FixedPoint", **o),
    Schur=lambda **o: _solver("schur", "adc::coupling::SchurCondensation", **o),
)


# --- preconditioners -------------------------------------------------------
preconditioners = SimpleNamespace(
    Identity=lambda: _native("identity", "adc::numerics::linear::Identity",
                             "identity", category="preconditioner"),
    Jacobi=lambda: _native("jacobi", "adc::numerics::linear::Jacobi", "jacobi",
                           category="preconditioner"),
    BlockJacobi=lambda: _native("block_jacobi", "adc::numerics::linear::BlockJacobi",
                                "block_jacobi", category="preconditioner"),
    GeometricMG=lambda **o: _native(
        "geometric_mg", "adc::numerics::elliptic::GeometricMG", "geometric_mg",
        category="preconditioner", **o),
    User=lambda native_id, **opts: BrickDescriptor(
        native_id, "external_cpp", category="preconditioner", native_id=native_id,
        scheme="user", options=opts or None),
)


# --- diagnostics -----------------------------------------------------------
def _diag(_dname, **o):
    return BrickDescriptor(_dname, "macro", category="diagnostic", scheme=_dname,
                           options=o or None)


diagnostics = SimpleNamespace(
    integral=lambda expr=None, **o: _diag("integral", expr=expr, **o),
    norm=lambda kind="l2", **o: _diag("norm", kind=kind, **o),
    mass=lambda **o: _diag("mass", **o),
    momentum=lambda **o: _diag("momentum", **o),
    energy=lambda **o: _diag("energy", **o),
    invariant_error=lambda name=None, **o: _diag("invariant_error", name=name, **o),
    residual=lambda **o: _diag("residual", **o),
)


# --- projections -----------------------------------------------------------
projections = SimpleNamespace(
    positivity=lambda **o: _native("positivity", "adc::numerics::fv::Positivity",
                                   "positivity", category="projection", **o),
    bound_preserving=lambda **o: _native(
        "bound_preserving", "adc::numerics::fv::BoundPreserving",
        "bound_preserving", category="projection", **o),
    conservative_fix=lambda **o: BrickDescriptor(
        "conservative_fix", "generated", category="projection",
        scheme="conservative_fix", options=o or None),
    divergence_cleaning=lambda **o: BrickDescriptor(
        "divergence_cleaning", "generated", category="projection",
        scheme="divergence_cleaning", options=o or None),
)


# --- invariants ------------------------------------------------------------
def _invariant(name, expression=None, over=None):
    """A catalog invariant descriptor; the value ``expression`` is kept off the
    identity key (it may be an unhashable board node) as a plain attribute."""
    return BrickDescriptor(name, "macro", category="invariant", scheme="invariant",
                           options={"over": tuple(over) if over else ()},
                           expression=expression)


invariants = SimpleNamespace(
    invariant=_invariant,
    conservation_check=lambda name, tolerance=1e-10, **o: BrickDescriptor(
        name, "macro", category="invariant", scheme="conservation_check",
        options={"tolerance": tolerance, **o}),
)


# --- time (MACRO bricks: build Program IR via adc.time.std) ----------------
def _std():
    from . import time as _time
    return _time.std


def _time_macro(std_name):
    """A macro brick that forwards to ``adc.time.std.<std_name>``; builds IR only."""
    def macro(P, block, *args, **kwargs):
        return getattr(_std(), std_name)(P, block, *args, **kwargs)
    macro.__name__ = std_name
    macro.__doc__ = "Build the %r time scheme into Program P (adc.time.std)." % std_name
    return macro


time = SimpleNamespace(
    forward_euler=_time_macro("forward_euler"),
    ssprk2=_time_macro("ssprk2"),
    ssprk3=_time_macro("ssprk3"),
    rk4=_time_macro("rk4"),
    rk=_time_macro("rk"),
    adams_bashforth=_time_macro("adams_bashforth"),
    bdf=_time_macro("bdf"),
    strang=_time_macro("strang"),
    lie=_time_macro("lie"),
    imex=_time_macro("imex_local"),
    predictor_corrector=_time_macro("predictor_corrector_local_linear"),
)
