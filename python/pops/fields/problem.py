"""pops.fields.problem -- the typed elliptic field-problem descriptor (Spec 5 sec.5.5).

:class:`FieldProblem` is the inert, typed declaration of a field solve: the unknown, the
governing :class:`pops.math.Equation`, the input/coefficient/boundary/nullspace objects,
the outputs and the solver brick. It declares its requirements / capabilities / options
and validates that it was built from a real :class:`~pops.math.Equation` (not a Python
bool or some other object) and that a solver was provided -- before the runtime is ever
touched.

It computes nothing; codegen / runtime consume the descriptor.
"""
from pops.descriptors import Availability, Descriptor, reject_string_selector
from pops.math import Equation

from .policies import FieldSolvePolicy

# The legacy elliptic-solver tokens the board facade still accepts, lowered to the typed
# descriptor factory name so the string is turned into a real descriptor before it reaches the
# FieldProblem constructor (which rejects a bare string). Spec 5 sec.7: strings are not a public
# API, but the board shortcut lowers them here rather than propagating an untyped selector.
_LEGACY_SOLVER_TOKENS = {"geometric_mg": "GeometricMG", "fft": "FFT", "fft_spectral": "FFT"}


def lower_field_solver(solver):
    """Lower a legacy string solver token to its typed elliptic descriptor (Spec 5 sec.7).

    A descriptor / ``None`` passes through unchanged. A recognised legacy token
    (``"geometric_mg"`` / ``"fft"`` / ``"fft_spectral"``) is turned into the matching typed
    :mod:`pops.solvers.elliptic` descriptor (``fft_spectral`` -> ``FFT(spectral=True)``). An
    unrecognised string is rejected, naming the typed factory. The board facade calls this so its
    string shortcut yields a typed solver instead of propagating an untyped selector.
    """
    if not isinstance(solver, str):
        return solver
    factory = _LEGACY_SOLVER_TOKENS.get(solver)
    if factory is None:
        reject_string_selector(solver, "solver", "pops.solvers.GeometricMG() / pops.solvers.FFT()")
    from pops.solvers.elliptic import FFT, GeometricMG  # lazy: fields must not import solvers.
    if factory == "FFT":
        return FFT(spectral=(solver == "fft_spectral"))
    return GeometricMG()


def _summarize(side, limit=60):
    """A short repr of one equation side, truncated to keep the print() summary small."""
    text = repr(side)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class SolveCadence(Descriptor):
    """The inert record of a field-solve cadence: a schedule + a not-due policy.

    Recorded on a :class:`FieldProblem` by :meth:`FieldProblem.solve` and surfaced by the
    problem's :meth:`~FieldProblem.inspect`. It pairs the typed :class:`pops.time.Schedule`
    (WHEN to solve) with the typed :class:`~pops.fields.policies.FieldSolvePolicy` (what to
    do when the solve is not due). It is authoring metadata only: it carries the schedule and
    policy, it does NOT lower into the Program / codegen. The runtime wiring is a
    codegen-adjacent follow-up (see :meth:`FieldProblem.solve`).
    """

    category = "solve_cadence"

    def __init__(self, schedule, policy):
        self.schedule = schedule
        self.policy = policy

    def options(self):
        return {"schedule": repr(self.schedule), "policy": self.policy.name}

    def requirements(self):
        return dict(self.policy.requirements())

    def capabilities(self):
        return dict(self.policy.capabilities())

    def inspect(self):
        info = super().inspect()
        info["schedule"] = repr(self.schedule)
        info["policy"] = self.policy.inspect()
        return info

    def __repr__(self):
        return "SolveCadence(schedule=%r, policy=%r)" % (self.schedule, self.policy)


class FieldProblem(Descriptor):
    """A typed elliptic field problem: an unknown solved from an :class:`~pops.math.Equation`.

    ``FieldProblem(unknown=phi, equation=(-laplacian(phi) == rhs), solver=GeometricMG())``.
    The ``inputs`` / ``coefficients`` / ``bcs`` / ``nullspace`` / ``outputs`` / ``postprocess``
    fields are typed descriptors (from :mod:`pops.fields`); they are stored and surfaced, not
    interpreted, here. :meth:`validate` rejects a non-:class:`~pops.math.Equation` equation
    (a Python bool produced by ``==`` on plain values is the common mistake) and a missing
    solver.
    """

    category = "field_problem"

    def __init__(self, name=None, unknown=None, equation=None, inputs=(),
                 coefficients=None, bcs=(), nullspace=None, outputs=None,
                 postprocess=None, solver=None):
        self._name = None if name is None else str(name)
        self.unknown = unknown
        self.equation = equation
        self.inputs = tuple(inputs)
        self.coefficients = coefficients
        self.bcs = tuple(bcs)
        self.nullspace = nullspace
        self.outputs = outputs
        self.postprocess = postprocess
        # Spec 5 sec.7 (ADC-534): the elliptic solver is a TYPED descriptor, never a bare string.
        # Reject solver="geometric_mg" / "fft" at construction, pointing at the typed factory, so
        # the mistake is caught before the (silently string-keeping) options()/validate path.
        if isinstance(solver, str):
            reject_string_selector(
                solver, "solver",
                "pops.solvers.GeometricMG() / pops.solvers.FFT() "
                "(pops.solvers.elliptic.GeometricMG())")
        self.solver = solver
        # The inert field-solve cadence recorded by solve(); None until authored.
        self.cadence = None

    @property
    def name(self):
        return self._name if self._name is not None else type(self).__name__

    def options(self):
        return {"name": self._name,
                "unknown": getattr(self.unknown, "name", repr(self.unknown))
                if self.unknown is not None else None,
                "n_inputs": len(self.inputs),
                "n_bcs": len(self.bcs),
                "has_coefficients": self.coefficients is not None,
                "has_nullspace": self.nullspace is not None,
                "solver": getattr(self.solver, "name", self.solver),
                "has_cadence": self.cadence is not None}

    def requirements(self):
        req = {"equation": True, "solver": True}
        if self.nullspace is not None:
            req["nullspace"] = getattr(self.nullspace, "name", repr(self.nullspace))
        return req

    def capabilities(self):
        return {"elliptic": True}

    def available(self, context=None):
        if not isinstance(self.equation, Equation):
            return Availability.no(
                "%s needs a pops.math.Equation; got %r" % (self.name, type(self.equation).__name__),
                missing=["equation"])
        if self.solver is None:
            return Availability.no("%s needs a solver" % self.name, missing=["solver"])
        return Availability.yes()

    def validate(self, context=None):
        if isinstance(self.equation, bool):
            raise TypeError(
                "%s: equation must be a pops.math.Equation built from PoPS operators "
                "(e.g. -laplacian(phi) == rhs); got a Python bool, which usually means '==' "
                "was applied to plain values instead of PoPS expressions" % self.name)
        if not isinstance(self.equation, Equation):
            raise TypeError(
                "%s: equation must be a pops.math.Equation; got %r"
                % (self.name, type(self.equation).__name__))
        if self.solver is None:
            raise ValueError("%s: a solver must be provided" % self.name)
        self._require_periodic_compatible_solver()
        self._require_layout_compatible_solver(context)
        self._require_declared_nullspace(context)
        self._require_declared_outputs(context)
        return True

    def _require_declared_nullspace(self, context):
        """Refuse a singular field solve that declares no nullspace (Spec 5 sec.7, criterion 11).

        A pure-Neumann / fully periodic elliptic operator is SINGULAR: its solution is defined up to
        an additive constant and the solver must project the constant mode out (a
        :class:`~pops.fields.nullspace.ConstantNullspace`). When the route @p context flags the
        operator singular (``{"requires_nullspace": True}``) and the problem declares NO nullspace,
        refuse before the runtime hits an inconsistent system. Opt-in via the context flag, so a
        problem whose singularity is not known is never falsely rejected.
        """
        if not self._context_flag(context, "requires_nullspace"):
            return
        if self.nullspace is None:
            raise ValueError(
                "%s: the elliptic operator is singular (pure-Neumann / periodic) and needs a "
                "nullspace projection; declare nullspace=pops.fields.ConstantNullspace()."
                % self.name)

    def _require_declared_outputs(self, context):
        """Refuse a solve that does not expose a required derived output (Spec 5 sec.9).

        When the route @p context names outputs the downstream stage NEEDS
        (``{"required_outputs": ["E"]}``) but the problem's declared ``outputs`` do not cover them,
        refuse with the missing name so the gap is caught before a stage reads a field that was
        never produced. Opt-in via the context flag (no false positive on an unspecified context).
        """
        required = self._context_value(context, "required_outputs")
        if not required:
            return
        produced = self._declared_output_names()
        missing = [name for name in required if name not in produced]
        if missing:
            raise ValueError(
                "%s: the field solve does not declare the required output(s) %s; add a typed "
                "pops.fields.outputs descriptor (FieldOutput / GradientOutput / DerivedField) "
                "for each." % (self.name, missing))

    def _declared_output_names(self):
        """The set of names this problem's ``outputs`` expose (a single output or an iterable)."""
        outputs = self.outputs
        if outputs is None:
            return set()
        items = outputs if isinstance(outputs, (list, tuple, set)) else [outputs]
        names = set()
        for item in items:
            name = getattr(item, "name", None)
            if name is not None:
                names.add(name)
        return names

    @staticmethod
    def _context_flag(context, key):
        """True when @p context (a dict or attribute-bearing object) sets @p key truthy."""
        return bool(FieldProblem._context_value(context, key))

    @staticmethod
    def _context_value(context, key):
        """The value @p context carries under @p key (dict or attribute), or ``None``."""
        if context is None:
            return None
        if isinstance(context, dict):
            return context.get(key)
        return getattr(context, key, None)

    @staticmethod
    def _context_is_amr_layout(context):
        """True when @p context names an AMR mesh layout (duck-typed; no mesh import).

        ``Problem.validate`` passes the layout under the ``"layout"`` key; a layout advertises its
        kind through ``capabilities()["layout"]`` (``"amr"`` / ``"uniform"``), so AMR is detected
        without importing :mod:`pops.mesh` into the fields layer. Defensive: a context with no
        layout, or a layout whose ``capabilities`` is absent / non-callable / raises, is "not AMR".
        """
        if context is None:
            return False
        layout = context.get("layout") if isinstance(context, dict) else getattr(context, "layout",
                                                                                 None)
        if layout is None:
            layout = context  # the context may itself be the layout descriptor
        caps = getattr(layout, "capabilities", None)
        if not callable(caps):
            return False
        try:
            declared = caps()
        except Exception:
            return False
        return isinstance(declared, dict) and declared.get("layout") == "amr"

    def _require_layout_compatible_solver(self, context):
        """Refuse a solver that cannot serve an AMR mesh layout (Spec 6 sec.8/9, #11).

        SCOPED to an AMR layout -- the only mesh structure a field solver currently refuses (the
        spectral FFT needs a uniform periodic mesh, not a refined hierarchy). On an AMR route the
        solver answers through its own ``available(context)`` so the message stays the solver's
        PRECISE one (FFT names ``Uniform(periodic=True)`` / ``GeometricMG()``). Same
        NO-FALSE-POSITIVE discipline as :meth:`_require_periodic_compatible_solver`:

        * a non-AMR (Uniform / unspecified) route -> NOT checked, so a solver that returns a hard
          ``no`` for a NON-layout reason (e.g. an unresolved external brick) is not refused here;
        * a solver with no callable ``available`` (a bare ``BrickDescriptor``, bool ``available``)
          -> not checked;
        * a solver whose ``available`` RAISES on the layout-only context -> not a KNOWN
          incompatibility, so validate never propagates an unexpected exception;
        * only a hard ``no`` on the AMR route is refused, surfacing the solver's own reason.
        """
        if not self._context_is_amr_layout(context):
            return
        available = getattr(self.solver, "available", None)
        if not callable(available):
            return
        try:
            status = available(context)
        except Exception:
            return
        if getattr(status, "status", None) != "no":
            return
        solver_name = getattr(self.solver, "name", type(self.solver).__name__)
        reason = getattr(status, "reason", "") or ("solver %s cannot serve an AMR layout"
                                                    % solver_name)
        raise ValueError("%s: %s" % (self.name, reason))

    @staticmethod
    def _bc_kind(bc):
        """The declared boundary kind ("periodic"/"dirichlet"/...) of a field BC, or None.

        Reads the BC's own ``options()["bc"]``; a :class:`~pops.fields.bcs.FaceBC` is unwrapped
        to the condition it binds. Returns ``None`` for an object that does not declare a kind,
        so an unrecognized BC never triggers a (false) rejection.
        """
        inner = getattr(bc, "condition", bc)  # FaceBC(face, condition) -> the condition
        opts = getattr(inner, "options", None)
        if callable(opts):
            declared = opts()
            if isinstance(declared, dict):
                return declared.get("bc")
        return None

    def _require_periodic_compatible_solver(self):
        """Reject a periodic-only solver paired with a non-periodic boundary (Spec 5 sec.7, #11).

        The FFT Poisson solver only serves a periodic boundary (``supports_wall_bc`` False,
        ``supports_periodic_bc`` True); pairing it with a Dirichlet / Neumann / extrapolation BC
        is a known incompatibility the spec wants caught BEFORE execution. Same NO-FALSE-POSITIVE
        discipline as :meth:`_require_solver_capability`:

        * no ``bcs`` declared -> the boundary is unspecified (may default periodic): never reject;
        * a solver with no ``capabilities()`` dict -> capability absent, not declared: never reject;
        * a solver that is not periodic-only (``supports_wall_bc`` is not literally False, e.g.
          ``GeometricMG`` which serves walls) -> never reject;
        * only a periodic-only solver AND a KNOWN non-periodic BC is refused, naming GeometricMG.
        """
        if not self.bcs:
            return
        caps = getattr(self.solver, "capabilities", None)
        if not callable(caps):
            return
        declared = caps()
        if not isinstance(declared, dict):
            return
        periodic_only = (declared.get("supports_wall_bc") is False
                         and declared.get("supports_periodic_bc") is True)
        if not periodic_only:
            return
        if any(self._bc_kind(bc) not in (None, "periodic") for bc in self.bcs):
            solver_name = getattr(self.solver, "name", type(self.solver).__name__)
            raise ValueError(
                "%s: solver %s requires a periodic boundary (supports_wall_bc is False) but the "
                "problem declares a non-periodic boundary; use a periodic BC "
                "(pops.fields.bcs.Periodic()) or pops.solvers.elliptic.GeometricMG()."
                % (self.name, solver_name))

    def _require_solver_capability(self, tag, operator, alternative):
        """Reject the solver only when it declares ``supports_<tag>`` KNOWN-False (Spec 5 sec.7).

        The Poisson-family subclasses call this so a problem whose operator needs a special
        capability (a screened reaction term, an anisotropic / variable coefficient) is not
        paired with a solver that cannot serve it -- the spec's pre-runtime incompatible-solver
        check (criterion 11). The OVERRIDING discipline is NO FALSE POSITIVE:

        * a solver that exposes no ``capabilities()`` dict (a bare object, an external brick) is
          NOT rejected -- the capability is ABSENT, not declared-False;
        * a solver that declares ``supports_variable_epsilon`` True is NOT rejected -- a
          variable-coefficient elliptic kernel (``GeometricMG``) subsumes a screened reaction
          term and an anisotropic-by-coefficient operator, so its ``supports_<tag>=False`` is
          not a real incompatibility (this is exactly why ``GeometricMG`` is the recommended
          alternative for both);
        * only a solver that declares ``supports_<tag>`` literally ``False`` AND does not
          declare variable epsilon is refused, with a clear message naming the operator and the
          typed alternative.

        Args:
            tag: The capability tag the operator needs (``"screened"`` / ``"anisotropic"``).
            operator: A human phrase for the operator ("a screened operator").
            alternative: The typed solver to recommend ("GeometricMG()").

        Raises:
            ValueError: When the chosen solver declares the capability KNOWN-False.
        """
        caps = getattr(self.solver, "capabilities", None)
        if not callable(caps):
            return  # no capability surface -> absent, not declared-False: never reject.
        declared = caps()
        if not isinstance(declared, dict):
            return
        # A variable-coefficient elliptic kernel serves screened / anisotropic-by-coefficient.
        if declared.get("supports_variable_epsilon") is True:
            return
        if declared.get("supports_%s" % tag) is False:
            solver_name = getattr(self.solver, "name", type(self.solver).__name__)
            raise ValueError(
                "%s: solver %s does not support %s (supports_%s is False); "
                "use pops.solvers.elliptic.%s."
                % (self.name, solver_name, operator, tag, alternative))

    def solve(self, schedule, policy):
        """Record an inert field-solve cadence (a schedule + a not-due policy).

        Pairs a typed :class:`pops.time.Schedule` (WHEN to solve -- e.g. ``every(4)`` to
        solve every fourth macro-step, or ``when(cond)``) with a typed
        :class:`~pops.fields.policies.FieldSolvePolicy`
        (:class:`~pops.fields.policies.HoldPrevious` /
        :class:`~pops.fields.policies.Recompute`) deciding what happens on a step where the
        solve is not due. The pair is stored as :attr:`cadence` and surfaced by
        :meth:`inspect`; the method returns ``self`` so it chains after construction.

        This is AUTHORING metadata only. It deliberately does NOT lower the cadence into the
        Program / codegen: honouring a non-trivial field-solve cadence at runtime (the cached
        field carry, the residual gate) is the codegen-adjacent follow-up. The cadence is
        recorded and inspectable, never silently executed.

        Args:
            schedule: A typed :class:`pops.time.Schedule` (built with ``every`` / ``when`` /
                ``always`` / ...). A bare ``int`` or ``str`` is rejected with a clear
                ``TypeError`` -- the cadence is a typed object, not a free string or count.
            policy: A typed :class:`~pops.fields.policies.FieldSolvePolicy`
                (:class:`~pops.fields.policies.HoldPrevious` or
                :class:`~pops.fields.policies.Recompute`). A string such as ``"hold"`` is
                rejected with a clear ``TypeError``.

        Returns:
            FieldProblem: ``self``, so the call chains after construction.

        Raises:
            TypeError: When @p schedule is not a :class:`pops.time.Schedule` or @p policy is
                not a :class:`~pops.fields.policies.FieldSolvePolicy`.
        """
        # Lazy import: keep pops.fields free of a module-scope pops.time edge (the acyclic
        # layering of test_import_graph; fields.aux defers its mesh import the same way).
        from pops.time.schedule import Schedule

        if not isinstance(schedule, Schedule):
            raise TypeError(
                "%s.solve(schedule=...): schedule must be a typed pops.time.Schedule "
                "(e.g. pops.time.every(4) or pops.time.when(cond)); got %r. A bare int / "
                "string is not a cadence." % (self.name, schedule))
        if not isinstance(policy, FieldSolvePolicy):
            raise TypeError(
                "%s.solve(policy=...): policy must be a typed field-solve policy "
                "(pops.fields.HoldPrevious() or pops.fields.Recompute()); got %r"
                % (self.name, policy))
        self.cadence = SolveCadence(schedule, policy)
        return self

    def inspect(self):
        info = super().inspect()
        info["equation"] = self._equation_summary()
        info["bcs"] = [getattr(b, "name", repr(b)) for b in self.bcs]
        info["outputs"] = getattr(self.outputs, "name", self.outputs)
        info["cadence"] = self.cadence.inspect() if self.cadence is not None else None
        return info

    def _equation_summary(self):
        if not isinstance(self.equation, Equation):
            return repr(self.equation)
        return "%s == %s" % (_summarize(self.equation.lhs), _summarize(self.equation.rhs))

    def __str__(self):
        solver = getattr(self.solver, "name", self.solver)
        outputs = getattr(self.outputs, "name", self.outputs)
        return "%s [%s] %s | bcs=%d | solver=%s | outputs=%s" % (
            self.name, self.category, self._equation_summary(), len(self.bcs), solver, outputs)


__all__ = ["FieldProblem", "SolveCadence", "lower_field_solver"]
