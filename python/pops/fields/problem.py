"""pops.fields.problem -- the typed elliptic field-problem descriptor (Spec 5 sec.5.5).

:class:`FieldProblem` is the typed declaration of an elliptic solve. It validates its
equation, solver and lowerable behavior before codegen/runtime consume the descriptor.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability, Descriptor, reject_string_selector
from pops.math import Equation

from ._references import collect_references, reference_label, resolve_handle, resolve_value
from ._solver_lowering import lower_field_solver
from ._validation_context import context_flag, context_is_amr_layout, context_value
from .policies import FieldSolvePolicy


def _summarize(side: Any, limit: Any = 60) -> str:
    """A short repr of one equation side, truncated to keep the print() summary small."""
    text = repr(side)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class SolveCadence(Descriptor):
    """A field-solve schedule and its typed not-due policy.

    Nontrivial instances remain inspectable but are rejected at the ADC-659 lowering gate.
    """

    category = "solve_cadence"

    def __init__(self, schedule: Any, policy: Any) -> None:
        self.schedule = schedule
        self.policy = policy

    def options(self) -> dict:
        return {"schedule": repr(self.schedule), "policy": self.policy.name}

    def requirements(self) -> Any:
        return self.policy.requirements()

    def capabilities(self) -> Any:
        return self.policy.capabilities()

    def inspect(self) -> dict:
        info = super().inspect()
        info["schedule"] = repr(self.schedule)
        info["policy"] = self.policy.inspect()
        return info

    def __repr__(self) -> str:
        return "SolveCadence(schedule=%r, policy=%r)" % (self.schedule, self.policy)


class FieldProblem(Descriptor):
    """A typed elliptic field problem: an unknown solved from an :class:`~pops.math.Equation`.

    ``FieldProblem(unknown=phi, equation=(-laplacian(phi) == rhs), solver=GeometricMG())``.
    Its remaining fields are typed descriptors stored for validation and lowering.
    """

    category = "field_problem"

    def __init__(self, name: Any = None, unknown: Any = None, equation: Any = None,
                 inputs: Any = (), coefficients: Any = None, bcs: Any = (),
                 nullspace: Any = None, outputs: Any = None, postprocess: Any = None,
                 solver: Any = None) -> None:
        self._name = None if name is None else str(name)
        self.unknown = unknown
        self.equation = equation
        from pops.model import Handle
        self.inputs = tuple(inputs)
        invalid_inputs = [item for item in self.inputs if not isinstance(item, Handle)]
        if invalid_inputs:
            raise TypeError(
                "FieldProblem inputs must be declaration Handle values; names/strings are not "
                "references (got %r)" % type(invalid_inputs[0]).__name__)
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
    def name(self) -> str:
        return self._name if self._name is not None else type(self).__name__

    def options(self) -> dict:
        references = [
            reference_label(reference, where="FieldProblem reference")
            for reference in self.declaration_references()]
        return {"name": self._name,
                "unknown": getattr(self.unknown, "name", repr(self.unknown))
                if self.unknown is not None else None,
                "n_inputs": len(self.inputs),
                "n_bcs": len(self.bcs),
                "has_coefficients": self.coefficients is not None,
                "has_nullspace": self.nullspace is not None,
                "solver": getattr(self.solver, "name", self.solver),
                "has_cadence": self.cadence is not None,
                "references": references}

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        req: dict = {"equation": True, "solver": True}
        references = [
            reference_label(reference, where="FieldProblem reference")
            for reference in self.declaration_references()]
        if references:
            req["declaration_references"] = references
        if self.nullspace is not None:
            req["nullspace"] = getattr(self.nullspace, "name", repr(self.nullspace))
        return RequirementSet(req)

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet
        return CapabilitySet({"elliptic": True})

    def available(self, context: Any = None) -> Any:
        """Explain whether this field problem can lower: needs an Equation and a solver."""
        if not isinstance(self.equation, Equation):
            return Availability.no(
                "%s needs a pops.math.Equation; got %r" % (self.name, type(self.equation).__name__),
                missing=["equation"])
        if self.solver is None:
            return Availability.no("%s needs a solver" % self.name, missing=["solver"])
        if self._has_nontrivial_cadence():
            return Availability.no(
                "%s has a nontrivial field cadence that codegen does not lower (ADC-659)"
                % self.name,
                missing=["field_cadence_lowering"])
        return Availability.yes()

    def validate(self, context: Any = None) -> bool:
        """Refuse a malformed field problem loudly (non-Equation, missing solver, bad cadence)."""
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
        self._require_lowerable_cadence()
        self._require_periodic_compatible_solver()
        self._require_layout_compatible_solver(context)
        self._require_declared_nullspace(context)
        self._require_owned_references(context)
        self._require_declared_outputs(context)
        return True

    def _has_nontrivial_cadence(self) -> bool:
        """Whether the authored cadence changes behavior from solve-every-step."""
        if self.cadence is None:
            return False
        schedule_is_always = getattr(self.cadence.schedule, "is_always", None)
        schedule_is_trivial = bool(schedule_is_always()) if callable(schedule_is_always) else False
        from .policies import Recompute
        return not (schedule_is_trivial and isinstance(self.cadence.policy, Recompute))

    def _require_lowerable_cadence(self) -> None:
        """Reject cadence semantics until a runtime lowering exists (ADC-659)."""
        if not self._has_nontrivial_cadence():
            return
        from pops.codegen.lowering_coverage import (
            LoweringCoverageReport,
            LoweringCoverageRow,
            LoweringRejection,
        )

        source = "field_problem:%s:cadence" % self.name
        gate = "field_cadence_not_lowered"
        report = LoweringCoverageReport((LoweringCoverageRow(
            "field_problem:%s:metadata" % self.name, "documentary"),
            LoweringCoverageRow(source, "rejected", gate=gate),
        ))
        raise LoweringRejection(
            "%s: ADC-659 rejects nontrivial field cadence %r with policy %r because field "
            "cadence has no Program/codegen lowering; accepting it would create inert schedule "
            "semantics" % (self.name, self.cadence.schedule, self.cadence.policy),
            coverage_report=report, source=source, gate=gate)

    def lower(self, context: Any = None) -> Any:
        """Lower descriptor metadata only after rejecting unimplemented cadence behavior."""
        self._require_lowerable_cadence()
        return super().lower(context)

    def resolve_references(self, resolver: Any) -> FieldProblem:
        """Return a detached field problem with every declaration reference authenticated."""
        from copy import copy
        from pops.model import Handle

        resolved = copy(self)
        if isinstance(self.unknown, Handle):
            resolved.unknown = resolve_handle(
                self.unknown, resolver, where="FieldProblem unknown")
        resolved.inputs = tuple(
            resolve_handle(reference, resolver, where="FieldProblem inputs[%d]" % index)
            for index, reference in enumerate(self.inputs))
        resolved.coefficients = resolve_value(
            self.coefficients, resolver, where="FieldProblem coefficients")
        resolved.outputs = resolve_value(
            self.outputs, resolver, where="FieldProblem outputs")
        resolved.postprocess = resolve_value(
            self.postprocess, resolver, where="FieldProblem postprocess")
        resolved.equation = self._resolved_equation(resolver)
        return resolved

    def declaration_references(self) -> tuple[Any, ...]:
        values = (
            self.unknown,
            self.inputs,
            self.coefficients,
            getattr(self.equation, "lhs", None),
            getattr(self.equation, "rhs", None),
            self.outputs,
            self.postprocess,
        )
        return collect_references(values)

    def _resolved_equation(self, resolver: Any) -> Any:
        if not isinstance(self.equation, Equation):
            return self.equation

        def side(value: Any, where: str) -> Any:
            from pops.ir import Expr
            from pops.model import Handle

            if isinstance(value, Handle):
                return resolve_handle(value, resolver, where=where)
            if isinstance(value, Expr):
                references = value.declaration_references()
                return value.resolve_references(resolver) if references else value
            protocol = getattr(value, "resolve_references", None)
            return protocol(resolver) if callable(protocol) else value

        return Equation(
            side(self.equation.lhs, "FieldProblem equation.lhs"),
            side(self.equation.rhs, "FieldProblem equation.rhs"),
        )

    def _require_owned_references(self, context: Any) -> None:
        """Authenticate every retained declaration without mutating this authoring descriptor."""
        resolver = context_value(context, "declaration_resolver")
        if callable(resolver):
            self.resolve_references(resolver)

        outputs = self.outputs
        output_items = (outputs.values() if isinstance(outputs, dict)
                        else outputs if isinstance(outputs, (list, tuple, set))
                        else () if outputs is None else (outputs,))
        seen = set()
        for output in output_items:
            name = getattr(output, "name", None)
            if name is not None:
                if name in seen:
                    raise ValueError(
                        "%s: duplicate field output declaration %r" % (self.name, name))
                seen.add(name)

    def _require_declared_nullspace(self, context: Any) -> None:
        """Refuse a singular field solve that declares no nullspace (Spec 5 sec.7, criterion 11).

        A pure-Neumann / fully periodic elliptic operator is SINGULAR: its solution is defined up to
        an additive constant and the solver must project the constant mode out (a
        :class:`~pops.fields.nullspace.ConstantNullspace`). When the route @p context flags the
        operator singular (``{"requires_nullspace": True}``) and the problem declares NO nullspace,
        refuse before the runtime hits an inconsistent system. Opt-in via the context flag, so a
        problem whose singularity is not known is never falsely rejected.
        """
        if not context_flag(context, "requires_nullspace"):
            return
        if self.nullspace is None:
            raise ValueError(
                "%s: the elliptic operator is singular (pure-Neumann / periodic) and needs a "
                "nullspace projection; declare nullspace=pops.fields.ConstantNullspace()."
                % self.name)

    def _require_declared_outputs(self, context: Any) -> None:
        """Refuse a solve that does not expose a required derived output (Spec 5 sec.9).

        When the route @p context names outputs the downstream stage NEEDS
        (``{"required_outputs": ["E"]}``) but the problem's declared ``outputs`` do not cover them,
        refuse with the missing name so the gap is caught before a stage reads a field that was
        never produced. Opt-in via the context flag (no false positive on an unspecified context).
        """
        required = context_value(context, "required_outputs")
        if not required:
            return
        produced = self._declared_output_names()
        missing = [name for name in required if name not in produced]
        if missing:
            raise ValueError(
                "%s: the field solve does not declare the required output(s) %s; add a typed "
                "pops.fields.outputs descriptor (FieldOutput / GradientOutput / DerivedField) "
                "for each." % (self.name, missing))

    def _declared_output_names(self) -> set:
        """The set of names this problem's ``outputs`` expose (a single output or an iterable)."""
        outputs = self.outputs
        if outputs is None:
            return set()
        items = outputs if isinstance(outputs, (list, tuple, set)) else [outputs]
        names = set()
        for item in items:
            name = getattr(item, "name", None)
            if name is not None:
                if name in names:
                    raise ValueError(
                        "%s: duplicate field output declaration %r" % (self.name, name))
                names.add(name)
        return names

    def _require_layout_compatible_solver(self, context: Any) -> None:
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
        if not context_is_amr_layout(context):
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
    def _bc_kind(bc: Any) -> Any:
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

    def _require_periodic_compatible_solver(self) -> None:
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
        declared: Any = caps()
        # ``declared`` is a typed CapabilitySet (or a plain dict): both expose ``.get`` (ADC-625).
        if not hasattr(declared, "get"):
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

    def _require_solver_capability(self, tag: Any, operator: Any, alternative: Any) -> None:
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
        declared: Any = caps()
        # ``declared`` is a typed CapabilitySet (or a plain dict): both expose ``.get`` (ADC-625).
        if not hasattr(declared, "get"):
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

    def solve(self, schedule: Any, policy: Any) -> FieldProblem:
        """Record a typed schedule/policy pair and return ``self``.

        Bare values are rejected. Nontrivial typed cadence stays inspectable but validate/lower
        rejects it until codegen implements its cached-field and residual-gate behavior.
        """
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

    def inspect(self) -> dict:
        info = super().inspect()
        info["equation"] = self._equation_summary()
        info["bcs"] = [getattr(b, "name", repr(b)) for b in self.bcs]
        info["outputs"] = getattr(self.outputs, "name", self.outputs)
        info["cadence"] = self.cadence.inspect() if self.cadence is not None else None
        return info

    def _equation_summary(self) -> str:
        if not isinstance(self.equation, Equation):
            return repr(self.equation)
        return "%s == %s" % (_summarize(self.equation.lhs), _summarize(self.equation.rhs))

    def __str__(self) -> str:
        solver = getattr(self.solver, "name", self.solver)
        outputs = getattr(self.outputs, "name", self.outputs)
        return "%s [%s] %s | bcs=%d | solver=%s | outputs=%s" % (
            self.name, self.category, self._equation_summary(), len(self.bcs), solver, outputs)


__all__ = ["FieldProblem", "SolveCadence", "lower_field_solver"]
