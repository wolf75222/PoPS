"""Runtime-consumer and structural-constraint registries for a Problem."""
from __future__ import annotations

from typing import Any

from pops.model import Handle
from pops.model.ownership import (
    AmbiguousReferenceError,
    DoubleOwnershipError,
    MissingOwnershipError,
)
from pops.problem._registry_freeze import (
    FreezableRegistry as _FreezableRegistry,
    flatten_freeze_members,
)
from pops.problem._registry_support import descriptor_declaration_key, strict_name
from pops._report import ReportTree


class RuntimePolicyRegistry(_FreezableRegistry):
    """Static aux inputs plus output, checkpoint, and diagnostic consumers."""

    family = "runtime"
    _POLICY_CATEGORIES = ("output_policy", "checkpoint_policy")
    _OUTPUT_FIELD_KINDS = frozenset({"aux", "field", "state"})

    @staticmethod
    def _is_diagnostic_category(category: Any) -> bool:
        return isinstance(category, str) and (
            category.startswith("diagnostic_") or category == "conservation_check"
        )

    def __init__(self) -> None:
        self._aux = {}
        self._outputs = []
        self._diagnostics = []
        self._schedules = []
        self._bundle_declared = False

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(
            self._outputs, self._diagnostics, self._schedules)

    def add_aux(self, name: Any, value: Any = None) -> None:
        self._guard_frozen("declare an aux input")
        key = strict_name(name, "aux name")
        if key in self._aux:
            raise ValueError(
                "aux input %r is already declared; aux declarations are register-once" % key
            )
        self._aux[key] = value

    def add_output(self, policy: Any) -> None:
        self._guard_frozen("attach an output policy")
        key = descriptor_declaration_key(policy)
        if key in {descriptor_declaration_key(item) for item in self._outputs}:
            raise ValueError(
                "runtime output declaration %r is already registered"
                % getattr(policy, "name", type(policy).__name__)
            )
        self._outputs.append(policy)

    def add_diagnostic(self, measure: Any) -> None:
        self._guard_frozen("attach a diagnostic measure")
        key = descriptor_declaration_key(measure)
        if key in {descriptor_declaration_key(item) for item in self._diagnostics}:
            raise ValueError(
                "runtime diagnostic declaration %r is already registered"
                % getattr(measure, "name", type(measure).__name__)
            )
        self._diagnostics.append(measure)

    def set_policies(self, policies: Any) -> None:
        """Record a typed RuntimePolicies bundle transactionally and exactly once."""
        self._guard_frozen("attach runtime policies")
        from pops.output.runtime_policies import RuntimePolicies

        if not isinstance(policies, RuntimePolicies):
            raise TypeError(
                "problem.runtime(...) expects a typed pops.RuntimePolicies bundle; got %r. "
                "Group the runtime concerns with pops.RuntimePolicies(output=..., "
                "checkpoint=..., diagnostics=..., schedules=...)."
                % type(policies).__name__
            )
        if self._bundle_declared:
            raise ValueError(
                "runtime policies are already declared; the RuntimePolicies bundle is register-once"
            )
        outputs = policies.outputs()
        schedules = list(policies.schedules)
        output_keys = [descriptor_declaration_key(policy) for policy in outputs]
        diagnostic_keys = [
            descriptor_declaration_key(measure) for measure in policies.diagnostics
        ]
        if len(set(output_keys)) != len(output_keys):
            raise ValueError("RuntimePolicies contains a duplicate output declaration")
        if len(set(diagnostic_keys)) != len(diagnostic_keys):
            raise ValueError("RuntimePolicies contains a duplicate diagnostic declaration")
        schedule_keys = [descriptor_declaration_key(schedule) for schedule in schedules]
        if len(set(schedule_keys)) != len(schedule_keys):
            raise ValueError("RuntimePolicies contains a duplicate schedule declaration")
        if set(output_keys).intersection(
            descriptor_declaration_key(policy) for policy in self._outputs
        ):
            raise ValueError(
                "RuntimePolicies repeats an output declaration already registered on the problem"
            )
        if set(diagnostic_keys).intersection(
            descriptor_declaration_key(measure) for measure in self._diagnostics
        ):
            raise ValueError(
                "RuntimePolicies repeats a diagnostic declaration already registered on the problem"
            )
        self._bundle_declared = True
        self._outputs.extend(outputs)
        self._diagnostics.extend(policies.diagnostics)
        self._schedules.extend(schedules)

    @property
    def bundle_declared(self) -> bool:
        return self._bundle_declared

    @property
    def aux(self) -> Any:
        return dict(self._aux)

    @property
    def outputs(self) -> Any:
        return list(self._outputs)

    @property
    def diagnostics(self) -> Any:
        return list(self._diagnostics)

    @property
    def schedules(self) -> Any:
        return list(self._schedules)

    def names(self) -> Any:
        return sorted(self._aux) + [getattr(policy, "name", repr(policy)) for policy in self._outputs]

    def __iter__(self) -> Any:
        return iter(self._outputs)

    def validate(self, context: Any = None) -> Any:
        report = ReportTree(
            phase="validation", severity="info", code="validation.runtime.root",
            source=self.family)
        resolver = context.get("declaration_resolver") if isinstance(context, dict) else None
        for policy in self._outputs:
            category = getattr(policy, "category", None)
            if category not in self._POLICY_CATEGORIES:
                report = report.error(
                    self.family,
                    "bad_output_policy",
                    "output() expects a pops.output.OutputPolicy / CheckpointPolicy; got %r "
                    "(category %r)" % (type(policy).__name__, category),
                    context={"policy": type(policy).__name__, "category": category},
                )
                continue
            if category == "output_policy":
                for index, reference in enumerate(getattr(policy, "fields", ())):
                    report = self._validate_consumer_reference(
                        report,
                        resolver,
                        reference,
                        consumer="%s.fields[%d]" % (type(policy).__name__, index),
                        allowed_kinds=self._OUTPUT_FIELD_KINDS,
                    )
                for index, measure in enumerate(getattr(policy, "diagnostics", ())):
                    report = self._validate_measure_reference(
                        report,
                        resolver,
                        measure,
                        consumer="%s.diagnostics[%d]" % (type(policy).__name__, index),
                    )
        for measure in self._diagnostics:
            category = getattr(measure, "category", None)
            if not self._is_diagnostic_category(category):
                report = report.error(
                    self.family,
                    "bad_diagnostic_measure",
                    "diagnostics=[...] expects a pops.diagnostics measure; got %r (category %r)"
                    % (type(measure).__name__, category),
                    context={"measure": type(measure).__name__, "category": category},
                )
                continue
            report = self._validate_measure_reference(
                report, resolver, measure, consumer=type(measure).__name__
            )
        return self._validate_flat_members(report, context)

    def _validate_flat_members(self, report: ReportTree, context: Any) -> ReportTree:
        """Validate the sole flat declarations; the input bundle is never retained."""
        members = [*self._outputs, *self._diagnostics, *self._schedules]
        ctx = context or {}
        requirements = {}
        if self._schedules:
            report = report.error(
                "runtime_policies", "unattached_schedule",
                "the runtime registry contains %d unattached schedule(s); every schedule must be "
                "owned by a Program node, output, checkpoint, or diagnostic consumer"
                % len(self._schedules),
                context={"count": len(self._schedules)},
                alternatives=("attach each schedule to its consumer",),
            )
        for member in members:
            validate = getattr(member, "validate", None)
            if callable(validate):
                try:
                    validate(ctx)
                except Exception as exc:  # noqa: BLE001 -- aggregate member diagnostics
                    report = report.error(
                        self.family,
                        "runtime_member_invalid",
                        str(exc),
                        context={"member": type(member).__name__},
                    )
            requirement_provider = getattr(member, "requirements", None)
            if callable(requirement_provider):
                for key, value in requirement_provider().to_dict().items():
                    if key in requirements and requirements[key] != value:
                        report = report.error(
                            "runtime_policies", "requirement_conflict",
                            "runtime policy requirement %r is declared as both %r and %r; "
                            "requirement unions cannot overwrite earlier evidence"
                            % (key, requirements[key], value),
                            context={"requirement": key, "first": requirements[key],
                                     "second": value},
                        )
                    else:
                        requirements[key] = value
        if requirements.get("parallel_io") and isinstance(ctx, dict):
            declared = [key for key in ("parallel", "mpi", "supports_mpi") if key in ctx]
            if declared and not any(bool(ctx.get(key)) for key in declared):
                report = report.error(
                    "runtime_policies",
                    "incompatible_policy",
                    "a runtime policy requires 'parallel_io' but the resolved runtime context "
                    "declares a serial / non-MPI backend (%s)"
                    % ", ".join("%s=%r" % (key, ctx.get(key)) for key in declared),
                    context={"requirement": "parallel_io"},
                )
        return report

    def _validate_consumer_reference(
        self,
        report: ReportTree,
        resolver: Any,
        reference: Any,
        *,
        consumer: str,
        allowed_kinds: Any = None,
    ) -> ReportTree:
        if not isinstance(reference, Handle):
            return report.error(
                self.family,
                "untyped_declaration_reference",
                "%s must reference a declaration Handle, not %r"
                % (consumer, type(reference).__name__),
                context={"consumer": consumer, "reference_type": type(reference).__name__},
            )
        if allowed_kinds is not None and reference.kind not in allowed_kinds:
            return report.error(
                self.family,
                "invalid_output_field_kind",
                "%s accepts only writable state/field/aux handles; got kind %r"
                % (consumer, reference.kind),
                context={
                    "consumer": consumer,
                    "qualified_id": reference.qualified_id,
                    "kind": reference.kind,
                },
            )
        if not callable(resolver):
            return report
        try:
            resolver(reference)
        except AmbiguousReferenceError as exc:
            report = report.error(
                self.family,
                "ambiguous_declaration_reference",
                "%s: %s" % (consumer, exc),
                context={"consumer": consumer, "qualified_id": reference.qualified_id},
            )
        except (MissingOwnershipError, DoubleOwnershipError, TypeError, ValueError) as exc:
            report = report.error(
                self.family,
                "invalid_declaration_reference",
                "%s: %s" % (consumer, exc),
                context={"consumer": consumer, "qualified_id": reference.qualified_id},
            )
        return report

    def _validate_measure_reference(
        self,
        report: ReportTree,
        resolver: Any,
        measure: Any,
        *,
        consumer: str,
    ) -> ReportTree:
        category = getattr(measure, "category", None)
        if not self._is_diagnostic_category(category):
            return report.error(
                self.family,
                "bad_diagnostic_measure",
                "%s must reference a typed pops.diagnostics measure, not %r (category %r)"
                % (consumer, type(measure).__name__, category),
                context={"consumer": consumer, "category": category},
            )
        if category == "conservation_check":
            return self._validate_measure_reference(
                report,
                resolver,
                getattr(measure, "quantity", None),
                consumer="%s.quantity" % consumer,
            )
        block = getattr(measure, "block", None)
        if block is None:
            return report
        from pops.problem.handles import BlockHandle

        if not isinstance(block, BlockHandle):
            return report.error(
                self.family,
                "untyped_block_reference",
                "%s.block must be a BlockHandle, not %r" % (consumer, type(block).__name__),
                context={"consumer": consumer, "reference_type": type(block).__name__},
            )
        return self._validate_consumer_reference(
            report, resolver, block, consumer="%s.block" % consumer)

    def inspect(self) -> Any:
        info = {
            "aux": sorted(self._aux),
            "outputs": [getattr(policy, "name", repr(policy)) for policy in self._outputs],
            "diagnostics": [
                getattr(measure, "name", repr(measure)) for measure in self._diagnostics
            ],
            "schedules": [getattr(schedule, "name", repr(schedule))
                          for schedule in self._schedules],
            "bundle_declared": self._bundle_declared,
        }
        return info


class ConstraintRegistry(_FreezableRegistry):
    """Layout-free AMR refinement criteria owned only by the Problem."""

    family = "amr"

    def __init__(self) -> None:
        self._criteria = {}

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(self._criteria)

    def set_refinement(
        self,
        *,
        refine: Any = None,
        regrid: Any = None,
        nesting: Any = None,
        patches: Any = None,
    ) -> None:
        self._guard_frozen("record AMR refinement criteria")
        requested = {
            key: value
            for key, value in (
                ("refine", refine),
                ("regrid", regrid),
                ("nesting", nesting),
                ("patches", patches),
            )
            if value is not None
        }
        duplicates = sorted(set(requested).intersection(self._criteria))
        if duplicates:
            raise ValueError(
                "AMR refinement criteria already declared for: %s; criteria are register-once"
                % ", ".join(duplicates)
            )
        self._criteria.update(requested)

    @property
    def refinement(self) -> Any:
        return dict(self._criteria)

    def names(self) -> Any:
        return list(self._criteria)

    def __iter__(self) -> Any:
        return iter(self._criteria.items())

    def __len__(self) -> int:
        return len(self._criteria)

    def validate(self, context: Any = None) -> Any:
        return ReportTree(
            phase="validation",
            severity="info",
            code="validation.amr.root",
            source=self.family,
        )

    def inspect(self) -> Any:
        return {
            kind: getattr(descriptor, "name", repr(descriptor))
            for kind, descriptor in self._criteria.items()
        }


__all__ = ["ConstraintRegistry", "RuntimePolicyRegistry"]
