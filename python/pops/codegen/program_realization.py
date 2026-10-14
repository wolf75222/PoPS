"""Authenticate Program-owned balances and prepare their private storage carrier."""
from collections.abc import Mapping
import json

from pops.time.values import ProgramValue


def _leaves(value):
    if isinstance(value, ProgramValue):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)


def _data(value):
    if isinstance(value, Mapping):
        return {str(key): _data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_data(item) for item in value]
    return value


def _key(value):
    return json.dumps(_data(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _methods(numerics):
    from pops.numerics.program_realization import ProgramRealization
    return () if numerics is None else tuple(
        row for row in numerics.rates if isinstance(row.method, ProgramRealization))


def _same_reference(left, right):
    from pops.model import Handle
    return (isinstance(left, Handle) and isinstance(right, Handle)
            and left._resolved() == right._resolved())


def validate_program_realizations(numerics, program):
    """A diagnostic/read does not deliver an owned balance occurrence to its state."""
    methods = _methods(numerics)
    if not methods:
        return True
    from pops.codegen._resolved_block_operations import _program_values
    values = {value.id: value for value, _, _ in _program_values(program)}
    for row in methods:
        method = row.method
        target = method.target
        if target is None or not target.is_instance:
            raise ValueError("ProgramRealization lost its exact block-qualified target")
        commits = [value for state, value in program._commits.items() if _same_reference(state, target)]
        if len(commits) != 1:
            raise ValueError("ProgramRealization target has no native Program commit")
        commit = commits[0]
        live = set()
        def visit(value):
            if value.id in live:
                return
            live.add(value.id)
            support_only = value.attrs.get("support_only_inputs", ())
            for index, dependency in enumerate(value.inputs):
                if index not in support_only:
                    visit(dependency)
            for name, attribute in value.attrs.items():
                if not name.endswith("_block"):
                    for dependency in _leaves(attribute):
                        visit(dependency)
        visit(commit)
        expected = {_key(use.to_data()): use for use in method.uses}
        counts = {key: [] for key in expected}
        owned_occurrences = {_key(use.occurrence.to_data()) for use in method.uses}
        for value in values.values():
            evidence = value.attrs.get("realization_use_v02")
            if evidence is None:
                continue
            use = evidence.get("use", {})
            # The same Model may occur in several blocks. Its equation identity
            # does not authorize a contribution produced for another instance.
            if not _same_reference(value.block, target.block_ref):
                continue
            key = _key(use)
            if _key(use.get("occurrence")) in owned_occurrences and key not in expected:
                raise ValueError("ProgramRealization found an undeclared owned Use for its balance")
            if key in counts:
                counts[key].append(value)
        from .realization_delivery import delivered_frontiers
        for key, outputs in counts.items():
            if not outputs:
                raise ValueError("ProgramRealization Use needs exactly one owned SSA delivery; found 0")
            output_ids = {value.id for value in outputs}
            frontiers = delivered_frontiers(commit, lambda node: node.id in output_ids)
            if not frontiers or any(not frontier for frontier in frontiers):
                raise ValueError("ProgramRealization owned output does not reach its exact state commit")
            if any(len(frontier) != 1 for frontier in frontiers):
                raise ValueError("ProgramRealization Use needs exactly one owned SSA delivery per candidate path")
            retained = {reference.value_id for frontier in frontiers for reference in frontier}
            for value in outputs:
                if (value.id in retained and value.state_ref is not None
                        and not _same_reference(value.state_ref, target)):
                    raise ValueError("ProgramRealization output belongs to another physical state")
        # An explicitly Program-owned rate is not also evaluated by a hidden
        # numerical residual. Such an evaluation needs its own selected route.
        for value in values.values():
            operator = value.attrs.get("operator_handle")
            declaration = getattr(row.rate, "declaration_ref", None)
            if (_same_reference(value.block, target.block_ref)
                    and (_same_reference(operator, row.rate)
                         or _same_reference(operator, declaration)
                         or any(_same_reference(operator, use.occurrence.balance)
                                for use in method.uses))):
                raise ValueError("ProgramRealization rate cannot also use a native rate evaluation")
    return True


def prepare_program_realization_carrier(emitter, module, numerics, *, emitter_is_private=False, frame=None):
    """Keep source equations but remove unused default FV/eigenvalue installation."""
    methods = _methods(numerics)
    if not methods:
        return emitter
    def storage_only(method):
        requirements = getattr(method, "runtime_storage_requirements", None)
        return callable(requirements) and requirements() is not None
    if any(not storage_only(row.method) for row in numerics.rates):
        # A separately selected native FV rate still owns that transport carrier.
        return emitter
    targets = {row.method.target.local_id for row in methods}
    if len(targets) != 1:
        raise ValueError("Program storage carrier needs one exact selected state")
    state = module.state_spaces()[next(iter(targets))]
    if state.layout != "cell" or state.centering != "cell" or state.storage != "multifab":
        raise ValueError("ProgramRealization native storage requires authored cell/multifab state")
    if not emitter_is_private:
        from pops.model.state_symbols import native_formula_view
        emitter = native_formula_view(emitter, module)
    impl = getattr(emitter, "_m", emitter)
    axes = tuple(getattr(impl, "_flux", {}))
    if not axes:
        axes = tuple(getattr(impl, "_program_only_storage_axes", ()))
    if not axes:
        # The exact frame is retained by the declared balance target. Do not
        # derive axes from the installed native dimension or a physical name.
        if frame is None or frame.canonical_id != state.frame:
            raise ValueError("ProgramRealization storage lost its authored frame")
        axes = tuple(axis.name for axis in frame.axes)
    from pops._cartesian_axes import canonical_axis_mapping
    axes = tuple(canonical_axis_mapping(dict.fromkeys(axes), where="ProgramRealization state frame"))
    object.__setattr__(impl, "_program_only_storage_axes", axes)
    object.__setattr__(impl, "_flux", {})
    object.__setattr__(impl, "_eig", {})
    object.__setattr__(impl, "_wave_speeds", {})
    return emitter
