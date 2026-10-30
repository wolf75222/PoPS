"""accepted-update-ssp@1: exact coefficient-one convex proof for an accepted value.

This is conditional on each explicit rate's existing forward-Euler spatial
guard. It neither changes expression lowering nor assigns a global method/order
to a Program containing independent Field or diagnostic effects.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from .program_emit_kernels import _coeff_metadata_terms
from pops.time._evaluation_point import evaluation_stage_fraction


def _exact_terms(value):
    encoded = value.to_data() if hasattr(value, "to_data") else value

    def boolean(item):
        pending = [item]
        seen = set()
        while pending:
            current = pending.pop()
            if type(current) is bool:
                return True
            if isinstance(current, (Mapping, tuple, list)):
                marker = id(current)
                if marker in seen:
                    continue
                seen.add(marker)
                if isinstance(current, Mapping):
                    if current.get("kind") == "boolean":
                        return True
                    for key, part in current.items():
                        pending.extend((key, part))
                else:
                    pending.extend(current)
        return False

    if boolean(encoded):
        raise ValueError("accepted coefficient has Boolean rather than real scalar metadata")
    return _coeff_metadata_terms(value)


def _proved_affine_coefficients(value):
    """Authenticate affine algebra and its optional pointwise storage contract.

    Stencil access is typed storage metadata rather than an evaluation effect.
    A neighbour read or any other attribute still needs a separate proof.
    """
    attrs = value.attrs
    if set(attrs) not in ({"coeffs"}, {"coeffs", "stencil_access"}):
        raise ValueError("affine accepted input has an unproved evaluation effect")
    if "stencil_access" in attrs:
        from pops.time.stencil import StencilAccess

        access = attrs["stencil_access"]
        if (
            type(access) is not StencilAccess
            or type(access.required_ghost_depth) is not int
            or access.required_ghost_depth != 0
        ):
            raise ValueError("affine accepted input lacks exact pointwise stencil access")
    return attrs["coeffs"]


def successful_guard_input(value):
    """A typed terminal guard is a value alias only on its successful path."""
    from pops.time.solve_outcome import FailRun, RejectAttempt

    if value.op != "acceptance_guard" or len(value.inputs) != 2:
        raise ValueError("accepted guard lacks its exact value/condition inputs")
    actual, condition = value.inputs
    allowed = {"guard", "action"}
    if value.attrs.get("numerical_stability") == "pops.authored-stability-admission@1":
        allowed.add("numerical_stability")
    if "assessment" in value.attrs:
        from pops.time._program.control import ADVANCE_ASSESSMENT_CONTRACT

        # An assessment of an Advance rejects its attempt; it never mutates the candidate.
        if value.attrs["assessment"] == ADVANCE_ASSESSMENT_CONTRACT \
                and type(value.attrs.get("action")) is RejectAttempt:
            allowed.add("assessment")
    if (
        set(value.attrs) != allowed
        or type(value.attrs.get("action")) not in (FailRun, RejectAttempt)
        or condition.vtype != "bool"
    ):
        raise ValueError("accepted guard has no proved terminal nonmutating action")
    if (
        value.vtype != actual.vtype
        or value.block != actual.block
        or value.clock != actual.clock
        or value.point != actual.point
    ):
        raise ValueError("accepted guard changes its value's exact type/owner/point")
    if (
        value.space != actual.space
        or value.state_ref != actual.state_ref
        or value.field_context != actual.field_context
    ):
        raise ValueError("accepted guard changes its exact mathematical storage/context type")
    return actual


@dataclass(frozen=True, slots=True)
class AcceptedUpdateSSP:
    graph_hash: str
    accepted_ssa: int
    contributing_rates: tuple[int, ...]
    A: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    c: tuple[Fraction, ...]
    base_weights: tuple[Fraction, ...]
    forward_euler_weights: tuple[tuple[Fraction, ...], ...]
    coefficient: Fraction = Fraction(1)
    version: int = 1


@dataclass(frozen=True, slots=True)
class AcceptedSSPPlan:
    updates: tuple[AcceptedUpdateSSP, ...]
    coefficient: Fraction = Fraction(1)
    version: int = 1


def _convex_coefficient_one(A, b):
    """Prove q = v*u0 + alpha*(q + dt*F(q)) using exact rational algebra."""
    n = len(b) + 1
    K = [list(row) + [Fraction()] for row in A] + [list(b) + [Fraction()]]
    inverse = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            inverse[i][j] = -sum((K[i][k] * inverse[k][j] for k in range(j, i)), Fraction())
    base = tuple(sum(row, Fraction()) for row in inverse)
    alpha = tuple(tuple(Fraction(int(i == j)) - inverse[i][j] for j in range(n)) for i in range(n))
    if any(value < 0 for value in base) or any(value < 0 for row in alpha for value in row):
        raise ValueError("coefficient-one forward-Euler convex weights are negative")
    assert all(base[i] + sum(alpha[i], Fraction()) == 1 for i in range(n))
    return base, alpha


def _independent_effect(program, value):
    """Typed effects that never write a live accepted State input.

    Dependency closure is checked separately: a Field, history, or diagnostic
    used by an accepted rate is not silently discarded by this classification.
    """
    if value.op in program._REMOVABLE_OPS or value.op == "state":
        # Reuse the frontend's verified fresh-result, non-input-writing contract.
        return True
    if value.op in {"solve_fields", "solve_fields_from_blocks"}:
        return value.vtype == "fields" and value.field_context is not None
    if value.op == "solve_outcome":
        return value.vtype == "solve_outcome" and len(value.inputs) == 1
    if value.op == "solve_outcome_component":
        return len(value.inputs) == 1 and value.inputs[0].op == "solve_outcome"
    if value.op == "store_history":
        return len(value.inputs) == 1
    if value.op == "record_scalar":
        return value.vtype in {"scalar", "bool"}
    if value.op == "acceptance_guard":
        successful_guard_input(value)
        return True
    # These mathematical Field operations produce/inspect solver-owned values;
    # they never publish an accepted State endpoint.
    if value.op in {
        "field_problem_load",
        "field_problem_coefficients",
        "field_component",
        "solve_linear",
    }:
        return value.vtype in {"scalar_field", "solve_outcome"}
    if value.op == "matrix_free_operator":
        return value.vtype == "matrix_free_op"
    if value.op in {"field_publication", "input_fields"}:
        return value.vtype == "fields"
    return False


def prove_accepted_update_ssp(
    program: Any, *, model_authority: Any = None
) -> tuple[AcceptedUpdateSSP | AcceptedSSPPlan | None, str]:
    """Prove each independently owned diffusive accepted State.

    Operators may have different authored expressions. Only exact typed
    dependencies, owners, coordinates and coefficient arithmetic enter this
    proof; names, fingerprints and known tableaux do not select a recipe.
    """
    try:
        from .program_emit_field_routes import _walk_program_nodes

        issued = {}
        for value in _walk_program_nodes(program._values):
            if type(value.id) is not int:
                raise ValueError("accepted input has a noninteger SSA identifier")
            if value.id in issued and issued[value.id] is not value:
                raise ValueError("executed graph has detached duplicate SSA definitions")
            issued[value.id] = value
        for value in issued.values():
            for child in value.inputs:
                if issued.get(child.id) is not child:
                    raise ValueError("executed input/condition has detached SSA authority")
        # Check static availability contexts before serialization follows their
        # stage references. A forged self-stage must fail closed, not recurse.
        from .program_static_provider_proof import prove_static_provider_read

        for value in issued.values():
            if value.op == "diffusive_rhs":
                for extra in value.inputs[1:]:
                    if extra.op == "input_fields":
                        prove_static_provider_read(
                            program, extra, value, set(), model_authority, issued
                        )
        graph_hash = program.to_graph().graph_hash  # authenticates issued SSA references
        if any(not _independent_effect(program, value) for value in program._values):
            raise ValueError("an executed effect has no proved accepted-State nonmutation contract")
        frozen = set()

        def frozen_atoms(value, weight, out):
            # Cache each closed subexpression, then multiply each incoming edge.
            # Shared aliases keep their multiplicity without unfolding a DAG.
            complete = {}
            active_atoms = set()
            pending = [(value, False)]
            while pending:
                current, exiting = pending.pop()
                if current.id in complete:
                    continue
                if not exiting and current.id in active_atoms:
                    raise ValueError("accepted frozen input closure contains a cycle")
                if current.op == "state":
                    complete[current.id] = {current.state_ref: Fraction(1)}
                    continue
                if current.op == "acceptance_guard":
                    children = (successful_guard_input(current),)
                    factors = (Fraction(1),)
                elif current.op == "linear_combine":
                    children = current.inputs
                    factors = []
                    for factor in _proved_affine_coefficients(current):
                        terms = tuple(_exact_terms(factor))
                        if any(power != 0 for power, _, _ in terms):
                            return False
                        factors.append(sum((Fraction(a, b) for _, a, b in terms), Fraction()))
                    if len(children) != len(factors):
                        raise ValueError(
                            "affine accepted input has different input/coefficient counts"
                        )
                else:
                    return False
                if not exiting:
                    active_atoms.add(current.id)
                    pending.append((current, True))
                    pending.extend((child, False) for child in reversed(children))
                    continue
                result = {}
                for child, factor in zip(children, factors, strict=True):
                    for reference, amount in complete[child.id].items():
                        result[reference] = result.get(reference, Fraction()) + factor * amount
                complete[current.id] = result
                active_atoms.remove(current.id)
            for reference, amount in complete[value.id].items():
                out[reference] = out.get(reference, Fraction()) + weight * amount
            return True

        for reference, value in program._commits.items():
            atoms = {}
            if frozen_atoms(value, Fraction(1), atoms) and atoms == {reference: Fraction(1)}:
                frozen.add(reference)
        verified = {}
        visiting_reads = set()

        def frozen_read(value, reads, consumer, bound=None):
            # Explicit DFS frames preserve all() short-circuit order and memoize
            # only completed contexts, including each exact operator binder.
            frames = []
            task = (value, {} if bound is None else bound)
            read_key = frozenset(reads)
            try:
                while True:
                    current, local_bound = task
                    if issued.get(current.id) is not current:
                        raise ValueError("Field read closure has a detached SSA definition")
                    if current.id in local_bound:
                        if local_bound[current.id] is not current:
                            raise ValueError("Field binder has detached storage authority")
                        result = True
                    else:
                        key = (
                            current.id,
                            read_key,
                            consumer,
                            tuple(sorted(local_bound)),
                        )
                        if key in visiting_reads:
                            raise ValueError("Field mathematical read closure contains a cycle")
                        if key in verified:
                            result = verified[key]
                        else:
                            body = frozen_read_body(current, reads, consumer, local_bound)
                            if type(body) is bool:
                                verified[key] = result = body
                            else:
                                visiting_reads.add(key)
                                iterator = iter(body)
                                frames.append((key, iterator))
                                task = next(iterator, None)
                                if task is not None:
                                    continue
                                result = True
                    while frames:
                        key, iterator = frames[-1]
                        if result:
                            task = next(iterator, None)
                            if task is not None:
                                break
                        verified[key] = result
                        visiting_reads.remove(key)
                        frames.pop()
                    else:
                        return result
            finally:
                for key, _ in frames:
                    visiting_reads.remove(key)

        def frozen_read_body(value, reads, consumer, bound):
            if value.op == "acceptance_guard":
                return ((successful_guard_input(value), bound),)
            if value.op == "state":
                return value.state_ref in frozen
            if value.op == "field_publication":
                from pops.fields._program_publication import validate_field_publication
                from .program_models import ProgramModelGraph
                from .program_field_publication import publication_target_space

                target_space = (
                    (lambda destination: publication_target_space(model_authority, destination))
                    if type(model_authority) is ProgramModelGraph
                    else None
                )
                bindings = validate_field_publication(value, target_space=target_space)
                supplied = {
                    row["component"] for row in bindings if row["target"].block_ref == consumer
                }
                if not reads <= supplied:
                    return False
                return ((child, bound) for child in value.inputs[: len(bindings)])
            if value.op == "matrix_free_operator":
                formal_in, formal_out = (
                    value.attrs["apply_in"],
                    value.attrs["apply_out"],
                )
                body = value.attrs["apply_block"]
                if (
                    formal_in.op != "apply_in"
                    or formal_out.op != "apply_out"
                    or formal_in is formal_out
                ):
                    raise ValueError("Field operator has no exact input/output binders")
                if not any(child is formal_in for child in body) or not any(
                    child is formal_out for child in body
                ):
                    raise ValueError("Field operator binders do not belong to its actual body")
                local = {formal_in.id: formal_in, formal_out.id: formal_out}
                return ((child, local) for child in body)
            if value.op == "field_problem_apply":
                if (
                    len(value.inputs) != 3
                    or value.inputs[0].id not in bound
                    or value.inputs[1].id not in bound
                ):
                    return False
                return ((value.inputs[2], bound),)
            if not _independent_effect(program, value):
                return False
            return ((child, bound) for child in value.inputs)

        certificates = []
        for state_ref, accepted in program._commits.items():
            closure, rates, visiting = {}, {}, set()
            owner, clock = state_ref.block_ref, accepted.clock

            def visit(
                root,
                *,
                rates=rates,
                visiting=visiting,
                closure=closure,
                owner=owner,
                clock=clock,
                state_ref=state_ref,
            ):
                visit_pending = [(root, False)]
                while visit_pending:
                    value, exiting = visit_pending.pop()
                    if exiting:
                        if value.op == "diffusive_rhs":
                            rates[value.id] = value
                        visiting.remove(value.id)
                        closure[value.id] = value
                        continue
                    if issued.get(value.id) is not value:
                        raise ValueError("accepted input has a detached SSA definition")
                    if value.id in visiting:
                        raise ValueError("accepted input closure contains a cycle")
                    if value.id in closure:
                        if closure[value.id] is not value:
                            raise ValueError("accepted input closure has a detached SSA identity")
                        continue
                    if value.block != owner or value.clock != clock:
                        raise ValueError("accepted input mixes distinct State owners or clocks")
                    visiting.add(value.id)
                    visit_pending.append((value, True))
                    if value.op == "acceptance_guard":
                        visit_pending.append((successful_guard_input(value), False))
                        continue
                    if value.op == "state":
                        if value.state_ref != state_ref or evaluation_stage_fraction(value) != 0:
                            raise ValueError("accepted base is not its exact initial State")
                    elif value.op == "linear_combine":
                        _proved_affine_coefficients(value)
                        visit_pending.extend(
                            (child, False) for child in reversed(value.inputs)
                        )  # include zero-weight effect inputs
                    elif value.op == "diffusive_rhs":
                        if not value.inputs:
                            raise ValueError(
                                "forward-Euler premise unproved for dependent Field/evaluation inputs"
                            )
                        if value.attrs.get("schedule") is not None:
                            raise ValueError("forward-Euler premise unproved for a cadenced rate")
                        # A Field publication can change shared auxiliary reads even
                        # without a value edge. Refuse that unproved premise rather
                        # than pretending the observation was independent.
                        from pops._ir.expr import Var
                        from pops._ir.quantity import QuantityRef
                        from pops._ir.visitors import _children
                        from pops._ir.primitive_expansion import (
                            expand_primitive_recipes,
                        )
                        from pops.time.references import canonical_handle

                        view = value.attrs["physical_balance"]
                        from .program_models import ProgramModelGraph

                        if type(model_authority) is ProgramModelGraph:
                            source_owner = model_authority.owner_for_block(value.block)
                            module = model_authority.source_module_for_owner(source_owner)
                            if (
                                module.owner_path.canonical()
                                != value.block.model_owner_path.canonical()
                            ):
                                raise ValueError(
                                    "constitutive read contract has a different source Module owner"
                                )
                        else:
                            registry = getattr(value.block, "_instance_registry", None)
                            if registry is None:
                                raise ValueError(
                                    "detached accepted proof requires its ProgramModelGraph source authority"
                                )
                            model = registry.spec(value.block.local_id)["model"]
                            module = getattr(model, "module", model)
                        if module.owner_path.canonical() != view.target.owner_path.canonical():
                            raise ValueError(
                                "constitutive read contract differs from the physical State owner"
                            )
                        roots = []
                        for occurrence in view.occurrences:
                            if hasattr(occurrence.payload, "law"):
                                roots.extend(occurrence.payload.law.expressions)
                            elif occurrence.kind in {"source", "flux"}:
                                roots.append(
                                    module.operator_registry().get(occurrence.payload.reg_name).body
                                )
                            else:
                                raise ValueError(
                                    "forward-Euler premise lacks a closed constitutive read contract"
                                )
                        pending = list(expand_primitive_recipes(roots, module.primitive_recipes()))
                        constitutive_reads = set()
                        while pending:
                            expr = pending.pop()
                            if isinstance(expr, Mapping):
                                pending.extend(expr.values())
                                continue
                            if isinstance(expr, (tuple, list)):
                                pending.extend(expr)
                                continue
                            if isinstance(expr, QuantityRef) and expr.handle.kind == "field":
                                constitutive_reads.add(expr.component)
                            if isinstance(expr, Var) and expr.kind in {
                                "aux",
                                "field",
                                "prim",
                            }:
                                constitutive_reads.add(expr.name)
                            if (
                                isinstance(expr, QuantityRef)
                                and expr.handle.kind == "state"
                                and canonical_handle(expr.handle) != canonical_handle(view.target)
                            ):
                                raise ValueError(
                                    "forward-Euler premise reads a different constitutive State"
                                )
                            pending.extend(_children(expr))
                        if constitutive_reads and len(value.inputs) == 1:
                            raise ValueError(
                                "forward-Euler premise lacks closed constitutive reads"
                            )
                        from .program_static_provider_proof import (
                            prove_static_provider_read,
                        )
                        from .program_stage_provider_proof import (
                            prove_stage_provider_read,
                        )

                        if any(
                            not (
                                prove_static_provider_read(
                                    program,
                                    extra,
                                    value,
                                    constitutive_reads,
                                    model_authority,
                                    issued,
                                )
                                if extra.op == "input_fields"
                                else (
                                    frozen_read(extra, constitutive_reads, value.block)
                                    or prove_stage_provider_read(
                                        program,
                                        extra,
                                        value,
                                        constitutive_reads,
                                        model_authority,
                                        issued,
                                        frozen_read,
                                    )
                                )
                            )
                            for extra in value.inputs[1:]
                        ):
                            raise ValueError(
                                "forward-Euler premise unproved for dependent Field/evaluation inputs"
                            )
                        visit_pending.append((value.inputs[0], False))
                    else:
                        raise ValueError(
                            "accepted input has an unproved non-affine/effect dependency: "
                            + value.op
                        )

            visit(accepted)
            states = [value for value in closure.values() if value.op == "state"]
            if len(states) != 1:
                raise ValueError("accepted affine update lacks one exact initial State")
            initial = states[0]
            ordered_rates = list(rates.values())  # dependency order, never SSA-label order
            index = {value.id: i for i, value in enumerate(ordered_rates)}
            cache = {}

            def expression(value, stage, *, cache=cache, initial=initial, index=index):
                pending = [(value, False)]
                while pending:
                    current, exiting = pending.pop()
                    key = (current.id, stage)
                    if key in cache:
                        continue
                    if not exiting and current is not initial and current.id not in index:
                        if current.op == "acceptance_guard":
                            children = (successful_guard_input(current),)
                        elif current.op == "linear_combine":
                            _proved_affine_coefficients(current)
                            children = current.inputs
                        else:
                            raise ValueError(
                                "accepted affine input lacks an exact coefficient expression"
                            )
                        pending.append((current, True))
                        pending.extend((child, False) for child in reversed(children))
                        continue
                    result = [{} for _ in range(stage + 1)]
                    if current is initial:
                        result[0][0] = Fraction(1)
                    elif current.id in index:
                        i = index[current.id]
                        if i >= stage:
                            raise ValueError("accepted stage reads a nonprevious rate")
                        result[i + 1][0] = Fraction(1)
                    elif current.op == "acceptance_guard":
                        result = cache[(successful_guard_input(current).id, stage)]
                    else:
                        for child, coeffs in zip(
                            current.inputs,
                            _proved_affine_coefficients(current),
                            strict=True,
                        ):
                            terms = cache[(child.id, stage)]
                            for power, numerator, denominator in _exact_terms(coeffs):
                                if type(power) is not int or power < 0:
                                    raise ValueError(
                                        "accepted coefficient has a noninteger/negative dt power"
                                    )
                                factor = Fraction(numerator, denominator)
                                for target, term in zip(result, terms, strict=True):
                                    for degree, amount in term.items():
                                        target[degree + power] = (
                                            target.get(degree + power, Fraction()) + factor * amount
                                        )
                    cache[key] = result
                return cache[(value.id, stage)]

            def affine(value, stage):
                terms = [{p: x for p, x in row.items() if x} for row in expression(value, stage)]
                if set(terms[0]) - {0} or any(set(row) - {1} for row in terms[1:]):
                    raise ValueError("accepted State/rate coefficients have incompatible dt powers")
                if terms[0].get(0, Fraction()) != 1:
                    raise ValueError(
                        "accepted affine expression does not preserve its initial State"
                    )
                return tuple(row.get(1, Fraction()) for row in terms[1:])

            b = affine(accepted, len(ordered_rates))
            if not ordered_rates:
                continue  # its exact algebraic image is the initial State
            if evaluation_stage_fraction(accepted) != 1 or sum(b, Fraction()) != 1:
                raise ValueError("accepted update does not span one consistent step")
            A = []
            c = []
            for i, rate in enumerate(ordered_rates):
                row = affine(rate.inputs[0], i)
                point = evaluation_stage_fraction(rate)
                if (
                    evaluation_stage_fraction(rate.inputs[0]) != point
                    or sum(row, Fraction()) != point
                ):
                    raise ValueError(
                        "rate/input stage coordinate differs from its exact affine row sum"
                    )
                A.append(row + (Fraction(),) * (len(b) - len(row)))
                c.append(point)
            base, alpha = _convex_coefficient_one(A, b)
            certificates.append(
                AcceptedUpdateSSP(
                    graph_hash,
                    accepted.id,
                    tuple(value.id for value in ordered_rates),
                    tuple(A),
                    b,
                    tuple(c),
                    base,
                    alpha,
                )
            )
        if not certificates:
            raise ValueError("accepted update has no contributing explicit rates")
        return (
            certificates[0] if len(certificates) == 1 else AcceptedSSPPlan(tuple(certificates))
        ), ""
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return None, str(error)
