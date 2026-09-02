"""Frozen-IR lowering for the AMR two-table flux ABI.

The runtime must never infer a conservative face basis from aggregate bounds.  This module turns
the finite, already-frozen Program graph into exact occurrence rows and surviving ``dt`` terms.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.time.points import StagePoint, TimePoint


def _fraction(value: Any) -> Fraction:
    value = value.to_python() if hasattr(value, "to_python") else value
    try:
        return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise TypeError("AMR flux lowering requires exact rational coefficients") from error


def _polynomial(value: Any) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for power, coefficient in value.items():
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise TypeError("AMR flux lowering received an invalid dt power")
        ratio = _fraction(coefficient)
        if ratio:
            result[power] = result.get(power, Fraction(0)) + ratio
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def _multiply(left: dict[int, Fraction], right: dict[int, Fraction]) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = left_power + right_power
            result[power] = result.get(power, Fraction(0)) + left_coefficient * right_coefficient
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def _combine(terms: Any) -> dict[int, dict[int, Fraction]]:
    result: dict[int, dict[int, Fraction]] = {}
    for expression, coefficient in terms:
        for basis_slot, polynomial in expression.items():
            destination = result.setdefault(basis_slot, {})
            for power, factor in _multiply(polynomial, coefficient).items():
                destination[power] = destination.get(power, Fraction(0)) + factor
            result[basis_slot] = {power: factor for power, factor in destination.items() if factor}
            if not result[basis_slot]:
                del result[basis_slot]
    return result


def _has_flux(values: Any) -> bool:
    for value in values:
        if getattr(value, "op", None) == "rhs" and value.attrs.get("flux", True):
            return True
        for key in ("cond_block", "body_block", "apply_block", "residual_block",
                    "true_block", "false_block"):
            nested = value.attrs.get(key)
            if isinstance(nested, (list, tuple)) and _has_flux(nested):
                return True
    return False


def _stage(value: Any) -> tuple[Fraction, str]:
    """Return the conservative RHS coordinate and its authenticated clock.

    ``ProgramValue`` stores the whole ``StagePoint`` rather than a separate partition label.  A
    conservative RHS is the explicit member of an additive (IMEX/ARK) stage, so a split stage must
    select its named ``explicit`` coordinate when the convenience ``StagePoint.time`` accessor is
    ambiguous.  This is a semantic selection, not a positional or first-partition fallback.  Any
    point shape which cannot prove that coordinate remains a code-generation error.
    """
    point = getattr(value, "point", None)
    if type(point) is TimePoint:
        time_point = point
    elif type(point) is StagePoint:
        try:
            time_point = point.time
        except ValueError:
            try:
                # Conservative RHS evaluation is owned by the explicit ARK partition.  Do not
                # silently choose implicit (or an insertion-order-dependent coordinate).
                time_point = point.time_for("explicit")
            except (KeyError, TypeError, ValueError) as error:
                raise NotImplementedError(
                    "AMR flux lowering requires an exact explicit StagePoint partition"
                ) from error
    else:
        raise TypeError(
            "AMR flux lowering requires an exact TimePoint or StagePoint evaluation point"
        )
    stage = Fraction(time_point.step) + _fraction(time_point.offset)
    if stage < 0 or stage > 1:
        raise NotImplementedError("AMR flux lowering requires a stage fraction in [0, 1]")
    return stage, time_point.clock.qualified_id


def _resource_provenance(
    plan: Any,
    value: Any,
    *,
    table: str,
    occurrence_path: str,
) -> tuple[int, str, str, int]:
    """Return the sealed row used by a flux-table occurrence.

    ``ProgramBlockRecord.name`` is a runtime routing/display name, whereas a
    ``ProgramResourcePlan`` row retains the owner-qualified Program identity.
    Flux rows must carry the latter verbatim: re-deriving it from a block (or
    from a numeric value id) would detach the ABI provenance from the sealed
    resource it names.
    """

    try:
        # The persistent-plan occurrence closure is the sole authority for both
        # reachability and identity.  Passing its canonical path is essential
        # for a rehydrated plan, which intentionally has no object-id bindings;
        # omitting it would turn a missing row into a value-id fallback.
        row = plan.row_for_value(value, occurrence_path=occurrence_path)
        slot = row.slot
        key = row.key
        owner = key.owner
        clock = key.clock
        level = key.level
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "AMR %s lacks its exact retained ProgramResourcePlan row" % table
        ) from error
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError("AMR %s has an invalid sealed ProgramResourcePlan slot" % table)
    if not isinstance(owner, str) or not owner or not isinstance(clock, str) or not clock:
        raise ValueError("AMR %s has incomplete sealed resource provenance" % table)
    if level is None:
        return slot, owner, clock, -1
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("AMR %s has an invalid sealed resource level" % table)
    return slot, owner, clock, level


def flux_table_budgets(
    block_count: int,
    basis_occurrences: tuple[tuple[Any, ...], ...],
    face_flux_stages: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[int, int], ...]:
    """Return exact per-block bounds for the already-lowered flux ABI tables.

    ``program_block`` is the routing authority.  Resource owners intentionally
    remain qualified provenance strings and must not be reconstructed from a
    display block name.  Computing the bounds here, after reachability and
    affine-term cancellation, prevents the candidate budget from advertising
    an active basis which has no retained ABI row.
    """
    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 0:
        raise ValueError("AMR flux budget requires a non-negative exact block count")
    budgets = [[0, 0] for _ in range(block_count)]
    basis_blocks: dict[int, int] = {}
    for expected_slot, basis in enumerate(basis_occurrences):
        try:
            basis_slot = basis[0]
            program_block = basis[2]
        except (IndexError, TypeError) as error:
            raise ValueError("AMR flux budget received a malformed basis occurrence") from error
        if (isinstance(basis_slot, bool) or not isinstance(basis_slot, int)
                or basis_slot != expected_slot):
            raise ValueError("AMR flux budget requires dense exact basis slots")
        if (isinstance(program_block, bool) or not isinstance(program_block, int)
                or not 0 <= program_block < block_count):
            raise ValueError("AMR flux basis has an invalid Program block route")
        basis_blocks[basis_slot] = program_block
        budgets[program_block][0] += 1

    terms_by_expression_basis: dict[tuple[int, int], int] = {}
    for expected_slot, term in enumerate(face_flux_stages):
        try:
            term_slot = term[0]
            basis_slot = term[1]
            expression_slot = term[2]
        except (IndexError, TypeError) as error:
            raise ValueError("AMR flux budget received a malformed face-flux stage") from error
        if (isinstance(term_slot, bool) or not isinstance(term_slot, int)
                or term_slot != expected_slot):
            raise ValueError("AMR flux budget requires dense exact face-flux stage slots")
        if isinstance(basis_slot, bool) or not isinstance(basis_slot, int) or basis_slot not in basis_blocks:
            raise ValueError("AMR flux stage refers to a missing retained basis occurrence")
        if isinstance(expression_slot, bool) or not isinstance(expression_slot, int) or expression_slot < 0:
            raise ValueError("AMR flux stage has an invalid final expression slot")
        key = (basis_slot, expression_slot)
        terms_by_expression_basis[key] = terms_by_expression_basis.get(key, 0) + 1

    for (basis_slot, _expression_slot), count in terms_by_expression_basis.items():
        program_block = basis_blocks[basis_slot]
        budgets[program_block][1] = max(budgets[program_block][1], count)
    return tuple((rhs_basis, coefficient_terms) for rhs_basis, coefficient_terms in budgets)


def lower_amr_flux_tables(program: Any, plan: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return exact ``(basis_occurrences, final_terms)`` for an AMR Program.

    Occurrences are retained even if later affine work cancels them.  Only final commit terms are
    pruned, and each must be a single exact rational ``dt^1`` coefficient.  Finite range/subcycle
    bodies receive structural mixed-radix paths; flux-bearing dynamic control and history are
    intentionally refused rather than assigned a runtime counter or lookup key.
    """
    blocks = program._block_indices()
    cell_local = program.cell_local_time_contract() is not None
    # The bounded cell-temporal executor owns ExactFace accumulation directly.  It has no
    # two-table final-expression consumer, so emitting ordinary flux rows here would create a
    # second, orphaned receipt authority.  Keep the ABI tables empty (and the per-block budget at
    # zero in program_codegen) until a mixed executor supplies an explicit final consumer.
    if cell_local:
        return (), ()
    bases: list[tuple[Any, ...]] = []
    environment: dict[int, dict[int, dict[int, Fraction]]] = {}
    aliases = {"acceptance_guard": 0, "fill_boundary": 0, "project": 0,
               "solve_fields": 0, "synchronize": 0}
    # Keep this closure identical to ProgramResourcePlan lowering.  In
    # particular, an unconsumed matrix-free/interface declaration and its
    # captured RHS are absent here as well as in the sealed resource plan.
    from pops.codegen.program_persistent_plan import _reachable_program_occurrences

    reachable_occurrences = tuple(_reachable_program_occurrences(program))
    reachable_paths = {id(value): path for value, path in reachable_occurrences}
    reachable_ids = frozenset(reachable_paths)

    def provider(value: Any) -> int:
        fluxes = value.attrs.get("fluxes")
        if fluxes and tuple(fluxes) != ("default",):
            return 3  # NamedCell
        requested = value.attrs.get("sources")
        return 0 if requested is None or "default" in requested else 1

    def execute(values: Any, path: tuple[str, ...]) -> None:
        for ordinal, value in enumerate(values):
            if id(value) not in reachable_ids:
                continue
            here = path + ("%04d:%s:%d" % (ordinal, value.op, value.id),)
            occurrence_path = reachable_paths[id(value)]
            if value.op in ("branch", "while"):
                if any(_has_flux(value.attrs.get(key, ())) for key in (
                    "true_block", "false_block", "cond_block", "body_block")):
                    raise NotImplementedError(
                        "AMR flux lowering refuses branch/while control with a conservative flux basis"
                    )
                continue
            if value.op in ("range", "subcycle"):
                count = value.attrs.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("AMR flux lowering requires a finite integer range/subcycle count")
                if _has_flux(value.attrs.get("body_block", ())):
                    raise NotImplementedError(
                        "AMR flux lowering refuses range/subcycle flux until emitted RHS identities "
                        "carry the same mixed-radix occurrence slots"
                    )
                if not value.inputs or value.attrs.get("body") is None:
                    raise NotImplementedError(
                        "AMR flux lowering requires an explicit loop-carried input and body result"
                    )
                loop_input = value.inputs[0]
                loop_body = value.attrs["body"]
                carried = environment.get(loop_input.id, {})
                for index in range(count):
                    environment[loop_input.id] = carried
                    execute(value.attrs.get("body_block", ()), here + ("iteration:%d" % index,))
                    if loop_body.id not in environment:
                        raise NotImplementedError(
                            "AMR flux lowering could not prove the finite loop body result"
                        )
                    carried = environment[loop_body.id]
                environment[value.id] = carried
                continue

            expression: dict[int, dict[int, Fraction]] = {}
            if value.op == "rhs" and value.attrs.get("flux", True):
                if value.block not in blocks:
                    raise ValueError("AMR flux lowering found an RHS without an owned Program block")
                stage, point_clock = _stage(value)
                slot = len(bases)
                expression_slot, owner, clock, level = _resource_provenance(
                    plan, value, table="flux basis", occurrence_path=occurrence_path
                )
                if point_clock != clock:
                    raise ValueError(
                        "AMR flux basis evaluation clock differs from its sealed resource provenance"
                    )
                bases.append((slot, expression_slot, blocks[value.block], level, int(value.id), provider(value),
                              stage.numerator, stage.denominator,
                              "flux-basis:%s:%d:%s" % (owner, value.id, occurrence_path),
                              occurrence_path, owner, clock))
                expression = {slot: {0: Fraction(1)}}
            elif value.op == "linear_combine":
                expression = _combine((environment.get(source.id, {}), _polynomial(coefficient))
                                      for source, coefficient in zip(
                                          value.inputs, value.attrs["coeffs"], strict=True))
            elif value.op == "history":
                # A history read is harmless when the ring contains ordinary state/source data (or
                # is absent); only a retained conservative expression would require an
                # attempt-local face basis which this static lowering cannot authenticate.  The
                # store records the already-lowered expression under a private environment key so
                # this check follows the value actually rehydrated rather than the mere presence
                # of a history operation.
                expression = environment.get(("history_flux", value.attrs["history"]), {})
                if expression:
                    raise NotImplementedError(
                        "AMR flux lowering refuses history rehydration carrying an effective "
                        "conservative flux expression"
                    )
            elif value.op == "store_history":
                # Keep the expression for a later history read.  Storing a flux expression alone
                # does not rehydrate it at the current attempt boundary and therefore must not be
                # rejected prematurely.
                environment[("history_flux", value.attrs["history"])] = (
                    environment.get(value.inputs[0].id, {}) if value.inputs else {}
                )
            elif value.op in aliases and len(value.inputs) > aliases[value.op]:
                expression = environment.get(value.inputs[aliases[value.op]].id, {})
            environment[value.id] = expression

    execute(program._values, ("root",))
    terms: list[tuple[Any, ...]] = []
    for state_ref, committed in sorted(program._commits.items(), key=lambda item: item[0].qualified_id):
        for basis_slot, polynomial in sorted(environment.get(committed.id, {}).items()):
            if set(polynomial) != {1}:
                raise NotImplementedError(
                    "AMR flux lowering accepts only one exact rational dt^1 final term"
                )
            coefficient = polynomial[1]
            basis = bases[basis_slot]
            expression_slot, owner, clock, level = _resource_provenance(
                plan,
                committed,
                table="final flux expression",
                occurrence_path=reachable_paths[id(committed)],
            )
            # A final expression without an AMR-level qualifier is valid at every level and may
            # therefore retain one explicitly level-qualified basis.  Two explicit, different
            # levels remain an authority mismatch and are refused before artifact creation.
            if (owner, clock) != (basis[10], basis[11]) or (
                level != -1 and level != basis[3]
            ):
                raise ValueError(
                    "AMR final flux expression differs from its retained basis resource provenance"
                )
            terms.append((len(terms), basis_slot, expression_slot, 1, coefficient.numerator,
                          coefficient.denominator,
                          "face-flux:%s:%d:%d" % (state_ref.qualified_id, committed.id, basis_slot),
                          basis[9] + "/final:%d" % committed.id, owner, clock))
    return tuple(bases), tuple(terms)


__all__ = ["flux_table_budgets", "lower_amr_flux_tables"]
