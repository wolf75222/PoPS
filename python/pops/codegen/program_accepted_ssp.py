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
    encoded=value.to_data() if hasattr(value,'to_data') else value
    def boolean(item):
        if type(item) is bool:return True
        if isinstance(item,Mapping):
            return item.get('kind')=='boolean' or any(boolean(k) or boolean(v) for k,v in item.items())
        if isinstance(item,(tuple,list)):return any(boolean(part) for part in item)
        return False
    if boolean(encoded):raise ValueError("accepted coefficient has Boolean rather than real scalar metadata")
    return _coeff_metadata_terms(value)


def successful_guard_input(value):
    """A typed terminal guard is a value alias only on its successful path."""
    from pops.time.solve_outcome import FailRun, RejectAttempt
    if value.op!='acceptance_guard' or len(value.inputs)!=2:
        raise ValueError("accepted guard lacks its exact value/condition inputs")
    actual,condition=value.inputs
    if set(value.attrs)!={'guard','action'} or type(value.attrs.get('action')) not in (FailRun,RejectAttempt) or condition.vtype!='bool':
        raise ValueError("accepted guard has no proved terminal nonmutating action")
    if value.vtype!=actual.vtype or value.block!=actual.block or value.clock!=actual.clock or value.point!=actual.point:
        raise ValueError("accepted guard changes its value's exact type/owner/point")
    if value.space!=actual.space or value.state_ref!=actual.state_ref or value.field_context!=actual.field_context:
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
            inverse[i][j] = -sum(K[i][k] * inverse[k][j] for k in range(j, i))
    base = tuple(sum(row, Fraction()) for row in inverse)
    alpha = tuple(tuple(Fraction(int(i == j)) - inverse[i][j]
                        for j in range(n)) for i in range(n))
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
    if value.op=='acceptance_guard':
        successful_guard_input(value)
        return True
    # These mathematical Field operations produce/inspect solver-owned values;
    # they never publish an accepted State endpoint.
    if value.op in {"field_problem_load","field_problem_coefficients","field_component","solve_linear"}:
        return value.vtype in {"scalar_field","solve_outcome"}
    if value.op == "matrix_free_operator":
        return value.vtype == "matrix_free_op"
    if value.op in {"field_publication","input_fields"}:
        return value.vtype == "fields"
    return False


def prove_accepted_update_ssp(program: Any) -> tuple[AcceptedUpdateSSP | AcceptedSSPPlan | None, str]:
    """Prove each independently owned diffusive accepted State.

    Operators may have different authored expressions. Only exact typed
    dependencies, owners, coordinates and coefficient arithmetic enter this
    proof; names, fingerprints and known tableaux do not select a recipe.
    """
    try:
        graph_hash = program.to_graph().graph_hash  # authenticates issued SSA references
        from .program_emit_field_routes import _walk_program_nodes
        issued={}
        for value in _walk_program_nodes(program._values):
            if type(value.id) is not int:
                raise ValueError("accepted input has a noninteger SSA identifier")
            if value.id in issued and issued[value.id] is not value:
                raise ValueError("executed graph has detached duplicate SSA definitions")
            issued[value.id]=value
        for value in issued.values():
            for child in value.inputs:
                if issued.get(child.id) is not child:
                    raise ValueError("executed input/condition has detached SSA authority")
        if any(not _independent_effect(program, value) for value in program._values):
            raise ValueError("an executed effect has no proved accepted-State nonmutation contract")
        frozen=set()
        def frozen_atoms(value,weight,out):
            if value.op=='state':
                out[value.state_ref]=out.get(value.state_ref,Fraction())+weight
            elif value.op=='linear_combine' and set(value.attrs)=={'coeffs'}:
                for child,factors in zip(value.inputs,value.attrs['coeffs'],strict=True):
                    terms=tuple(_exact_terms(factors))
                    if any(power!=0 for power,_,_ in terms):return False
                    if not frozen_atoms(child,weight*sum((Fraction(a,b) for _,a,b in terms),Fraction()),out):return False
            else:return False
            return True
        for reference,value in program._commits.items():
            atoms={}
            if frozen_atoms(value,Fraction(1),atoms) and atoms=={reference:Fraction(1)}:
                frozen.add(reference)
        verified={};visiting_reads=set()
        def frozen_read(value,reads,consumer,bound=None):
            bound={} if bound is None else bound
            if issued.get(value.id) is not value:
                raise ValueError("Field read closure has a detached SSA definition")
            if value.id in bound:
                if bound[value.id] is not value:raise ValueError("Field binder has detached storage authority")
                return True
            key=(value.id,frozenset(reads),consumer,tuple(sorted(bound)))
            if key in visiting_reads:raise ValueError("Field mathematical read closure contains a cycle")
            if key in verified:return verified[key]
            visiting_reads.add(key)
            try:result=frozen_read_body(value,reads,consumer,bound)
            finally:visiting_reads.remove(key)
            verified[key]=result
            return result
        def frozen_read_body(value,reads,consumer,bound):
            if value.op=='state':return value.state_ref in frozen
            if value.op=='field_publication':
                from pops.fields._program_publication import validate_field_publication
                bindings=validate_field_publication(value)
                supplied={row['component'] for row in bindings if row['target'].block_ref==consumer}
                if not reads<=supplied:return False
                # Remaining inputs are consumer-State binding/shape authority.
                # Only authenticated published components in the actual read
                # union are used here; unknown prerequisite reads are refused.
                return all(frozen_read(child,reads,consumer,bound)
                           for child in value.inputs[:len(bindings)])
            if value.op=='matrix_free_operator':
                formal_in,formal_out=value.attrs['apply_in'],value.attrs['apply_out']
                body=value.attrs['apply_block']
                if formal_in.op!='apply_in' or formal_out.op!='apply_out' or formal_in is formal_out:
                    raise ValueError("Field operator has no exact input/output binders")
                if not any(child is formal_in for child in body) or not any(child is formal_out for child in body):
                    raise ValueError("Field operator binders do not belong to its actual body")
                local={formal_in.id:formal_in,formal_out.id:formal_out}
                return all(frozen_read(child,reads,consumer,local)
                           for child in value.attrs['apply_block'])
            if value.op=='field_problem_apply':
                if len(value.inputs)!=3 or value.inputs[0].id not in bound or value.inputs[1].id not in bound:
                    return False
                return frozen_read(value.inputs[2],reads,consumer,bound)
            return _independent_effect(program,value) and all(
                frozen_read(child,reads,consumer,bound) for child in value.inputs)
        certificates = []
        for state_ref, accepted in program._commits.items():
            closure, rates, visiting = {}, {}, set()
            owner, clock = state_ref.block_ref, accepted.clock

            def visit(value):
                if issued.get(value.id) is not value:
                    raise ValueError("accepted input has a detached SSA definition")
                if value.id in visiting:
                    raise ValueError("accepted input closure contains a cycle")
                if value.id in closure:
                    if closure[value.id] is not value:
                        raise ValueError("accepted input closure has a detached SSA identity")
                    return
                if value.block != owner or value.clock != clock:
                    raise ValueError("accepted input mixes distinct State owners or clocks")
                visiting.add(value.id)
                if value.op=='acceptance_guard':
                    visit(successful_guard_input(value))
                    visiting.remove(value.id);closure[value.id]=value
                    return
                if value.op == "state":
                    if value.state_ref != state_ref or evaluation_stage_fraction(value) != 0:
                        raise ValueError("accepted base is not its exact initial State")
                elif value.op == "linear_combine":
                    if set(value.attrs) != {"coeffs"}:
                        raise ValueError("affine accepted input has an unproved evaluation effect")
                    for child in value.inputs:visit(child)  # include zero-weight effect inputs
                elif value.op == "diffusive_rhs":
                    if not value.inputs:
                        raise ValueError("forward-Euler premise unproved for dependent Field/evaluation inputs")
                    if value.attrs.get("schedule") is not None:
                        raise ValueError("forward-Euler premise unproved for a cadenced rate")
                    # A Field publication can change shared auxiliary reads even
                    # without a value edge. Refuse that unproved premise rather
                    # than pretending the observation was independent.
                    from pops._ir.expr import Var
                    from pops._ir.quantity import QuantityRef
                    from pops._ir.visitors import _children
                    from pops._ir.primitive_expansion import expand_primitive_recipes
                    view=value.attrs['physical_balance']
                    model=value.block._instance_registry.spec(value.block.local_id)['model']
                    module=getattr(model,'module',model)
                    roots=[]
                    for occurrence in view.occurrences:
                        if hasattr(occurrence.payload,'law'):
                            roots.extend(occurrence.payload.law.expressions)
                        elif occurrence.kind in {'source','flux'}:
                            roots.append(module.operator_registry().get(occurrence.payload.reg_name).body)
                        else:
                            raise ValueError("forward-Euler premise lacks a closed constitutive read contract")
                    pending=list(expand_primitive_recipes(roots,module.primitive_recipes()))
                    constitutive_reads=set()
                    while pending:
                        expr=pending.pop()
                        if isinstance(expr,Mapping):
                            pending.extend(expr.values());continue
                        if isinstance(expr,(tuple,list)):
                            pending.extend(expr);continue
                        if isinstance(expr,QuantityRef) and expr.handle.kind=='field':
                            constitutive_reads.add(expr.component)
                        if isinstance(expr,Var) and expr.kind in {'aux','field','prim'}:
                            constitutive_reads.add(expr.name)
                        if isinstance(expr,QuantityRef) and expr.handle.kind=='state' and expr.handle!=view.target:
                            raise ValueError("forward-Euler premise reads a different constitutive State")
                        pending.extend(_children(expr))
                    if constitutive_reads and len(value.inputs)==1:
                        raise ValueError("forward-Euler premise lacks closed constitutive reads")
                    if any(not frozen_read(extra,constitutive_reads,value.block)
                           for extra in value.inputs[1:]):
                        raise ValueError("forward-Euler premise unproved for dependent Field/evaluation inputs")
                    visit(value.inputs[0]);rates[value.id] = value
                else:
                    raise ValueError("accepted input has an unproved non-affine/effect dependency: " + value.op)
                visiting.remove(value.id);closure[value.id] = value

            visit(accepted)
            states = [value for value in closure.values() if value.op == "state"]
            if len(states) != 1:
                raise ValueError("accepted affine update lacks one exact initial State")
            initial = states[0]
            ordered_rates = list(rates.values())  # dependency order, never SSA-label order
            index = {value.id:i for i,value in enumerate(ordered_rates)}
            cache = {}

            def expression(value, stage):
                key=(value.id,stage)
                if key in cache:return cache[key]
                result=[{} for _ in range(stage+1)]
                if value is initial:result[0][0]=Fraction(1)
                elif value.id in index:
                    i=index[value.id]
                    if i>=stage:raise ValueError("accepted stage reads a nonprevious rate")
                    result[i+1][0]=Fraction(1)
                else:
                    if value.op=='acceptance_guard':
                        result=expression(successful_guard_input(value),stage)
                        cache[key]=result;return result
                    if value.op != "linear_combine":
                        raise ValueError("accepted affine input lacks an exact coefficient expression")
                    for child,coeffs in zip(value.inputs,value.attrs['coeffs'],strict=True):
                        terms=expression(child,stage)
                        for power,numerator,denominator in _exact_terms(coeffs):
                            if type(power) is not int or power<0:
                                raise ValueError("accepted coefficient has a noninteger/negative dt power")
                            factor=Fraction(numerator,denominator)
                            for target,term in zip(result,terms,strict=True):
                                for degree,amount in term.items():
                                    target[degree+power]=target.get(degree+power,Fraction())+factor*amount
                cache[key]=result;return result

            def affine(value,stage):
                terms=[{p:x for p,x in row.items() if x} for row in expression(value,stage)]
                if set(terms[0])-{0} or any(set(row)-{1} for row in terms[1:]):
                    raise ValueError("accepted State/rate coefficients have incompatible dt powers")
                if terms[0].get(0,Fraction())!=1:
                    raise ValueError("accepted affine expression does not preserve its initial State")
                return tuple(row.get(1,Fraction()) for row in terms[1:])

            b=affine(accepted,len(ordered_rates))
            if not ordered_rates:
                continue  # its exact algebraic image is the initial State
            if evaluation_stage_fraction(accepted)!=1 or sum(b,Fraction())!=1:
                raise ValueError("accepted update does not span one consistent step")
            A=[];c=[]
            for i,rate in enumerate(ordered_rates):
                row=affine(rate.inputs[0],i)
                point=evaluation_stage_fraction(rate)
                if evaluation_stage_fraction(rate.inputs[0])!=point or sum(row,Fraction())!=point:
                    raise ValueError("rate/input stage coordinate differs from its exact affine row sum")
                A.append(row+(Fraction(),)*(len(b)-len(row)));c.append(point)
            base,alpha=_convex_coefficient_one(A,b)
            certificates.append(AcceptedUpdateSSP(graph_hash,accepted.id,
                tuple(value.id for value in ordered_rates),tuple(A),b,tuple(c),base,alpha))
        if not certificates:
            raise ValueError("accepted update has no contributing explicit rates")
        return (certificates[0] if len(certificates)==1 else AcceptedSSPPlan(tuple(certificates))), ""
    except (AttributeError,KeyError,TypeError,ValueError) as error:
        return None,str(error)
