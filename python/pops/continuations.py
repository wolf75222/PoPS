"""Python construction library for native pure trials and authenticated acceptance.

This layer adds no model-specific decision to the compiler. Trial, assessment and retry
policies are closed GraphFunctions. Live identities and scoped publication belong to the native
continuation authority, never to a Python record or a caller-chosen version integer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import value_signature, success_signature
from pops.time._program.value_validation import require_owned
from pops.time._program.versioned_values import ContinuationSchema
from pops.time.variant_types import Bool, Failure, Int64, Product, Result, Variant, ValueType, type_of


def _origin_checked(program, body, origin):
    """Keep a subgraph's whole error under its actual call-boundary origin.

    No user tag is reserved or renamed. In particular, a child Advance's Reject
    remains inside its complete error payload and cannot impersonate the current
    assessment's control decision. Native failures are captured at the same boundary.
    """
    def evaluate(p):
        value=body(p)
        signature=value_signature(value)
        if signature['vtype']=='variant' and set(dict(signature['members']))=={'ok','err'}:
            return value
        return p.variant(Result(type_of(value),Failure()),'ok',value)
    result=program.capture(evaluate)
    signature=_result(result,origin)
    members=dict(signature['members'])
    errors=Failure(**{origin:ValueType(members['err'])})
    output=Result(ValueType(members['ok']),errors)
    wrapped=program.match(result,
        ok=lambda p,value:p.variant(output,'ok',value),
        err=lambda p,error:p.variant(output,'err',p.variant(errors,origin,error)))
    return program.require_success(wrapped)


def _closed(value, where):
    from pops.graph_functions import GraphFunction
    if type(value) is not GraphFunction:
        raise TypeError(where + " requires an exact closed GraphFunction")
    return value


def _result(value, where):
    signature = value_signature(value)
    if signature["vtype"] != "variant" or set(dict(signature["members"])) != {"ok", "err"}:
        raise TypeError(where + " must return a typed Result")
    return signature


@atomic_authoring
def Candidate(program, *, base, proposal, continuation, increment, observations, checks):
    """Freeze the complete envelope with the native exact base/proposal dependency."""
    for name, value in (("continuation", continuation), ("increment", increment),
                        ("observations", observations), ("checks", checks)):
        require_owned(program, value, "Candidate." + name, vtype="product")
    envelope = program.product(continuation=continuation, increment=increment,
                               observations=observations, checks=checks)
    return program.continuation_freeze(base, proposal, envelope)


@dataclass(frozen=True)
class Assess:
    """Closed graph: (base,proposal,candidate) -> Product(accepted:Bool,checks:Product)."""
    body: object

    def __post_init__(self):
        _closed(self.body, "Assess")


@dataclass(frozen=True)
class DecisionAssess:
    """The four-way assessment algebra of API 0.2, composed from closed graphs.

    body(base,proposal,candidate) returns exactly Variant(accept=Checks,
    reject=Product(controller_update=Controller,reason=Reason),
    transform=Variant(tag=Payload,...),stop=StopReason). The accept tag denotes
    the exact candidate bound to this assessment, by construction; its payload
    is evidence, never a substitute candidate. Checks and Controller are Products.

    Each static transform graph receives base,proposal,candidate,payload and
    returns Result of a complete candidate envelope. Its output is transformed
    through the native authority and must undergo another assessment. Stop and
    Reject retain their complete native payloads. No callback is executed at run time.

    Escaping errors from Trial, Transform and Assess retain their complete payload
    under trial_failure, transform_failure and assessment_failure respectively.
    This distinguishes a child's Reject from this assessment's Reject without
    restricting the tags or compositions inside supplied graphs.
    """
    body: object
    transforms: object
    _ticket: object = field(init=False, repr=False)

    def __post_init__(self):
        from pops.graph_functions import graph
        _closed(self.body,"DecisionAssess")
        signature=self.body.output.signature
        if signature.get("vtype")!="variant" or set(dict(signature["members"]))!={"accept","reject","transform","stop"}:
            raise TypeError("DecisionAssess must return exactly accept/reject/transform/stop")
        members=dict(signature["members"])
        if members["accept"]["vtype"]!="product":
            raise TypeError("DecisionAssess accept evidence must be a Product")
        rejected=members["reject"]
        if rejected["vtype"]!="product" or set(dict(rejected["members"]))!={"controller_update","reason"}:
            raise TypeError("DecisionAssess Reject requires controller_update and reason")
        if dict(rejected["members"])["controller_update"]["vtype"]!="product":
            raise TypeError("DecisionAssess controller_update must be a Product")
        transformed=members["transform"]
        if transformed["vtype"]!="variant":
            raise TypeError("DecisionAssess Transform requires a closed typed tag table")
        table=dict(self.transforms)
        if set(table)!=set(dict(transformed["members"])):
            raise ValueError("DecisionAssess must supply exactly every Transform graph")
        for name,body in table.items(): _closed(body,"DecisionAssess Transform "+name)
        object.__setattr__(self,"transforms",MappingProxyType(table))
        body=self.body
        @graph(inputs=body.inputs,output=Product(accepted=Bool,checks=Product(decision=body.output)))
        def once(p,args):
            decision=body(p,base=args['base'],proposal=args['proposal'],candidate=args['candidate'])
            approved=p.match(decision,accept=lambda q,e:q.boolean(True),
                reject=lambda q,e:q.boolean(False),transform=lambda q,e:q.boolean(False),stop=lambda q,e:q.boolean(False))
            return p.product(accepted=approved,checks=p.product(decision=decision))
        object.__setattr__(self,"_ticket",Assess(once))


@atomic_authoring
def assess(program, candidate, assessment):
    if type(assessment) is not Assess:
        raise TypeError("assess requires an Assess descriptor")
    return program.continuation_attest(candidate, assessment.body)


@atomic_authoring
def accept(program, candidate, assessment):
    """A rejected assessment is an observable typed Failure, never an accepted value."""
    from pops.time._program.versioned_values import _cap
    candidate_type = value_signature(candidate)
    assessment_type = value_signature(assessment)
    checks_type = assessment_type["checks"]
    errors = Failure(assessment_rejected=Product(checks=ValueType(checks_type)))
    accepted_type = ValueType(_cap("continuation_accepted", candidate=candidate_type, checks=checks_type))
    result_type = Result(accepted_type, errors)

    def accepted(p):
        return p.variant(result_type, "ok", p.continuation_accept(candidate, assessment))

    def rejected(p):
        checks = p.continuation_assessment_checks(assessment)
        payload = p.product(checks=checks)
        return p.variant(result_type, "err", p.variant(errors, "assessment_rejected", payload))

    return program.branch(program.continuation_assessment_ok(assessment), accepted, rejected)


@dataclass(frozen=True)
class Advance:
    """Closed trial and assessment; optional bounded retry is a supplied Python graph.

    Trial binds base and proposal, returning Result of the complete four-member envelope.
    A Trial declaring base as a continuation_base capability receives the exact native
    BaseHandle, so it may compose child Advances; otherwise it receives the read view.
    With legacy Assess, Retry binds base,proposal,controller,failure. With DecisionAssess,
    only Reject invokes Retry, binding base,proposal,controller=controller_update,reason.
    Both return Result(Product(proposal,controller)); neither can replace the scientific base.
    """
    trial: object
    assessment: Assess | DecisionAssess
    retry: object | None = None

    def __post_init__(self):
        _closed(self.trial, "Advance.trial")
        if type(self.assessment) not in (Assess,DecisionAssess):
            raise TypeError("Advance.assessment requires Assess or DecisionAssess")
        if self.retry is not None:
            _closed(self.retry, "Advance.retry")


@atomic_authoring
def advance_once(program, method, *, base, proposal, max_transforms=None):
    if type(method) is not Advance:
        raise TypeError("advance_once requires Advance")
    decisions=type(method.assessment) is DecisionAssess
    if decisions:
        require_owned(program,max_transforms,"DecisionAssess transformation budget",vtype="int64")
    elif max_transforms is not None:
        raise ValueError("max_transforms is only defined for DecisionAssess")

    def trial(p):
        def evaluate(q):
            base_type=dict(method.trial.inputs.signature['members']).get('base',{})
            argument=base if base_type.get('vtype')=='continuation_base' else q.continuation_read(base)
            result = method.trial(q, base=argument, proposal=proposal)
            _result(result, "Trial")
            return result
        envelope = (_origin_checked(p,evaluate,'trial_failure') if decisions
                    else p.require_success(evaluate(p)))
        candidate = p.continuation_freeze(base, proposal, envelope)
        if decisions:
            return _assess_decisions(p,candidate,method.assessment,max_transforms)
        receipt = assess(p, candidate, method.assessment)
        return p.require_success(accept(p, candidate, receipt))

    return program.continuation_capture(trial)


def _assess_decisions(p,candidate,assessment,budget):
    """One assessment evaluation per current version; finite native transform loop."""
    from pops.time._program.versioned_values import _cap
    from pops.time._program.structured_control import new_typed
    budget_error=Failure(decision_budget=Product(limit=Int64,used=Int64))
    def fail(q,output,error_type,tag,payload):
        return q.require_success(q.variant(Result(output,error_type),'err',q.variant(error_type,tag,payload)))
    candidate=p.require_success(p.branch(budget>=0,
        lambda q:q.variant(Result(type_of(candidate),budget_error),'ok',candidate),
        lambda q:q.variant(Result(type_of(candidate),budget_error),'err',q.variant(budget_error,'decision_budget',
            q.product(limit=budget,used=q.int64(0))))))
    first_ticket=_origin_checked(p,lambda q:assess(q,candidate,assessment._ticket),'assessment_failure')
    checked=value_signature(first_ticket)['checks']
    accepted_type=ValueType(_cap('continuation_accepted',candidate=value_signature(candidate),checks=checked))
    status=Variant(pending=Product(),accepted=accepted_type)
    carried_type=Product(candidate=type_of(candidate),used=Int64,status=status)
    decisions=dict(assessment.body.output.signature['members'])
    rejection=Failure(decision_reject=ValueType(decisions['reject']))
    stopping=Failure(decision_stop=ValueType(decisions['stop']))
    def dispatch(q,current,ticket,used):
        decision=q.continuation_assessment_checks(ticket)['decision']
        def approve(r,evidence):
            approved=r.continuation_accept(current,ticket)
            return r.product(candidate=current,used=used,status=r.variant(status,'accepted',approved))
        def transform(r,tagged):
            permitted=r.require_success(r.branch(used<budget,
                lambda z:z.variant(Result(type_of(used),budget_error),'ok',used),
                lambda z:z.variant(Result(type_of(used),budget_error),'err',z.variant(budget_error,'decision_budget',
                    z.product(limit=budget,used=used)))))
            def handler(body):
                def apply(z,payload):
                    def evaluate(w):
                        replacement=body(w,base=w.continuation_read(w.continuation_candidate_base(current)),
                            proposal=w.continuation_proposal(current),candidate=w.continuation_candidate_value(current),payload=payload)
                        _result(replacement,"DecisionAssess Transform")
                        return replacement
                    updated=z.continuation_transform(current,_origin_checked(z,evaluate,'transform_failure'))
                    return z.product(candidate=updated,used=permitted+z.int64(1),status=z.variant(status,'pending',z.product()))
                return apply
            return r.match(tagged,**{tag:handler(body) for tag,body in assessment.transforms.items()})
        return q.match(decision,accept=approve,transform=transform,
            reject=lambda r,data:fail(r,carried_type,rejection,'decision_reject',data),
            stop=lambda r,reason:fail(r,carried_type,stopping,'decision_stop',reason))
    initial=dispatch(p,candidate,first_ticket,p.int64(0))
    def step(q,state):
        current=state['candidate']
        ticket=_origin_checked(q,lambda r:assess(r,current,assessment._ticket),'assessment_failure')
        return dispatch(q,current,ticket,state['used'])
    def stop(q,state):
        return q.match(state['status'],accepted=lambda r,value:r.boolean(True),pending=lambda r,value:r.boolean(False))
    completed=p.require_success(p.iterate(initial,step,stop,max_steps=budget))['state']
    return p.match(completed['status'],accepted=lambda q,value:value,
        pending=lambda q,unused:new_typed(q,accepted_type.signature,'variant_unreachable',(completed['status'],),{},point=completed.point))


@atomic_authoring
def advance(program, method, *, base, proposal, controller=None, max_attempts=None, max_transforms=None):
    """Return Result(AcceptedProvisional,Failure); retries execute in native bounded regions."""
    if type(method) is not Advance:
        raise TypeError("advance requires Advance")
    if method.retry is None:
        if controller is not None or max_attempts is not None:
            raise ValueError("controller/budget require a declared closed retry policy")
        return advance_once(program, method, base=base, proposal=proposal,max_transforms=max_transforms)
    require_owned(program, controller, "retry controller", vtype="product")
    require_owned(program, max_attempts, "retry budget", vtype="int64")
    def outer(p):
        # Reject an invalid budget before executing even the first Trial.
        errors = Failure(continuation_budget=Product(value=Int64))
        budget_type = Result(type_of(max_attempts), errors)
        budget = p.require_success(p.branch(max_attempts > p.int64(0),
            lambda q: q.variant(budget_type, "ok", max_attempts),
            lambda q: q.variant(budget_type, "err", q.variant(errors, "continuation_budget",
                                                              q.product(value=max_attempts)))))
        # This is the actual first attempt, not an unexecuted type-witness operation.
        first = advance_once(p, method, base=base, proposal=proposal,max_transforms=max_transforms)
        accepted_payload = dict(value_signature(first)["members"])["ok"]
        status_type = Variant(pending=Product(), accepted=ValueType(accepted_payload))
        def finish(q, accepted):
            return q.product(proposal=proposal, controller=controller,
                             status=q.variant(status_type, "accepted", accepted))

        def retry(q, failure, old_proposal, old_controller):
            if type(method.assessment) is DecisionAssess:
                # Only Reject requests retry. Stop, transform/Trial failures and
                # exhausted budgets retain their exact payload and propagate.
                state_type=Product(proposal=type_of(proposal),controller=type_of(controller),status=status_type)
                error_type=type_of(failure)
                def propagate(r,unused):
                    return r.require_success(r.variant(Result(state_type,error_type),'err',failure))
                def rejected(r,payload):
                    answer=method.retry(r,base=r.continuation_read(base),proposal=old_proposal,
                        controller=payload['controller_update'],reason=payload['reason'])
                    _result(answer,"DecisionAssess retry policy")
                    updated=r.require_success(answer)
                    if set(dict(value_signature(updated)['members']))!={'proposal','controller'}:
                        raise TypeError("retry policy must return exactly proposal and controller")
                    return r.product(proposal=updated['proposal'],controller=updated['controller'],
                        status=r.variant(status_type,'pending',r.product()))
                arms={tag:propagate for tag,_ in value_signature(failure)['members']}
                if 'decision_reject' in arms: arms['decision_reject']=rejected
                return q.match(failure,**arms)
            answer = method.retry(q, base=q.continuation_read(base), proposal=old_proposal,
                                  controller=old_controller, failure=failure)
            _result(answer, "retry policy")
            updated = q.require_success(answer)
            members = dict(value_signature(updated)["members"])
            if set(members) != {"proposal", "controller"}:
                raise TypeError("retry policy must return exactly proposal and controller")
            return q.product(proposal=updated["proposal"], controller=updated["controller"],
                             status=q.variant(status_type, "pending", q.product()))

        initial = p.match(first, ok=finish,
                          err=lambda q, failure: retry(q, failure, proposal, controller))

        def step(q, state):
            attempt = advance_once(q, method, base=base, proposal=state["proposal"],max_transforms=max_transforms)
            return q.match(attempt,
                ok=lambda r, accepted: r.product(proposal=state["proposal"], controller=state["controller"],
                    status=r.variant(status_type, "accepted", accepted)),
                err=lambda r, failure: retry(r, failure, state["proposal"], state["controller"]))

        def stop(q, state):
            return q.match(state["status"], accepted=lambda r, value: r.boolean(True),
                            pending=lambda r, value: r.boolean(False))
        result = p.iterate(initial, step, stop, max_steps=budget - p.int64(1))
        state = p.require_success(result)["state"]
        from pops.time._program.structured_control import new_typed
        return p.match(state["status"], accepted=lambda q, value: value,
                       pending=lambda q, unused: new_typed(q, accepted_payload, "variant_unreachable",
                                                          (state["status"],), {}, point=state.point))

    return program.capture(outer)


__all__ = ("ContinuationSchema", "Candidate", "Assess", "DecisionAssess", "assess", "accept", "Advance", "advance_once", "advance")
