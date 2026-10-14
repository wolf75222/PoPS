"""Inventory increments whose numerical values are actual native Program SSA.

These are authoring records, never a second numerical interpreter. Application,
weighting, BDF recurrence and replacement materialize ordinary PoPS expressions.
Spatial transfer must already have been realized; this module does not invent
an interpolation between unrelated supports or an AMR reflux policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.identity.semantic import semantic_value
from pops.identity.scalar import exact_numeric_scalar
from pops._ir.symbolic import freeze_symbolic_metadata
from pops.model.hash_data import canonical_hash_data
from pops.time._authoring import authoring_transaction, require_program
from pops.time._program.value_validation import require_owned
from pops.time.expressions import component_names
from pops.time.values import ProgramValue
from pops.intervals import Interval, HistoryInterval
from pops.finite_contributions import FiniteInterval, LinearMap, recipient_key


def _field(value, where):
    if not isinstance(value, ProgramValue) or not value.is_field():
        raise TypeError(where + " requires a real Program field value")
    if not component_names(value):
        raise ValueError(where + " requires an explicit component space")
    require_owned(value.prog, value, where)
    return value


def _inventory(value, where):
    if isinstance(value, ProgramValue) and value.is_field():
        return _field(value, where)
    from pops.finite_contributions import inventory
    return inventory(value, where)


def _ssa(value):
    return {"program": value.prog.owner_path.canonical().to_data(), "value": value.id}


@dataclass(frozen=True, slots=True, eq=False)
class Contribution:
    """One value shared by its deliveries; measure distinguishes rates/increments."""

    id: str
    value: ProgramValue
    interval: Interval | HistoryInterval | FiniteInterval
    measure: str = "increment"
    kind: str = "exchange"
    origin: Any = None

    def __post_init__(self):
        if type(self.id) is not str or not self.id:
            raise TypeError("contribution id must be a nonempty string")
        _inventory(self.value, "Contribution.value")
        if type(self.interval) not in (Interval, HistoryInterval, FiniteInterval):
            raise TypeError("contribution requires an issued runtime Interval, not endpoint tags")
        if self.interval.program is not self.value.prog:
            raise ValueError("contribution Interval belongs to another Program")
        if type(self.interval) is HistoryInterval and self.value.id != self.interval.value.id:
            raise ValueError("historical Contribution requires its exact retained carrier")
        pending, seen = [self.value], set()
        while pending:
            node = pending.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            if node.op == "history_interval_begin":
                from pops.intervals import validate_history_interval_begin
                validate_history_interval_begin(node)
                # This dependency reads publication metadata, not an unguarded
                # historical numerical contribution. Issuance is checked natively.
                continue
            if (node.op == "acceptance_guard" and len(node.inputs) == 2
                    and node.inputs[1].op == "history_interval_check"):
                check = node.inputs[1]
                begin, duration, end, carrier = check.inputs
                authority = HistoryInterval(carrier, begin, duration, end)
                if (node.inputs[0].id != carrier.id
                        or check.attrs != freeze_symbolic_metadata(authority.to_data())):
                    raise ValueError("historical contribution guard differs from its exact carrier")
                # Only this authenticated carrier is discharged. Other expression
                # inputs are still traversed and cannot borrow its authority.
                continue
            if node.op == "history":
                if type(self.interval) is not HistoryInterval or node.id != self.interval.value.id:
                    raise NotImplementedError("historical Contribution requires reconstructed HistorySampleIdentity")
            pending.extend(node.inputs)
        canonical_hash_data(self.origin)
        object.__setattr__(self, "origin", freeze_symbolic_metadata(self.origin))
        if self.measure not in ("rate", "increment"):
            raise ValueError("contribution measure must distinguish rate from increment")
        if self.kind not in ("exchange", "production", "defect"):
            raise ValueError("unknown contribution kind")

    def to_data(self):
        return {"id": self.id, "value": _ssa(self.value),
                "interval": self.interval.to_data(), "measure": self.measure,
                "kind": self.kind, "origin": canonical_hash_data(self.origin)}

    def pack(self, program):
        """Retain the actual value and window in a complete native continuation."""
        if type(self.interval) is not FiniteInterval:
            raise TypeError("finite contribution persistence requires an explicit FiniteInterval")
        require_owned(program, self.value, "persistent contribution")
        return program.product(value=self.value, interval=self.interval.pack(program))

    @classmethod
    def from_history(cls, identifier, packet, *, clock, measure="increment", kind="exchange"):
        """Reconstruct from actual accepted history; no new numeric value is invented."""
        return cls(identifier, packet["value"], FiniteInterval.unpack(packet["interval"], clock=clock),
                   measure, kind, {"historical_packet": _ssa(packet)})


@dataclass(frozen=True, slots=True, eq=False)
class Delivery:
    """A dimensionless linear map between co-located component inventories.

    Different blocks and component orders are allowed. Spatial interpolation or
    dimensional conversion requires a separately authored, typed realization.
    Unknown units (None on both ends) remain unknown, never dimensionless.
    """

    contribution: Contribution
    recipient: ProgramValue
    linear_map: object

    def __post_init__(self):
        if type(self.contribution) is not Contribution:
            raise TypeError("Delivery requires its exact Contribution object")
        _inventory(self.recipient, "Delivery.recipient")
        source = self.contribution.value
        if source.prog is not self.recipient.prog:
            raise ValueError("delivery and recipient belong to different Programs")
        if not self.recipient.is_field():
            from pops.graph_derivatives import tangent_type
            from pops.time.variant_types import type_of, ValueType
            from pops.time._program.structured_control import value_signature
            if type(self.linear_map) is not LinearMap:
                raise TypeError("finite Delivery requires an explicitly defined LinearMap")
            require_owned(source.prog, self.linear_map.at, "delivery linearization point")
            selected = dict(value_signature(self.linear_map.at)["members"])[self.linear_map.wrt]
            if not tangent_type(ValueType(selected)).matches(source):
                raise TypeError("contribution differs from the linear map direction type")
            if not tangent_type(self.linear_map.function.output).matches(self.recipient):
                raise TypeError("linear map codomain differs from the recipient type")
            return
        if self.recipient.state_ref is None:
            raise ValueError("delivery requires a qualified recipient state authority")
        rows = tuple(tuple(exact_numeric_scalar(x, where="delivery map") for x in row)
                     for row in self.linear_map)
        if (len(rows) != len(component_names(self.recipient))
                or any(len(row) != len(component_names(source)) for row in rows)):
            raise ValueError("delivery map dimensions differ from source/recipient spaces")
        for attribute in ("layout", "centering", "support", "sampling", "frame", "clock"):
            if getattr(source.space, attribute) != getattr(self.recipient.space, attribute):
                raise ValueError("delivery co-location obligation: different " + attribute)
        for i, row in enumerate(rows):
            for j, weight in enumerate(row):
                if weight != 0 and source.space.units[j] != self.recipient.space.units[i]:
                    raise ValueError("delivery dimensionless map connects incompatible component units")
        object.__setattr__(self, "linear_map", rows)

    def to_data(self):
        return {"contribution": self.contribution.id, "recipient": _ssa(self.recipient),
                "linear_map": (self.linear_map.to_data() if type(self.linear_map) is LinearMap
                               else canonical_hash_data(self.linear_map))}


@dataclass(frozen=True, slots=True, eq=False)
class Increment:
    contributions: tuple[Contribution, ...] = ()
    deliveries: tuple[Delivery, ...] = ()
    supersedes: tuple[str, ...] = ()

    def __post_init__(self):
        contributions, deliveries = tuple(self.contributions), tuple(self.deliveries)
        if any(type(c) is not Contribution for c in contributions):
            raise TypeError("Increment requires Contributions")
        if len({c.id for c in contributions}) != len(contributions):
            raise ValueError("a physical contribution cannot be inserted twice")
        ids = {id(c) for c in contributions}
        if any(type(d) is not Delivery or id(d.contribution) not in ids for d in deliveries):
            raise ValueError("each delivery must reference an exact contribution in this increment")
        if {id(d.contribution) for d in deliveries} != ids:
            raise ValueError("every contribution must have at least one recipient")
        keys = [(id(d.contribution), recipient_key(d.recipient)) for d in deliveries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate delivery to one recipient; combine the linear maps explicitly")
        programs = {id(c.value.prog) for c in contributions}
        if len(programs) > 1:
            raise ValueError("an increment cannot combine different Programs")
        if len({id(c.interval) for c in contributions}) > 1:
            raise ValueError("an increment requires one exact target Interval")
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "deliveries", deliveries)
        object.__setattr__(self, "supersedes", tuple(self.supersedes))

    def to_data(self):
        return {"contributions": tuple(c.to_data() for c in self.contributions),
                "deliveries": tuple(d.to_data() for d in self.deliveries),
                "supersedes": self.supersedes}

    @property
    def identity(self):
        return make_identity("program-increment-v02", semantic_value(
            self.to_data(), where="program increment"))

    def pack(self, program, *, bindings=()):
        packet = program.product(**{contribution.id: contribution.pack(program)
                                   for contribution in self.contributions})
        if all(type(c.interval) is FiniteInterval for c in self.contributions):
            from pops.ledger_provenance import bind_ledger
            return bind_ledger(program, packet, self, bindings)
        if bindings:
            raise NotImplementedError("authenticated persistent bindings require finite intervals")
        return packet

    def apply(self, program, base, *, name, at=None, byte_capacity=None):
        """Build base+B*J+P+d from these exact values and recipient maps."""
        require_program(program, exact=False, where="Increment.apply")
        _inventory(base, "increment base")
        require_owned(program, base, "increment base")
        if any(c.measure != "increment" for c in self.contributions):
            raise ValueError("cannot apply a rate without explicit temporal integration")
        if any(type(c.interval) is HistoryInterval for c in self.contributions):
            raise ValueError("historical increment needs an explicit weight into a reached Interval")
        if not base.is_field():
            if at is not None:
                raise ValueError("finite inventory endpoints belong in their explicit continuation")
            from pops.finite_contributions import apply_increment
            with authoring_transaction(program):
                return apply_increment(self, program, base, name=name, byte_capacity=byte_capacity)
        selected = tuple(d for d in self.deliveries if d.recipient.state_ref == base.state_ref)
        if self.contributions and not selected:
            raise ValueError("increment has no delivery for this recipient")
        components = [base[i] for i in range(len(component_names(base)))]
        for delivery in selected:
            value = delivery.contribution.value
            require_owned(program, value, "increment contribution")
            if base.space != delivery.recipient.space:
                raise ValueError("increment base differs from its exact recipient representation")
            # The dense authored map may have a different input block/component
            # width. Its target remains the exact recipient state authority.
            for i, row in enumerate(delivery.linear_map):
                for j, weight in enumerate(row):
                    if weight != 0:
                        components[i] = components[i] + weight * value[j]
        with authoring_transaction(program):
            # This scalar sidecar is consumed by an ordinary native guard. Its
            # attempt and endpoints are checked before any recipient is exposed.
            from pops.linalg.finite import FiniteSupport
            value = FiniteSupport(name + "_recipient", component_names(base)).bind(
                components).materialize(program, name, template=base, at=at)
            attrs = dict(value.attrs)
            attrs["increment_v02"] = self.to_data()
            attrs["increment_identity_v02"] = self.identity.token
            value = program._replace_value(value, attrs=attrs)
            return self.contributions[0].interval.bind(program, value, name=name) if self.contributions else value

    def require_application(self, value):
        """Candidate builders use this to reject a stale pre-replacement update."""
        _inventory(value, "increment application")
        if self.contributions and value.prog is not self.contributions[0].value.prog:
            raise ValueError("increment application belongs to a different Program")
        issued = value
        while issued.op == "acceptance_guard":
            issued = issued.inputs[0]
        if issued.attrs.get("increment_identity_v02") != self.identity.token:
            raise ValueError("candidate was not built from this exact increment version")
        return value

    def replace(self, old, new):
        """Return a new ledger; callers must rebuild dependent states/fields."""
        if not any(c is old for c in self.contributions):
            raise ValueError("replacement requires an exact member of this increment")
        if type(new) is not Contribution:
            raise TypeError("replacement requires a concrete Contribution")
        from pops.time._program.structured_control import value_signature, same_signature
        if (new.value.prog is not old.value.prog
                or new.measure != old.measure or new.interval != old.interval
                or not same_signature(value_signature(new.value), value_signature(old.value))):
            raise ValueError("replacement changes measure, interval or component representation")
        values = tuple(new if c is old else c for c in self.contributions)
        deliveries = tuple(Delivery(new, d.recipient, d.linear_map)
                           if d.contribution is old else d for d in self.deliveries)
        return Increment(values, deliveries, (*self.supersedes, self.identity.token))

    def scale(self, program, factor, *, name, interval, measure="increment", byte_capacity=None):
        """Weight numerical values once; new interval identities retain history."""
        require_program(program, exact=False, where="Increment.scale")
        if any(old.interval is not interval and not (
                type(old.interval) is HistoryInterval and type(interval) is Interval
                and interval.program is program and interval.role == "reached") and not (
                type(old.interval) is FiniteInterval and type(interval) is FiniteInterval
                and interval.program is program and old.interval.clock == interval.clock)
               for old in self.contributions):
            raise NotImplementedError("Interval relabeling requires an explicit history/transfer realization")
        with authoring_transaction(program):
            pairs = []
            for index, old in enumerate(self.contributions):
                require_owned(program, old.value, "scaled contribution")
                source = old.value
                if type(old.interval) is HistoryInterval:
                    source = old.interval.bind(program, source, name="%s_origin_%d" % (name, index))
                if source.is_field():
                    from pops.time.expressions import as_expression
                    value = program.value("%s_value_%d" % (name, index),
                                          as_expression(source) * factor, at=old.value.point)
                else:
                    from pops.finite_contributions import scale
                    source = old.interval.bind(program, source, name=name + "_origin")
                    value = scale(program, source, factor, byte_capacity=byte_capacity)
                new = Contribution("%s_%d" % (name, index), value, interval, measure,
                                   old.kind, {"previous": old.to_data()})
                pairs.append((old, new))
            deliveries = tuple(Delivery(new, d.recipient, d.linear_map)
                               for d in self.deliveries for old, new in pairs
                               if d.contribution is old)
            return Increment(tuple(new for _, new in pairs), deliveries)


def combine(*increments):
    """Disjoint union; shared references cannot masquerade as independent events."""
    if any(type(inc) is not Increment for inc in increments):
        raise TypeError("combine requires Increments")
    return Increment(tuple(c for inc in increments for c in inc.contributions),
                     tuple(d for inc in increments for d in inc.deliveries),
                     tuple(s for inc in increments for s in inc.supersedes))


def bdf2_increment(program, rate, previous, *, h, h_previous, solve_residual,
                   interval, name="bdf2"):
    """Compile h/alpha*F + beta/alpha*J_previous + R_n/alpha.

    R_n = alpha*Q_new-(1+r)*Q_old+beta*Q_prev-h*F_new is authored by
    the caller from the ORIGINAL equation and supplied as a defect Increment.
    An empty defect increment is an explicit exact-solve assumption. Previous
    contributions must already be loaded from actual method history by Program.
    Positive h/h_previous and compatible historical representations are method
    preconditions, not inferred from the BDF name.
    """
    for value in (rate, previous, solve_residual):
        if type(value) is not Increment:
            raise TypeError("BDF2 requires concrete rate/history/residual increments")
    if any(c.measure != "rate" for c in rate.contributions):
        raise ValueError("BDF2 current rate must have rate measure")
    if any(c.measure != "increment" for c in (*previous.contributions,
                                              *solve_residual.contributions)):
        raise ValueError("BDF2 history and solve residual must have inventory measure")
    if any(c.kind != "defect" for c in solve_residual.contributions):
        raise ValueError("BDF2 solve residual is a defect, not another physical exchange")
    from pops._ir.expr import is_scalar_expression
    from pops.time.values import _Coeff
    for step in (h, h_previous):
        if is_scalar_expression(step) or isinstance(step, _Coeff):
            continue  # Dynamic positivity is an explicit caller guard obligation.
        if isinstance(step, ProgramValue):
            require_owned(program, step, "BDF2 step length", vtype="scalar")
            continue
        if isinstance(step, bool) or exact_numeric_scalar(step, where="BDF2 step length") <= 0:
            raise ValueError("BDF2 step lengths must be positive")
    with authoring_transaction(program):
        if isinstance(h, ProgramValue) or isinstance(h_previous, ProgramValue):
            from pops.finite_contributions import checked
            if isinstance(h, ProgramValue):
                h = checked(program, h, h > 0, begin=h, end=h)
            if isinstance(h_previous, ProgramValue):
                h_previous = checked(program, h_previous, h_previous > 0,
                                     begin=h_previous, end=h_previous)
        r = h / h_previous
        alpha, beta = (1 + 2 * r) / (1 + r), r * r / (1 + r)
        return combine(
            rate.scale(program, h / alpha, name=name + "_rate", interval=interval),
            previous.scale(program, beta / alpha, name=name + "_history", interval=interval),
            solve_residual.scale(program, 1 / alpha, name=name + "_defect", interval=interval),
        )


__all__ = ["Contribution", "Delivery", "Increment", "combine", "bdf2_increment",
           "FiniteInterval", "LinearMap"]
