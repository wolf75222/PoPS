"""Issued runtime intervals for the current uniform Program invocation.

Endpoints are real SSA, with a native attempt/clock sidecar. Immutable root
intervals may be read in collective branches. Historical windows are reconstructed
from the actual retained publication; AMR and arbitrary continuation transport remain
outside this implementation profile.
"""
from dataclasses import dataclass

from pops.time._authoring import authoring_transaction, require_program
from pops.time._program.value_validation import require_owned
from pops.time.values import ProgramValue
from pops._ir.symbolic import freeze_symbolic_metadata


def validate_history_interval_begin(value):
    if value.op != "history_interval_begin" or len(value.inputs) != 1:
        raise ValueError("historical interval requires its exact history dependency")
    source = value.inputs[0]
    require_owned(value.prog, source, "historical interval source", vtype="state")
    if source.op != "history" or source.state_ref is None:
        raise ValueError("historical interval requires an issued full-state history")
    expected = {"schema_version": 1, "clock": value.prog.clock.qualified_id,
                "role": "historical", "history": source.attrs["history"],
                "lag": source.attrs["lag"]}
    if dict(value.attrs) != expected:
        raise ValueError("historical interval differs from its exact history dependency")
    return source


@dataclass(frozen=True, slots=True, eq=False)
class Interval:
    begin: ProgramValue
    duration: ProgramValue
    end: ProgramValue
    role: str

    def __post_init__(self):
        if self.role not in ("requested", "reached"):
            raise ValueError("Interval role must be requested or reached")
        p = self.begin.prog
        for value in (self.begin, self.duration, self.end):
            require_owned(p, value, "Interval endpoint", vtype="scalar")
            if value.region:
                raise NotImplementedError("Interval currently requires the root region")
        if (self.begin.op != "interval_begin" or self.end.op != "interval_end"
                or tuple(v.id for v in self.end.inputs) != (self.begin.id, self.duration.id)
                or dict(self.begin.attrs) != {"schema_version": 1, "clock": p.clock.qualified_id, "role": self.role}
                or dict(self.end.attrs) != {"schema_version": 1}):
            raise ValueError("Interval requires endpoints issued by Program.interval")

    @property
    def program(self):
        return self.begin.prog

    def to_data(self):
        return {"schema_version": 1, "role": self.role,
                "clock": self.begin.attrs["clock"], "region": 0,
                "program": self.program.owner_path.canonical().to_data(),
                "begin": self.begin.id, "duration": self.duration.id, "end": self.end.id}

    def bind(self, program, value, *, name):
        """Guard this value's actual interval before it can be delivered."""
        require_owned(program, value, "interval delivery")
        if program is not self.program:
            raise ValueError("Interval belongs to another Program")
        if self.role != "reached":
            raise ValueError("delivery requires a reached Interval, not a requested window")
        with authoring_transaction(program):
            check = program._new("bool", "interval_check",
                (self.begin, self.duration, self.end), self.to_data(), name + "_interval_check", None)
            from pops.time.solve_outcome import FailRun
            return program.guard(name + "_interval", value, check, action=FailRun())


@dataclass(frozen=True, slots=True, eq=False)
class HistoryInterval:
    """One actual retained publication, issued lazily after its strict history read."""
    value: ProgramValue
    begin: ProgramValue
    duration: ProgramValue
    end: ProgramValue

    def __post_init__(self):
        p=self.value.prog
        require_owned(p,self.value,"HistoryInterval.value",vtype="state")
        if self.value.op!="history" or self.value.state_ref is None:
            raise TypeError("HistoryInterval requires an issued full-state history read")
        for node in (self.begin,self.duration,self.end):
            require_owned(p,node,"HistoryInterval endpoint",vtype="scalar")
        expected={"schema_version":1,"clock":p.clock.qualified_id,"role":"historical",
                  "history":self.value.attrs["history"],"lag":self.value.attrs["lag"]}
        if (self.begin.op!="history_interval_begin" or dict(self.begin.attrs)!=expected
                or tuple(v.id for v in self.begin.inputs)!=(self.value.id,)
                or self.duration.op!="history_interval_duration"
                or tuple(v.id for v in self.duration.inputs)!=(self.begin.id,)
                or dict(self.duration.attrs)!={"schema_version":1}
                or self.end.op!="interval_end"
                or tuple(v.id for v in self.end.inputs)!=(self.begin.id,self.duration.id)
                or dict(self.end.attrs)!={"schema_version":1}):
            raise ValueError("HistoryInterval requires exact issued publication dependencies")

    @property
    def program(self):return self.value.prog

    @property
    def role(self):return "historical"

    def to_data(self):
        return {"schema_version":1,"role":"historical","clock":self.begin.attrs["clock"],
                "program":self.program.owner_path.canonical().to_data(),
                "history":self.value.attrs["history"],"lag":self.value.attrs["lag"],
                "value":self.value.id,"begin":self.begin.id,"duration":self.duration.id,
                "end":self.end.id,"region":self.begin.region}

    def bind(self,program,value,*,name):
        require_owned(program,value,"historical interval value")
        if program is not self.program or value.id!=self.value.id:
            raise ValueError("historical interval requires its exact retained numerical carrier")
        with authoring_transaction(program):
            check=program._new("bool","history_interval_check",
                (self.begin,self.duration,self.end,self.value),self.to_data(),name+"_history_check",None,
                inherit_state_ref=False)
            from pops.time.solve_outcome import FailRun
            return program.guard(name+"_history_origin",value,check,action=FailRun())


def issue_history_interval(program,value):
    require_program(program,exact=False,where="Program.history_interval")
    require_owned(program,value,"history interval source",vtype="state")
    if value.op!="history" or value.state_ref is None:
        raise TypeError("HistoryInterval requires an issued full-state history read")
    with authoring_transaction(program):
        begin=program._new("scalar","history_interval_begin",(value,),
            {"schema_version":1,"clock":program.clock.qualified_id,"role":"historical",
             "history":value.attrs["history"],"lag":value.attrs["lag"]},"history_interval_begin",None,
             inherit_state_ref=False)
        duration=program._new("scalar","history_interval_duration",(begin,),
            {"schema_version":1},"history_interval_duration",None)
        end=program._new("scalar","interval_end",(begin,duration),
            {"schema_version":1},"history_interval_end",None)
        return HistoryInterval(value,begin,duration,end)


def issue_interval(program, *, duration=None, role="reached"):
    require_program(program, exact=False, where="Program.interval")
    if program._recording:
        raise NotImplementedError("Interval currently requires the root region")
    if role not in ("requested", "reached"):
        raise ValueError("Interval role must be requested or reached")
    with authoring_transaction(program):
        if duration is None:
            duration = program.requested_dt()
        require_owned(program, duration, "Interval.duration", vtype="scalar")
        if duration.region:
            raise NotImplementedError("Interval duration cannot escape a child region")
        begin = program._new("scalar", "interval_begin", (),
            {"schema_version": 1, "clock": program.clock.qualified_id, "role": role},
            "interval_begin", None)
        end = program._new("scalar", "interval_end", (begin, duration),
            {"schema_version": 1}, "interval_end", None)
        return Interval(begin, duration, end, role)


def validate_intervals(program):
    def walk(values,ancestry=()):
        for value in values:
            yield value,ancestry
            for key in ("cond_block","body_block","true_block","false_block","apply_block","residual_block"):
                yield from walk(value.attrs.get(key,()),(*ancestry,(value.op,key)))
    annotated=tuple(walk(program._values))
    values=tuple(value for value,_ in annotated)
    interval_ops = {"interval_begin", "interval_end", "interval_check",
                    "history_interval_begin","history_interval_duration","history_interval_check"}
    if not any(v.op in interval_ops for v in values):
        return
    # Native point/attempt sidecars do not yet cross a continuation or persistent
    # history. Fail before code generation rather than retaining bare timestamps.
    dependent = set()
    for value,ancestry in annotated:
        if value.op == "history_interval_begin":
            validate_history_interval_begin(value)
        elif value.op == "interval_check":
            if len(value.inputs) != 3:
                raise ValueError("Interval check requires begin/duration/end dependencies")
            begin,duration,end=value.inputs
            issued=Interval(begin,duration,end,begin.attrs.get("role"))
            if issued.role != "reached" or value.attrs != freeze_symbolic_metadata(issued.to_data()):
                raise ValueError("Interval check differs from its issued endpoint authority")
        elif value.op=="history_interval_check":
            if len(value.inputs)!=4:
                raise ValueError("historical interval check requires endpoints and the retained carrier")
            begin,duration,end,source=value.inputs
            issued=HistoryInterval(source,begin,duration,end)
            if value.attrs!=freeze_symbolic_metadata(issued.to_data()):
                raise ValueError("historical interval check differs from its issued publication")
        if value.op in interval_ops or any(v.id in dependent for v in value.inputs):
            if value.vtype in {"scalar", "bool", "int64", "product"}:
                dependent.add(value.id)
            if value.region and not all(op=="branch" and key in ("true_block","false_block") for op,key in ancestry):
                raise NotImplementedError("Interval capture is qualified only for collective branches")
            if value.op in {"product", "cache", "synchronize"}:
                raise NotImplementedError("Interval persistence/transport is not qualified")
    if any(value.op in {"subcycle", "post_synchronization", "layout_map_export", "layout_map_import"}
           or value.attrs.get("schedule") is not None for value in values):
        raise NotImplementedError("Interval version 1 requires one unscheduled uniform invocation")


__all__ = ["Interval","HistoryInterval"]
