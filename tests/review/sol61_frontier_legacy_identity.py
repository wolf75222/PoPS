# ruff: noqa: E402 -- import only after selecting the authenticated source package
"""Print the same public FixedDt Program identities against an explicit source package."""

import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "python"))
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import FixedDt
from pops.math import ddt

assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"
frame = Rectangle("legacy", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
model = pops.Model("legacy_model", frame=frame)
state = model.state("U", components=("x", "y"))
x, y = state
force = model.source("rotation", on=state, value=(-y, x))
rate = model.rate("balance", equation=ddt(state) == force)
owner = pops.Case("legacy_case")
block = owner.block("field", model)
plan = DiscretizationPlan()
plan.rates.add(rate, StateStorage())
owner.numerics(plan, block=block)
program = pops.Program("legacy_program")
temporal = program.state(block[state])
initial = temporal.n
rhs = program.source(model.module.operator_handle("rotation"), initial)
candidate = program.value("candidate", initial + program.dt * rhs, at=temporal.next.point)
program.commit(temporal.next, candidate)
program.step_strategy(FixedDt(1.0))
owner.program(program)
serialized = program._serialize(include_provenance=False)
semantic = program._semantic_serialize()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


print(
    json.dumps(
        {
            "version": serialized["version"],
            "ir_hash": program._ir_hash(),
            "serialization_sha256": digest(serialized),
            "semantic_sha256": digest(semantic),
        },
        sort_keys=True,
    )
)
