"""Emit the native tags and binding tables for build-time C++ fixtures."""
import json
import operator
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("The generated AMR fixture targets require Python >=3.10")

try:
    import numpy
except ImportError as error:
    raise SystemExit("The generated AMR fixture targets require NumPy >=1.24") from error

if numpy.lib.NumpyVersion(numpy.__version__) < "1.24.0":
    raise SystemExit("The generated AMR fixture targets require NumPy >=1.24")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

import pops
from pops._ir.handle_expr import ValueExpr
from pops.amr import Coarsen, ConflictPolicy, EqualityPolicy, Hysteresis, Tag
from pops.codegen.program_emit_refinement_criterion import emit_refinement_criterion
from pops.identity import make_identity
from pops.identity.semantic import semantic_value
from pops.mesh._amr.refinement_criterion import resolve_refinement_criterion


def cpp_string(value):
    return json.dumps(value, ensure_ascii=True)


def write_if_changed(path, content):
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def main(output, dimension):
    mounts = (
        ("synthetic_history", ("tracer",), ("u",), "u", 1.03, False,
         "tests.synthetic-loader.clock"),
        ("synthetic_interface", ("tracer", "tracer2"), ("u",), "u", 1.03, False,
         "tests.synthetic-loader.clock"),
        ("synthetic_attempt", ("tracer",), ("u",), "u", 1.5, True, "clock.macro"),
        ("field_rhs_value", ("material",), ("u",), "u", 2.4, False, "test.field-rhs-v2.clock"),
        ("field_rhs_incidence", ("material",), ("u", "tag"), "tag", 2.4, False,
         "test.field-rhs-v2.clock"),
    )
    output.mkdir(parents=True, exist_ok=True)
    declarations = []
    for name, blocks, variables, variable, threshold, coarsen, clock in mounts:
        model = pops.Model(name)
        state = model.state("U", components=variables)
        case = pops.Case(name)
        handles = [case.block(block, model, states=(state,))[state] for block in blocks]
        checked = pops.validate(case)
        routes = [checked.resolve(handle) for handle in handles]
        read = routes[0]
        value = read if len(variables) == 1 else ValueExpr(read)[variable]

        def admit(candidate):
            if candidate is not read:
                raise ValueError(f"{name} fixture reads an unregistered state")

        def region(wrap, compare):
            return wrap(lambda cell: compare(cell.value(value), threshold)).predicate

        rules = (("refine", region(Tag, operator.gt)),)
        if coarsen:
            rules += (("coarsen", region(Coarsen, operator.lt)),)
        criterion = resolve_refinement_criterion(
            rules, hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS, dimension=dimension, admit_state=admit,
        )
        data = criterion.runtime_criterion_data({})
        payload = {
            "schema_version": 1,
            "program_type": "bound_amr_refinement_criterion",
            "criterion_identity": data["identity"],
            "fields": data["fields"],
            "parameters": data["parameters"],
            "minimum_cycles": data["hysteresis"]["min_cycles"],
        }
        provider = make_identity(
            "bound-amr-refinement-criterion", semantic_value(payload, where=name),
        ).token
        field, = data["fields"]
        observed_variable, = field["variables"]
        if (field["reach"] != [0] * dimension or data["parameters"] or data.get("model_parameters")
                or data["hysteresis"]["min_cycles"] != 0):
            raise ValueError(f"{name} fixture binding table requires one local parameter-free read")
        component, = field["components"]
        bindings = [data["identity"], field["state"], field["block"], observed_variable, clock, provider]
        bindings = ", ".join(map(cpp_string, bindings))
        declarations.append(f"inline constexpr FixtureCriterion {name}{{{bindings}, {component}}};")
        if len(routes) > 1:
            declarations.append(f"inline constexpr char {name}_second_state[] = "
                                f"{cpp_string(routes[1].qualified_id)};")
        write_if_changed(output / (name + ".cpp"),
                         emit_refinement_criterion(criterion, native_dimension=dimension))
    write_if_changed(output / "amr_fixture_criteria_generated.hpp",
                     "\n".join(declarations) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
