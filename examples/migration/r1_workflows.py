"""Run eight scientific Python workflows through the public PoPS lifecycle.

Every model and numerical method is in scientific/<workflow>.py. Shared code is
limited to this CLI and runtime measurements. Use --scheme inline/imported with
scalar-amr to execute the same user method from two ordinary Python definitions.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

# Support direct script execution and package imports.
if __package__:
    from .scientific import (
        scalar_amr,
        field_transport,
        euler_poisson,
        heterogeneous_interaction,
        explicit_diffusion,
        implicit_diffusion,
        variable_coefficient_field,
        imported_native_primitive,
    )
else:
    from scientific import (
        scalar_amr,
        field_transport,
        euler_poisson,
        heterogeneous_interaction,
        explicit_diffusion,
        implicit_diffusion,
        variable_coefficient_field,
        imported_native_primitive,
    )

DEFAULT_CELLS = 16


@dataclass(frozen=True, slots=True)
class Workflow:
    run: Callable[[argparse.Namespace], dict[str, Any]]
    summary: str


WORKFLOWS = {
    "scalar-amr": Workflow(
        scalar_amr.run, "scalar AMR with explicit user SSPRK2, inline or imported"
    ),
    "field-transport": Workflow(
        field_transport.run, "transport driven by a consumed Poisson field"
    ),
    "euler-poisson": Workflow(
        euler_poisson.run, "Euler momentum driven by a consumed Poisson field"
    ),
    "heterogeneous-interaction": Workflow(
        heterogeneous_interaction.run, "exchange over unequal species"
    ),
    "explicit-diffusion": Workflow(
        explicit_diffusion.run, "periodic heat equation with Forward Euler"
    ),
    "implicit-diffusion": Workflow(
        implicit_diffusion.run, "periodic heat equation with implicit stage"
    ),
    "variable-coefficient-field": Workflow(
        variable_coefficient_field.run, "nonconstant coefficient Poisson"
    ),
    "imported-native-primitive": Workflow(
        imported_native_primitive.run, "Python advection with external arithmetic"
    ),
}


# Compatibility names only dispatch; physics lives in the scientific modules.
def _field_consumer_case(cells, *, transport):
    return (field_transport if transport else euler_poisson).build_case(cells)


def _diffusion_case(cells, *, implicit):
    return (implicit_diffusion if implicit else explicit_diffusion).build_case(cells)


_heterogeneous_case = heterogeneous_interaction.build_case
_variable_field_case = variable_coefficient_field.build_case


def _legacy_interface():
    # Existing protocol evidence can still exercise the original interface until
    # native qualification authorizes its removal. The public CLI uses Python law.
    if __package__:
        from .scientific import legacy_interface_flux
    else:
        from scientific import legacy_interface_flux
    return legacy_interface_flux


def _native_flux_source_component(work_dir):
    return _legacy_interface()._native_flux_source_component(work_dir)


def _native_flux_component(work_dir):
    return _legacy_interface()._native_flux_component(work_dir)


def _imported_primitive_case(cells, component):
    return _legacy_interface()._imported_primitive_case(cells, component)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", nargs="?", choices=tuple(WORKFLOWS))
    parser.add_argument("--list", action="store_true", help="list workflow names and requirements")
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--scheme",
        choices=("inline", "imported"),
        default="inline",
        help="select the scalar AMR user method definition",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/migration-r1"),
        help="generated component and delegated-example output directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        for name, workflow in WORKFLOWS.items():
            print("%-29s %s" % (name, workflow.summary))
        return
    if args.workflow is None:
        raise SystemExit("choose a workflow or pass --list")
    if args.cells is None:
        args.cells = 128 if args.workflow == "scalar-amr" else DEFAULT_CELLS
    if args.steps is None and args.workflow != "scalar-amr":
        args.steps = 1
    if isinstance(args.cells, bool) or args.cells < 4:
        raise SystemExit("--cells must be an integer >= 4")
    if args.steps is not None and (isinstance(args.steps, bool) or args.steps < 1):
        raise SystemExit("--steps must be an integer >= 1")
    args.work_dir = args.work_dir.resolve()
    result = WORKFLOWS[args.workflow].run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
