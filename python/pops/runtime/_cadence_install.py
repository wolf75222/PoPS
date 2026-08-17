"""Install hold-then-catch-up and step_adaptive Programs on a live engine."""

from __future__ import annotations

from typing import Any, Mapping


def _record_block_time(engine: Any, name: str, time: Any) -> None:
    records = dict(getattr(engine, "_block_time_descriptors", {}))
    records[str(name)] = time
    engine._block_time_descriptors = records


def _treatment(time: Any) -> str:
    if time is None:
        return "explicit"
    kind = getattr(time, "kind", "explicit")
    token = str(kind).lower()
    if "imex" in token or getattr(time, "implicit_vars", None) or getattr(time, "implicit_roles", None):
        return "imex"
    return "explicit"


def _hold_blocks(handles: Mapping[str, Any], descriptors: Mapping[str, Any],
                 linear_rates: Mapping[str, float] | None) -> Any:
    from pops.time import HoldCatchupBlock

    rates = dict(linear_rates or {})
    blocks = []
    for name, handle in handles.items():
        time = descriptors.get(name)
        blocks.append(
            HoldCatchupBlock(
                handle,
                stride=int(getattr(time, "stride", 1) or 1),
                linear_rate=rates.get(name),
                implicit_vars=tuple(getattr(time, "implicit_vars", ()) or ()) if time else (),
                implicit_roles=tuple(getattr(time, "implicit_roles", ()) or ()) if time else (),
                kind=_treatment(time),
                name=name,
            )
        )
    return blocks


def _engine_clock(engine: Any) -> tuple[float, int]:
    native = getattr(engine, "_s", engine)
    time = getattr(engine, "time", None)
    if not callable(time):
        time = getattr(native, "time", None)
    macro = getattr(engine, "macro_step", None)
    if not callable(macro):
        macro = getattr(native, "macro_step", None)
    if not callable(time) or not callable(macro):
        raise RuntimeError("cadence install requires time() and macro_step() on the engine")
    return float(time()), int(macro())


def _seal_amr_program_checkpoint(engine: Any) -> None:
    """Seal the AMR Program accepted-state capacity the cadence hold path restores."""
    seal = getattr(engine, "_checkpoint_program_state_capacity", None)
    if not callable(seal):
        seal = getattr(getattr(engine, "_s", None), "_checkpoint_program_state_capacity", None)
    if not callable(seal):
        raise RuntimeError(
            "AMR cadence install requires the native Program checkpoint-capacity seal"
        )
    seal()


def _finish_cadence_program_install(engine: Any, program: Any, *, target: str) -> None:
    """Attach the compiled Program the same way bind's ``_finish_program_install`` does."""
    engine._step_strategy = getattr(program, "_step_strategy", None)
    transaction = getattr(program, "transaction_plan", None)
    engine._step_transaction_plan = transaction() if callable(transaction) else None
    temporal = getattr(engine, "_temporal_restart_state", None)
    configure = getattr(temporal, "configure_program", None)
    if callable(configure):
        accepted_time, macro_step = _engine_clock(engine)
        configure(program.temporal_manifest(), time=accepted_time, macro_step=macro_step)
    if target == "amr_system":
        _seal_amr_program_checkpoint(engine)


def install_hold_catchup(
    engine: Any,
    blocks: Any,
    *,
    model: Any,
    include: Any,
    cxx: Any,
    so_path: Any,
    native_dimension: Any = None,
    target: str | None = None,
) -> Any:
    """Compile and install one hold-then-catch-up Program on ``engine``."""
    from pops.codegen._compile_drivers import compile_problem
    from pops.runtime._system import AmrSystem
    from pops.time import hold_catchup_program

    program = hold_catchup_program(blocks)
    resolved_target = target
    if resolved_target is None:
        resolved_target = "amr_system" if isinstance(engine, AmrSystem) else "system"
    compiled = compile_problem(
        so_path=str(so_path),
        model=model,
        time=program,
        include=include,
        cxx=cxx,
        native_dimension=native_dimension,
        target=resolved_target,
    )
    contract = program.cadence_contract()
    if not contract.is_default:
        engine.set_program_cadence(contract.substeps, contract.stride)
    engine.install_program(compiled.so_path)
    _finish_cadence_program_install(engine, program, target=resolved_target)
    return program


def install_equation_cadence(
    engine: Any,
    handles: Mapping[str, Any],
    *,
    model: Any,
    include: Any,
    cxx: Any,
    so_path: Any,
    native_dimension: Any = None,
    linear_rates: Mapping[str, float] | None = None,
    target: str | None = None,
) -> Any:
    """Lower recorded ``add_equation(..., time=Explicit(stride=M))`` descriptors to a Program."""
    descriptors = dict(getattr(engine, "_block_time_descriptors", {}))
    missing = [name for name in handles if name not in descriptors]
    if missing:
        raise ValueError(
            "install_equation_cadence: add_equation has no time descriptor for %s" % missing
        )
    return install_hold_catchup(
        engine,
        _hold_blocks(handles, descriptors, linear_rates),
        model=model,
        include=include,
        cxx=cxx,
        so_path=so_path,
        native_dimension=native_dimension,
        target=target,
    )


def install_step_adaptive(
    engine: Any,
    handles: Mapping[str, Any],
    wave_speeds: Mapping[str, Any],
    *,
    model: Any,
    include: Any,
    cxx: Any,
    so_path: Any,
    native_dimension: Any = None,
    linear_rates: Mapping[str, float] | None = None,
    target: str | None = None,
) -> Any:
    """Install the oracle adaptive-stride Program (not a private native stepper)."""
    from pops.codegen._compile_drivers import compile_problem
    from pops.runtime._system import AmrSystem
    from pops.time import adaptive_strides, step_adaptive_program

    adaptive_strides(wave_speeds)
    program = step_adaptive_program(handles, wave_speeds, linear_rates=linear_rates)
    resolved_target = target
    if resolved_target is None:
        resolved_target = "amr_system" if isinstance(engine, AmrSystem) else "system"
    compiled = compile_problem(
        so_path=str(so_path),
        model=model,
        time=program,
        include=include,
        cxx=cxx,
        native_dimension=native_dimension,
        target=resolved_target,
    )
    contract = program.cadence_contract()
    if not contract.is_default:
        engine.set_program_cadence(contract.substeps, contract.stride)
    engine.install_program(compiled.so_path)
    _finish_cadence_program_install(engine, program, target=resolved_target)
    return program


def engine_min_cell_spacing(engine: Any, override: Any = None) -> float:
    if override is not None:
        value = float(override)
        if not (value > 0.0):
            raise ValueError("min_cell_spacing must be positive")
        return value
    native = getattr(engine, "cfl_min_dx", None)
    if native is None:
        native = getattr(getattr(engine, "_s", None), "cfl_min_dx", None)
    if callable(native):
        value = float(native())
        if value > 0.0:
            return value
    raise ValueError(
        "step_adaptive needs min_cell_spacing= or a native cfl_min_dx inspection method"
    )


def step_adaptive(
    engine: Any,
    cfl: Any,
    *,
    wave_speeds: Mapping[str, Any],
    min_cell_spacing: Any = None,
) -> float:
    """Advance one oracle macro-step ``dt = cfl * h / w_max`` through the installed Program."""
    from pops.time import adaptive_strides

    speeds = {str(name): float(speed) for name, speed in wave_speeds.items()}
    if not speeds:
        raise ValueError("step_adaptive requires wave_speeds")
    adaptive_strides(speeds)
    fastest = max(speeds.values())
    dt = float(cfl) * engine_min_cell_spacing(engine, min_cell_spacing) / fastest
    if not (dt > 0.0):
        raise ValueError("step_adaptive produced a non-positive macro-step")
    engine.step(dt)
    return dt


__all__ = [
    "_record_block_time",
    "engine_min_cell_spacing",
    "install_equation_cadence",
    "install_hold_catchup",
    "install_step_adaptive",
    "step_adaptive",
]
