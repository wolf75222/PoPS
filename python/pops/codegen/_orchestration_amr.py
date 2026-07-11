"""AMR compile branch of ``pops.compile`` -- one native loader per block, plus an optional Program.

Split out of :mod:`pops.codegen.orchestration` for the 500-line cap (ADC-550): the two AMR
helpers ``_compile_amr`` / ``_compile_block_amr`` that lower each block of an ``AMR`` layout to a
``target='amr_system'`` production ``CompiledModel`` (the native ``add_native_block`` loader),
carrying the ``{block: CompiledModel}`` table on the handle. When the effective time is a
whole-system ``Program``, ``_compile_amr`` ALSO compiles it with
``compile_problem(target='amr_system')`` and returns the resulting ``CompiledProblem`` (ADC-634:
``AmrSystem.install_program`` drives it per level over the hierarchy). ``orchestration.compile``
imports ``_compile_amr`` lazily (like its other in-function imports) so the module load order stays
acyclic.

The shared resolve / field-solver / snapshot helpers stay in ``orchestration`` and are reused here;
this module adds no new codegen and no forbidden cross-layer import (the runtime / mesh / problem
types are reached only through those reused helpers and the ``compile_problem`` driver).
"""

from __future__ import annotations

from typing import Any

from pops.codegen.orchestration import (
    _attach_problem_snapshot,
    _capture_runtime_declarations,
    _problem_field_solvers,
    _resolve_problem_model,
)


def _compile_amr(problem: Any, layout: Any, backend: Any, target: Any, time: Any = None, *,
                 problem_snapshot: Any, bind_schema: Any = None, **kwargs: Any) -> Any:
    """Compile each AMR block to a ``target='amr_system'`` ``CompiledModel``, plus an optional Program.

    Each block's resolved physics model is compiled to a production native loader
    (``Model.compile(backend='production', target='amr_system')`` -> a ``CompiledModel`` whose adder is
    ``add_native_block``), and the ``{block: CompiledModel}`` table is carried on
    ``_block_compiled_models`` for bind's ``_assemble_instances`` to install each block via
    ``add_equation`` (single AND multi block).

    When @p time is a whole-system ``Program`` (ADC-634), the Program is ALSO compiled for the AMR
    target -- ``compile_problem(model=<first block engine model>, time=time, target='amr_system')`` ->
    a ``CompiledProblem`` whose ``.so`` exports ``pops_install_program_amr`` -- and THAT handle is
    returned, with the per-block loaders on ``_block_compiled_models`` (symmetric to the Uniform route,
    which returns a CompiledProblem carrying ``_block_models``). Bind installs it on the hierarchy via
    ``AmrSystem.install_program`` (the seam ``_finish_program_install`` already drives). Without a
    Program (native per-block time policy) the returned handle is the FIRST block's ``CompiledModel``,
    byte-identical to before. Either shape carries a real ``.so_path`` (so bind's so_path guard passes)
    and ``_target`` / ``_layout`` / ``_problem`` / ``_block_compiled_models`` for the AMR dispatch;
    bind discriminates the two by the duck-typed ``getattr(compiled, "program", None)`` signal.

    Only the compile-driver kwargs ``Model.compile`` accepts are forwarded to the per-block loaders
    (include / cxx / std / name / require_metadata / hoist_reciprocals). The whole-system Program .so
    additionally takes the ``compile_problem`` subset (so_path / force / debug / libraries) via
    @p program_kwargs. An explicit per-block ``so_path`` is forwarded ONLY for a single block: pinning
    the SAME path for several blocks would make their loaders collide (one .so per block), so a
    multi-block Problem lets each block fall back to its model-hash-keyed cache path. On the Program
    shape ``so_path`` pins the PROGRAM artifact (a single .so), not a per-block loader.
    """
    compile_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("include", "cxx", "std", "name", "require_metadata",
                               "hoist_reciprocals")}
    if "so_path" in kwargs and len(problem._blocks) == 1 and time is None:
        # A single per-block loader may pin its path (the no-Program shape). With a Program present,
        # so_path pins the PROGRAM artifact instead (below), so the per-block loaders fall back to
        # their model-hash-keyed cache paths and do not collide with it.
        compile_kwargs["so_path"] = kwargs["so_path"]
    block_compiled = {name: _compile_block_amr(name, spec["model"], backend, compile_kwargs)
                      for name, spec in problem._blocks.items()}

    if time is not None:
        return _compile_amr_program(
            problem, layout, backend, target, time, block_compiled, problem_snapshot,
            kwargs, bind_schema=bind_schema)

    _, compiled = next(iter(block_compiled.items()))
    compiled._problem = problem
    compiled._target = target
    compiled._block_compiled_models = block_compiled
    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592, parity with the Uniform route): the AMR route is already
    # snapshot-safe for the per-block models (_block_compiled_models table), but the field solvers /
    # output policies / spatial are still re-read live at bind. Freeze them at compile so a Problem mutated
    # between compile and bind is caught by bind()'s drift check rather than silently rebound.
    compiled._block_specs = {name: {"model": block_compiled[name], "spatial": spec["spatial"]}
                             for name, spec in problem._blocks.items()}
    compiled._field_solvers = _problem_field_solvers(problem)
    compiled._outputs, compiled._diagnostics = _capture_runtime_declarations(problem)
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem.
    compiled._layout = layout
    compiled.bind_schema = bind_schema
    _attach_problem_snapshot(compiled, problem_snapshot)
    for child in block_compiled.values():
        child._seal()
    return compiled


def _compile_amr_program(problem: Any, layout: Any, backend: Any, target: Any, time: Any,
                         block_compiled: Any, problem_snapshot: Any, kwargs: Any,
                         bind_schema: Any = None) -> Any:
    """Compile the whole-system time Program for the AMR target and attach the AMR snapshot (ADC-634).

    Compiles @p time with ``compile_problem(model=<first block engine model>, target='amr_system')`` --
    the SAME driver + single lowering the Uniform route uses, folding ``target`` into the cache key --
    then attaches the per-block loaders and the compile-time snapshot (block specs / field solvers /
    outputs / diagnostics / layout) so bind lowers from the frozen truth and installs the Program on
    the hierarchy. Symmetric to the Uniform branch of ``orchestration.compile``.

    Only the ``compile_problem`` kwargs are forwarded (so_path / force / cxx / include / std / debug /
    libraries); the per-block ``Model.compile`` kwargs were already consumed by the loaders above.
    """
    from pops.codegen.compile_drivers import compile_problem

    program_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("so_path", "force", "cxx", "include", "std", "debug", "libraries")}
    # problem._blocks is a BlockRegistry (an items()-style mapping with no .values()); take the
    # FIRST block's spec as the codegen representative, exactly like the Uniform route.
    _, first_spec = next(iter(problem._blocks.items()))
    first_model = _resolve_problem_model(first_spec["model"])
    compiled = compile_problem(model=first_model, time=time, backend=backend,
                               target="amr_system", problem_snapshot=problem_snapshot,
                               **program_kwargs)
    compiled._problem = problem
    compiled._target = target
    compiled._block_compiled_models = block_compiled
    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592), symmetric to the Uniform route: freeze the per-block
    # models (their target='amr_system' loaders) + spatial, the field solvers, the output policies and
    # the declared diagnostics so bind lowers from the compile-time truth and a Problem mutated between
    # compile and bind is caught by bind()'s drift check.
    compiled._block_specs = {name: {"model": block_compiled[name], "spatial": spec["spatial"]}
                             for name, spec in problem._blocks.items()}
    compiled._field_solvers = _problem_field_solvers(problem)
    compiled._outputs, compiled._diagnostics = _capture_runtime_declarations(problem)
    # Carry the AMR layout so bind() rebuilds the AmrSystemConfig and flows the typed refinement, and
    # so the introspection (arguments / estimate_memory / inspect_amr) reports the AMR hierarchy.
    compiled._layout = layout
    compiled.bind_schema = bind_schema
    _attach_problem_snapshot(compiled, problem_snapshot)
    for child in block_compiled.values():
        child._seal()
    return compiled


def _compile_block_amr(name: Any, physics: Any, backend: Any, compile_kwargs: Any) -> Any:
    """Compile one AMR block's physics to a ``target='amr_system'`` ``CompiledModel``.

    Resolves the block's physics to the underlying engine model (``_resolve_problem_model``: a
    blackboard ``pops.physics.Model`` -> its ``.dsl`` engine, a ``pops.dsl.Model`` as-is) and calls its
    ``.compile(backend=..., target='amr_system', **compile_kwargs)``. A model that exposes no such
    ``.compile`` (e.g. a raw ``pops.model.Module``, which has no per-block native loader) raises a clear
    error rather than a cryptic ``AttributeError``.
    """
    model = _resolve_problem_model(physics)
    block_compile = getattr(model, "compile", None)
    if not callable(block_compile):
        raise NotImplementedError(
            "pops.compile: block %r resolves to a %r, which has no .compile(...) producing a "
            "target='amr_system' CompiledModel; an AMR block needs a pops.dsl.Model / "
            "pops.physics.Model physics (a raw pops.model.Module has no per-block AMR loader)."
            % (name, type(model).__name__))
    return block_compile(backend=backend, target="amr_system", **compile_kwargs)


__all__ = ["_compile_amr", "_compile_amr_program", "_compile_block_amr"]
