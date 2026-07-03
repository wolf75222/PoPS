"""AMR compile branch of ``pops.compile`` -- one native loader per block, no time Program.

Split out of :mod:`pops.codegen.orchestration` for the 500-line cap (ADC-550): the two AMR
helpers ``_compile_amr`` / ``_compile_block_amr`` that lower each block of an ``AMR`` layout to a
``target='amr_system'`` production ``CompiledModel`` (the native ``add_native_block`` loader),
carrying the ``{block: CompiledModel}`` table on the handle. ``orchestration.compile`` imports
``_compile_amr`` lazily (like its other in-function imports) so the module load order stays acyclic.

The shared resolve / field-solver / snapshot helpers stay in ``orchestration`` and are reused here;
this module adds no new codegen and no forbidden cross-layer import (the runtime / mesh / problem
types are reached only through those reused helpers).
"""

from pops.codegen.orchestration import (
    _freeze_and_snapshot,
    _problem_field_solvers,
    _resolve_problem_model,
)


def _compile_amr(problem, layout, backend, target, **kwargs):
    """Compile each AMR block to a ``target='amr_system'`` ``CompiledModel`` (single AND multi block).

    There is no whole-system time Program on AMR: each block's resolved physics model is compiled to a
    production native loader (``Model.compile(backend='production', target='amr_system')`` -> a
    ``CompiledModel`` whose adder is ``add_native_block``), and the ``{block: CompiledModel}`` table is
    carried on ``_block_compiled_models`` for bind's ``_assemble_instances`` to install each block via
    ``add_equation``. The returned handle is the FIRST block's ``CompiledModel`` -- it carries a real
    ``.so_path`` so bind's so_path guard passes, and ``_target`` / ``_layout`` / ``_problem`` /
    ``_block_compiled_models`` are attached for the AMR dispatch.

    Only the compile-driver kwargs ``Model.compile`` accepts are forwarded (include / cxx / std /
    name / require_metadata / hoist_reciprocals); a whole-system kwarg like ``force`` / ``debug`` /
    ``libraries`` has no per-block-loader equivalent and is dropped (the AMR route does not build a
    Program .so). An explicit ``so_path`` is forwarded ONLY for a single block: pinning the SAME path
    for several blocks would make their loaders collide (one .so per block), so a multi-block Problem lets
    each block fall back to its model-hash-keyed cache path.
    """
    compile_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("include", "cxx", "std", "name", "require_metadata",
                               "hoist_reciprocals")}
    if "so_path" in kwargs and len(problem._blocks) == 1:
        compile_kwargs["so_path"] = kwargs["so_path"]
    block_compiled = {name: _compile_block_amr(name, spec["model"], backend, compile_kwargs)
                      for name, spec in problem._blocks.items()}
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
    compiled._outputs = list(problem._outputs or [])
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem.
    compiled._layout = layout
    _freeze_and_snapshot(problem, None, compiled)
    return compiled


def _compile_block_amr(name, physics, backend, compile_kwargs):
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


__all__ = ["_compile_amr", "_compile_block_amr"]
