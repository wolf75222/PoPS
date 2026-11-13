"""N11 lot 2b (spec 0.3 CC2, CN9, G07 criterion (ii)): owner names leave ``program_id``.

A case with two blocks and two models, compiled once and once with the case, the blocks and the
models renamed so that the alphabetical order of the blocks and of the models reverses, has one
``program_id``. The names stay in each manifest, and until name-free emission (lot 2c) the two
compiles still write two shared objects. Another optimization flag, or another emitter
(``codegen_source_authority``), gives another ``program_id``.

A checkpoint is authenticated by ``program_id`` first (lot 2b-ii): a checkpoint written under
another optimization flag is refused as another program, before any state changes (criterion (iv)).

A checkpoint written under the first names and restarted under the second has the same
``program_id``; it is refused by name (the missing member, and lot 2c whose name-free native images
let it restart), before any state changes: never bound to another block, never a raw ``KeyError``.
Lot 2c turns this refusal into a restart (``n11/lot2b_ii_design.md``, D5).
"""

import os
import tempfile

import numpy as np
import pytest

import pops
from tests.python.support.presentation_case import FIRST, RENAMED, resolve_named_case


def _compile(names, **options):
    return pops.compile(resolve_named_case(names, **options))


def _bind(artifact):
    from tests.python.support.native_execution_context import artifact_execution_context

    initial = {binding.subject: np.ones((1, 8, 8))
               for binding in artifact.plan.initial_condition_plan.bindings}
    return pops.bind(artifact, initial_values=initial,
                     resources={"execution_context": artifact_execution_context(artifact)})


def _restart_is_refused(source, target, blocks, message):
    """Checkpoint ``source`` after one step, restart ``target``: refused, nothing changed."""
    with tempfile.TemporaryDirectory() as tmp:
        sim = _bind(source)
        pops.run(sim, t_end=0.01, max_steps=1, output_dir=os.path.join(tmp, "run"),
                 console=False)
        checkpoint = sim.checkpoint(os.path.join(tmp, "source"))

        candidate = _bind(target)
        before = {block: np.array(candidate.state_global(block), copy=True) for block in blocks}
        clock = (candidate.time(), candidate.macro_step())
        with pytest.raises(ValueError, match=message):
            candidate.restart(checkpoint)
        for block, values in before.items():
            assert np.array_equal(candidate.state_global(block), values)
        assert (candidate.time(), candidate.macro_step()) == clock


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_renamed_owners_keep_the_program_id(
        isolated_native_cache, native_cxx, kokkos_root, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
    first = _compile(FIRST)
    renamed = _compile(RENAMED)

    assert first.program_id.domain == "program"
    assert first.program_id == renamed.program_id
    assert first.semantic_identity == renamed.semantic_identity
    assert first.plan.plan_identity == renamed.plan.plan_identity
    # The names are presentation: each artifact keeps its own, and the binaries still differ.
    assert [block.name for block in first.blocks] == list(FIRST["blocks"])
    assert [block.name for block in renamed.blocks] == list(RENAMED["blocks"])
    assert first.so_path != renamed.so_path
    assert first.inspect().to_dict()["artifacts"]["program_id"] == first.program_id.token

    # The driver's condition on lot 2b-i: a restart across the rename is refused by name, after
    # program_id matched; the native images still spell the names until lot 2c.
    _restart_is_refused(first, renamed, RENAMED["blocks"],
                        r"equal program_id.*no member state_eta, state_zeta.*native checkpoint "
                        r"images.*lot 2c")

    # A kept name (a term name, gap C52) changes it.
    assert _compile(FIRST, rate_name="other_balance").program_id != first.program_id

    # Another emitter changes it, IR unchanged.
    from pops.codegen import model_compile_evidence

    authority = model_compile_evidence.codegen_source_authority()
    monkeypatch.setattr(
        model_compile_evidence, "codegen_source_authority",
        lambda: {**authority, "sha256": "0" * 64})
    assert _compile(FIRST).program_id != first.program_id
    monkeypatch.undo()

    # Another optimization flag changes it (criterion (iv): the toolchain axis this host varies),
    # and a checkpoint of one is refused by the other as another program, before any member.
    monkeypatch.setenv("POPS_DSL_OPTFLAGS", "-O2")
    optimized = _compile(FIRST)
    assert optimized.program_id != first.program_id
    _restart_is_refused(optimized, first, FIRST["blocks"],
                        r"program identity \(program_id\) does not match the bound runtime")
