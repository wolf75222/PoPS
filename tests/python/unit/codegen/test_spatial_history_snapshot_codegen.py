"""Macro history declarations preserve real keeper authority through detach."""
from __future__ import annotations

import json

import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_spatial_interaction import emit_history_source_declarations
from pops.codegen.program_lowerability import all_ops
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from tests.python.support.spatial_interaction_receipts import build


def _arguments(line, method):
    """Read literal call arguments without splitting commas inside JSON strings."""
    text = line.split("ctx." + method + "(", 1)[1].removesuffix(");").rstrip(",")
    result = []
    decoder = json.JSONDecoder()
    while text:
        text = text.lstrip()
        if text.startswith('"'):
            argument, end = decoder.raw_decode(text)
            result.append(argument)
            text = text[end:]
        else:
            argument, separator, rest = text.partition(",")
            token = argument.strip().removesuffix("ULL")
            result.append(int(token) if token.lstrip("-").isdigit() else token)
            text = separator + rest
        text = text.lstrip().removeprefix(",")
    return result


@pytest.mark.parametrize("width", (1, 3))
def test_real_amr_keeper_snapshot_declaration_matches_issued_consumer(width):
    case, layout, _selected, _authored = build(width, adaptive=True, amr_periodic=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    storage_images = []
    for program in (plan.time, detach_compiled_program(plan.time)):
        before = program._serialize()
        source = emit_cpp_program(program, model_graph=graph, target="amr_system")
        declaration_lines = [line.strip() for line in source.splitlines()
                             if "ctx.declare_spatial_interaction_history_source(" in line]
        consumer_lines = [line.strip() for line in source.splitlines()
                          if "ctx.spatial_interaction_history_owned(" in line]
        history_values = [value for value in all_ops(program)
                          if value.op == "spatial_interaction" and
                          value.attrs["contract"] == "pops.spatial-interaction@2"]
        assert len(declaration_lines) == len(consumer_lines) == len(history_values) == 1
        declaration = _arguments(declaration_lines[0], "declare_spatial_interaction_history_source")
        consumer = _arguments(consumer_lines[0], "spatial_interaction_history_owned")
        assert declaration == [consumer[index] for index in (0, 1, 8, 9, 10, 11, 6, 12, 4, 5)]
        assert source.index(declaration_lines[0]) < source.index("ctx.install(")
        keeper = history_values[0].attrs["history_source"]["history"]
        assert declaration[2] == keeper
        assert source.index("ctx.register_history(%s" % json.dumps(keeper)) < source.index(declaration_lines[0])
        accepted = next(value for value in all_ops(program)
                        if value.op == "spatial_interaction" and
                        value.attrs["contract"] == "pops.spatial-interaction@1")
        assert accepted.id != declaration[0]
        assert emit_history_source_declarations(
            program, program._block_indices(), target="amr_system") == declaration_lines
        assert program._serialize() == before
        storage_images.append((declaration[0:4], declaration[8],
                               program._histories, program._histories_ncomp))
    assert storage_images[0] == storage_images[1]


@pytest.mark.parametrize("width", (1, 3))
def test_uniform_keeper_keeps_its_existing_history_route(width):
    case, layout, _selected, _authored = build(width)
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    for program in (plan.time, detach_compiled_program(plan.time)):
        source = emit_cpp_program(program, model_graph=graph)
        assert "ctx.spatial_interaction_history(" in source
        assert "declare_spatial_interaction_history_source" not in source
        assert emit_history_source_declarations(
            program, program._block_indices(), target="system") == []
