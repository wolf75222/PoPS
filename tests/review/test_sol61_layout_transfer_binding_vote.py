"""Source ordering witness; genuine MPI rejection remains an SDK test."""
from pathlib import Path


def test_both_python_dtos_are_caught_and_voted_before_native_prepare():
    source = (Path(__file__).parents[2] / 'python/bindings/core/init/init_system.cpp').read_text()
    body = source.split('"_prepare_layout_transfer",', 1)[1].split('py::arg("target")', 1)[0]
    parse_spec = body.index('prepared_spec.emplace(layout_transfer_spec_from_python(spec))')
    parse_execution = body.index('prepared_execution.emplace(layout_transfer_execution_from_python(execution))')
    caught = body.index('preparation_error = std::current_exception()')
    vote = body.index('pops::all_reduce_max(preparation_error ? 1L : 0L, world.communicator())')
    enter = body.index('return PreparedSystemLayoutTransfer::prepare(')
    assert body.index('try {') < parse_spec < parse_execution < caught < vote < enter
    assert 'std::move(*prepared_spec)' in body[enter:]
    assert 'std::move(*prepared_execution)' in body[enter:]
    assert 'layout_transfer_spec_from_python' not in body[enter:]
    assert 'layout_transfer_execution_from_python' not in body[enter:]
    assert 'std::rethrow_exception(preparation_error)' in body[caught:enter]
