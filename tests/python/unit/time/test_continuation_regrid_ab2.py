"""Source contracts and a discriminating independent AB2 oracle."""
import numpy as np
import pytest

from examples.api_v02.continuation_regrid_amr import ab2_factors, build_continuation_regrid_amr
from pops.codegen.program_codegen import emit_cpp_program
from pops.time._program.detach import detach_compiled_program


def test_regridded_history_changes_the_next_ab2_step():
    factors = ab2_factors(.1, 3)
    np.testing.assert_allclose(factors, [1., 1.1, 1.215, 1.34225], rtol=0, atol=2*np.spacing(1.))
    # Replacing q^(n-1) with q^n at regrid silently changes AB2 into Euler.
    wrong = 1.1*factors[2]
    assert abs(factors[3]-wrong) > .005


@pytest.mark.parametrize("reject", (False, True))
def test_python_authors_both_field_transfers_and_the_parent_verdict(reject):
    program = build_continuation_regrid_amr(reject_regrid_parent=reject).case._time
    assert program.validate()
    source = emit_cpp_program(program, target="amr_system")
    assert source.count("ctx.import_field_bundle(") == 2
    assert "prepare_continuation_regrid_child" in source
    assert "install_continuation_regrid_graph" in source
    assert "continuation_child_dependencies" in source
    assert "continuation_source_topology" in source
    assert "continuation_target_topology" in source
    assert detach_compiled_program(program)._ir_hash() == program._ir_hash()
