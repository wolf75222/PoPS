"""Actual authored deep SSP/read closures preserve exact ownership and weights."""

from fractions import Fraction
import hashlib

import pytest

from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
from tests.python.support.guarded_field_diffusion_observation_case import (
    author_frozen_coefficient_case,
)
from pops._structural_data import canonical_json
from pops.time._program.equation_identity import _equation_value
from pops.time.solve_request import SolveRequestError


def test_equation_identity_preserves_pre_iterative_canonical_bytes():
    program, _ = author_frozen_coefficient_case()
    data = canonical_json(_equation_value(program, list(program._commits.values())))
    # Captured from the pre-iteration implementation on this unchanged fixture.
    assert len(data.encode()) == 42030
    assert hashlib.sha256(data.encode()).hexdigest() == (
        "791019415733915f74bd76b472a534bbe10f578d993431ef3026c72b9ac3b32a"
    )


def test_equation_identity_rejects_explicit_cycle_after_shared_reference():
    program, _ = author_frozen_coefficient_case()
    node = next(iter(program._commits.values()))
    object.__setattr__(node, "inputs", (node,))
    with pytest.raises(SolveRequestError, match="explicit SSA cycle"):
        _equation_value(program, [node, node])


def test_actual_deep_field_read_frozen_donor_and_affine_acceptance_keep_certificate():
    program, _ = author_frozen_coefficient_case(alias_depth=1024)
    certificate, reason = prove_accepted_update_ssp(program)
    assert certificate is not None, reason
    assert certificate.b == (Fraction(1),)
    assert certificate.c == (Fraction(0),)
    assert [weight for _, weight in accepted_diffusive_quadrature(program)] == [
        {1: Fraction(1)}
    ]


def test_shared_frozen_donor_dag_retains_edge_multiplicities_without_tree_expansion():
    program, _ = author_frozen_coefficient_case(shared_depth=30)
    certificate, reason = prove_accepted_update_ssp(program)
    assert certificate is not None, reason
    assert certificate.b == (Fraction(1),)
    reference = next(
        ref for ref in program._commits if "fixed-owner" in ref.block_ref.local_id
    )
    old = program._commits[reference]
    program._commits[reference] = program.value(
        "actually-doubled-frozen", 2 * old, at=old.point
    )
    certificate, reason = prove_accepted_update_ssp(program)
    assert certificate is None and reason
