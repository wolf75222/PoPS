"""N10 lot 4 (DM3, CC7; addendum §2): a case runs alike without its storage constraints.

Four replicated cases of ``examples/api_v02`` are built twice, as written and with their
``ownership="replicated"`` dropped (``tests/python/support/unconstrained.py``), compiled, bound and
run to their end time on one rank. The plan chooses the same ``Replicated`` storage, so the decoded
accepted state is bitwise equal. The distributed half runs on MPI2 in
``tests/python/integration/mpi/test_unconstrained_storage_n10_native_mpi2.py``.
"""
import importlib

import numpy as np
import pops
import pytest

from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile
from tests.python.support.unconstrained import without_storage_constraints

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CASES = {
    "finite_bdf2_v14": ("examples.api_v02.finite_bdf2_v14", "build_finite_bdf2_v14"),
    "methods_patankar_network": ("examples.api_v02.methods_patankar_network", "build_t11_network"),
    "extension_closure_v14": ("examples.api_v02.extension_closure_v14",
                              "build_extension_closure_v14"),
    "finite_amr": ("examples.api_v02.finite_amr", "build_finite_amr"),
}


def _run(world, name, *, constrained):
    module, builder = CASES[name]

    def build():
        return getattr(importlib.import_module(module), builder)()

    def spec_of():
        if constrained:
            return build()
        with without_storage_constraints():
            return build()
    spec = collective_call(world, spec_of)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    route = "N10 lot 4 %s %s" % (name, "constrained" if constrained else "unconstrained")
    artifact = compile_profile_plan_once(world, plan, route=route, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={subject: value.copy()
                                  for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=64,
                                            console=False))
    image = collective_call(world, lambda: _image(runtime))
    return spec, image


# The issuer of a set instance is derived from the program identity, which differs because the
# source differs (the unconstrained build has no ``ownership=``); every value is compared.
_IDENTITY_KEYS = frozenset(("issuer",))


def _equal(left, right, where="root"):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype and left.shape == right.shape, where
        assert left.tobytes() == right.tobytes(), where
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), where
        for key in left:
            if key not in _IDENTITY_KEYS:
                _equal(left[key], right[key], "%s.%s" % (where, key))
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), where
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _equal(a, b, "%s[%d]" % (where, index))
    elif isinstance(left, float):
        assert isinstance(right, float), where
        assert np.float64(left).tobytes() == np.float64(right).tobytes(), (where, left, right)
    else:
        assert left == right, (where, left, right)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_replicated_case_runs_bitwise_alike_without_its_constraints(
        name, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N10 lot 4 unconstrained " + name)
    spec, written = _run(world, name, constrained=True)
    _, chosen = _run(world, name, constrained=False)
    with collective_check(world):
        before, after = _decode(spec, written), _decode(spec, chosen)
        assert before["root"]["step"] > 0
        _equal(before["root"], after["root"])
    record_property("steps", before["root"]["step"])
