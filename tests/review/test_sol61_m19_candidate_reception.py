"""Independent finite equations and source route; no native saved-state claim."""

import ast
from fractions import Fraction
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/python/integration/runtime/test_m19_product_support_runtime.py"
CPP = ROOT / "tests/cpp/integration/mpi/test_mpi_system_layout_transfer.cpp"
spec = importlib.util.spec_from_file_location(
    "m19_counter_reference", Path(__file__).with_name("sol61_m19_product_oracle.py")
)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def fixture_inputs():
    tree = ast.parse(FIXTURE.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(name, ast.Name) and name.id in ("NAMES", "CASES") for name in node.targets
        )
        or isinstance(node, ast.FunctionDef)
        and node.name == "input_and_expected"
    ]
    scope = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(FIXTURE), "exec"), scope)
    return scope


def receive_original(initial, actual, nx, nv, width, weights):
    expected_shape = (width, nx, nv)
    assert initial["population"].shape == expected_shape
    source = initial["population"]
    # No author sum helper or emitted provider is used for original equations.
    uniform = oracle.reduce_exact(source, ("x", "v"), ("x",), {"v": (Fraction(4, nv),) * nv})[
        :, None, :
    ]
    signed = oracle.reduce_exact(source, ("x", "v"), ("x",), {"v": weights})[:, None, :]
    lifted = oracle.lift(signed[:, 0, :], ("x",), ("x", "v"), (nx, nv))
    for name, value, shape in zip(
        ("population", "integral", "weighted", "extended"),
        actual,
        (expected_shape, (width, 1, nx), (width, 1, nx), expected_shape),
        strict=True,
    ):
        assert value.shape == shape and np.isfinite(value).all(), name
    for observed, original in zip(actual, (source, uniform, signed, lifted), strict=True):
        np.testing.assert_allclose(observed, original, rtol=0, atol=2e-12)


@pytest.mark.parametrize("nx,nv,width", [(4, 3, 3), (2, 5, 1), (7, 3, 5)])
def test_authored_affine_expectations_satisfy_independent_original_tensor_functionals(
    nx, nv, width
):
    scope = fixture_inputs()
    assert scope["CASES"] == ((4, 3, 3), (2, 5, 1), (7, 3, 5))
    weights = tuple(Fraction((-1) ** j * (j + 1)) for j in range(nv))
    initial, expected = scope["input_and_expected"](nx, nv, width, tuple(int(w) for w in weights))
    receive_original(initial, expected, nx, nv, width, weights)
    for c in range(width):
        for x in range(nx):
            for j in range(nv):
                assert (
                    initial["population"][c, x, j]
                    == (c + 1) * (x + 1) + (c + 2) * j + (-1) ** c * x * j
                )


@pytest.mark.parametrize(
    "attack",
    [
        "unweighted",
        "absolute_weights",
        "normalize",
        "lost_component",
        "stale_sentinel",
        "wrong_axis",
        "uniform_measure",
        "lift_component_zero",
    ],
)
def test_artificial_countermodels_refused_by_original_equations(attack):
    nx, nv, width = 4, 3, 3
    weights = tuple(Fraction((-1) ** j * (j + 1)) for j in range(nv))
    initial, expected = fixture_inputs()["input_and_expected"](
        nx, nv, width, tuple(int(w) for w in weights)
    )
    actual = [value.copy() for value in expected]
    if attack in ("unweighted", "absolute_weights"):
        bad_weights = (1,) * nv if attack == "unweighted" else tuple(abs(w) for w in weights)
        actual[2] = oracle.reduce_exact(
            initial["population"], ("x", "v"), ("x",), {"v": bad_weights}
        )[:, None, :]
        actual[3] = oracle.lift(actual[2][:, 0, :], ("x",), ("x", "v"), (nx, nv))
    elif attack == "normalize":
        actual[2] /= float(sum(weights))
        actual[3] /= float(sum(weights))
    elif attack == "lost_component":
        actual[2] = actual[2][::-1]
        actual[3] = actual[3][::-1]
    elif attack == "stale_sentinel":
        actual[2] = initial["weighted"]
    elif attack == "wrong_axis":
        actual[3] = actual[3].transpose(0, 2, 1)
    elif attack == "uniform_measure":
        actual[1] = oracle.reduce_exact(
            initial["population"], ("x", "v"), ("x",), {"v": (1,) * nv}
        )[:, None, :]
    else:
        actual[3][:] = actual[3][0]
    with pytest.raises(AssertionError):
        receive_original(initial, actual, nx, nv, width, weights)


def physical_dso_body(source):
    match = re.search(r'R"CPP\((.*?)\)CPP"', source, re.S)
    assert match, "authentic DSO source literal missing"
    dso = match.group(1)
    start = dso.index("if (request->operation == POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1")
    end = dso.index("if (request->source.scalar_type", start)
    return dso, dso[start:end]


def receive_dso_route(source):
    dso, body = physical_dso_body(source)
    assert "#include <pops/runtime/dynamic/physical_support_transfer.hpp>" in dso
    assert body.count("apply_physical_support_transfer(descriptor, request, status)") == 1
    assert "return pops::component::apply_physical_support_transfer" in body
    assert "request->source.extents[reduced_axis]" in body
    assert "weights.assign(descriptor.reduction_cells[reduced_axis], 1.0)" in body
    assert "descriptor.source_to_target[0] = static_cast<int>(kDimension - 1)" in body
    assert "descriptor.source_to_target[kDimension - 1] = 0" in body
    assert "descriptor.weights = weights.data()" in body
    assert "for (" not in body and "static_cast<double*>" not in body
    # The emitted literal must actually reach the shared DSO compiler/loader.
    assert "output << transfer_component_source(Dim)" in source
    assert "pops::test::native_dso::compile_shared(source, library)" in source


def test_mpi_dso_op2_op3_call_shipped_kernel_not_fixture_scalar_loop():
    receive_dso_route(CPP.read_text())


@pytest.mark.parametrize(
    "attack",
    [
        "private_loop",
        "wrong_axis",
        "unwired_weights",
        "missing_header",
        "wrong_reduction_extent",
        "unpublished_source",
    ],
)
def test_dso_source_route_mutants_refused(attack):
    source = CPP.read_text()
    old, new = {
        "private_loop": (
            "return pops::component::apply_physical_support_transfer(descriptor, request, status)",
            "for (int i=0; i<4; ++i) {} return 0",
        ),
        "wrong_axis": (
            "descriptor.source_to_target[0] = static_cast<int>(kDimension - 1)",
            "descriptor.source_to_target[0] = 0",
        ),
        "unwired_weights": ("descriptor.weights = weights.data()", "descriptor.weights = nullptr"),
        "missing_header": ("#include <pops/runtime/dynamic/physical_support_transfer.hpp>", ""),
        "wrong_reduction_extent": (
            "request->source.extents[reduced_axis]",
            "request->source.extents[0]",
        ),
        "unpublished_source": (
            "output << transfer_component_source(Dim)",
            "output << fake_source(Dim)",
        ),
    }[attack]
    assert old in source
    with pytest.raises(AssertionError):
        receive_dso_route(source.replace(old, new, 1))


@pytest.mark.parametrize("same", [False, True])
def test_authentic_composite_checkpoint_target_guard_requires_exact_common_path(same):
    source = ROOT / "python/pops/runtime/_multi_layout_executor.py"
    tree = ast.parse(source.read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Raise)
            and "multi-layout checkpoint target differs across ranks" in ast.unparse(child)
            for child in node.body
        )
    ]
    assert len(guards) == 1
    root = "/tmp/rank-0/m19-product-receipts/accepted.npz"
    peer = root if same else "/tmp/rank-1/m19-product-receipts/accepted.npz"
    scope = {"target": Path(root), "rows": [{"value": root}, {"value": peer}]}
    check = compile(ast.Module(body=guards, type_ignores=[]), str(source), "exec")
    if same:
        exec(check, scope)
    else:
        with pytest.raises(
            ValueError, match="^multi-layout checkpoint target differs across ranks$"
        ):
            exec(check, scope)
