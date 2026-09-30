"""Independent extracted real public methods; host protocol, never native evidence."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "a6cf5c34f529c82a835984d80645ffadb33bb9d7"
PARENT = "ab1248be84b53118d30ffc798ab673c2218cb747"
FILES = ("python/pops/runtime/_runtime_instance.py", "python/pops/runtime/_multi_layout_executor.py")


def source(commit, filename):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + filename], text=True)


def extract(commit):
    scope = dict(Any=Any, Iterable=Iterable, cast=cast)
    instances = {}
    for filename, classname, names in (
        (FILES[0], "RuntimeInstance", {"_executor_for_block", "_executor_spatial_shape", "local_boxes", "local_state", "spatial_shape"}),
        (FILES[1], "_MultiLayoutUniformExecutor", {"executor_for_block", "executor_for_layout", "local_boxes", "local_state", "spatial_shape"}),
    ):
        text = source(commit, filename)
        tree = ast.parse(text)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"_require_iterable", "_require_exact_ints"}]
        methods = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers,
                ast.ClassDef(name=classname, bases=[], keywords=[], body=methods, decorator_list=[])]
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(module, filename, "exec"), scope)
        instances[classname] = scope[classname]
    return instances


classes, legacy = extract(CANDIDATE), extract(PARENT)


class NativeOwnerProbe:
    """Only native-call boundaries are substituted; no coordinates are synthesized by the relay."""
    def __init__(self, shape, rank, *, boxes=None, width=3):
        self.shape, self.rank = shape, rank
        self.boxes = tuple(boxes) if boxes is not None else (((0,) * len(shape), tuple(shape)),) if rank == 0 else ()
        self.array = np.arange(width * np.prod(shape), dtype="float64").reshape((width, *shape[::-1]))
        self.calls = []
        self.error = None

    def spatial_shape(self):
        self.calls.append(("shape", self.rank))
        return self.shape

    def local_boxes(self, block):
        self.calls.append(("boxes", block, self.rank))
        if self.error is not None:
            raise self.error
        return self.boxes

    def local_state(self, block, index):
        self.calls.append(("state", block, index, self.rank))
        if self.error is not None:
            raise self.error
        if index >= len(self.boxes):
            raise IndexError("native box does not belong to this rank")
        return self.array


def public(native, classes=classes):
    facade = classes["RuntimeInstance"]()
    facade._executor = native
    return facade


def composed(rank=0):
    engines = {"phase": NativeOwnerProbe((3, 4), rank), "moments": NativeOwnerProbe((7, 1), rank, width=5),
               "volume": NativeOwnerProbe((2, 5, 6), rank, width=5), "line": NativeOwnerProbe((9,), rank, width=1)}
    multi = classes["_MultiLayoutUniformExecutor"]()
    multi._engines = engines
    multi._block_layouts = dict(population="phase", integral="moments", weighted="moments", extended="phase",
                                line="line", volume="volume")
    return multi, engines


def test_exact_pinned_counterexample_parent_lacks_public_relay():
    multi = legacy["_MultiLayoutUniformExecutor"]()
    multi._engines = dict(phase=NativeOwnerProbe((3, 4), 0))
    multi._block_layouts = dict(population="phase")
    with pytest.raises(NotImplementedError, match="^this runtime provider does not expose rank-owned local boxes$"):
        public(multi, legacy).local_boxes("population")


@pytest.mark.parametrize("rank", (0, 1, 2))
@pytest.mark.parametrize("block,layout", (("population", "phase"), ("integral", "moments"), ("weighted", "moments"),
                                          ("extended", "phase"), ("line", "line"), ("volume", "volume")))
def test_real_methods_use_block_owner_dimension_and_native_rank(rank, block, layout):
    multi, engines = composed(rank)
    facade = public(multi)
    owner = engines[layout]
    boxes = facade.local_boxes(block)
    assert boxes == owner.boxes
    assert owner.calls == [("shape", rank), ("boxes", block, rank)]
    assert all(not engine.calls for name, engine in engines.items() if name != layout)
    owner.calls.clear()
    if rank == 0:
        assert facade.local_state(block, 0) is owner.array
    else:
        with pytest.raises(IndexError, match="^native box does not belong to this rank$"):
            facade.local_state(block, 0)
        assert boxes == ()  # No global/rank0 ownership fallback on empty peers.
    assert owner.calls == [("state", block, 0, rank)]


def test_direct_multi_relays_are_native_identity_preserving():
    multi, engines = composed()
    assert multi.local_boxes("weighted") is engines["moments"].boxes
    assert multi.local_state("weighted", 0) is engines["moments"].array
    assert engines["moments"].calls == [("boxes", "weighted", 0), ("state", "weighted", 0, 0)]


@pytest.mark.parametrize("method", ("local_boxes", "local_state"))
def test_foreign_block_or_stale_layout_refuses_without_touching_native(method):
    multi, engines = composed()
    facade = public(multi)
    args = (0,) if method == "local_state" else ()
    for block in ("foreign", "population"):
        if block == "population":
            multi._block_layouts[block] = "missing-owner-layout"
        with pytest.raises(KeyError, match="unknown RuntimeInstance block " + block):
            getattr(facade, method)(block, *args)
    assert all(not engine.calls for engine in engines.values())


@pytest.mark.parametrize("error", (RuntimeError("rank-local native failure"), OverflowError("native state overflow")))
@pytest.mark.parametrize("method", ("local_boxes", "local_state"))
def test_native_failures_are_not_masked_or_replaced(error, method):
    multi, engines = composed()
    engines["phase"].error = error
    args = (0,) if method == "local_state" else ()
    with pytest.raises(type(error)) as raised:
        getattr(public(multi), method)("population", *args)
    assert raised.value is error


@pytest.mark.parametrize("shape", ((True,), (2.0, 3), (0, 2), (-1,), (), (1, 2, 3, 4)))
def test_per_block_shape_remains_exact_fail_closed(shape):
    multi, engines = composed()
    engines["phase"].shape = shape
    with pytest.raises(TypeError, match="(?:exact integers|exact positive integers)"):
        public(multi).local_boxes("population")
    assert engines["phase"].calls == [("shape", 0)]


@pytest.mark.parametrize("boxes,error", ((((0, 0), (3,)), TypeError), (((True, 0), (3, 4)), TypeError),
                                         (((0, 0), (0, 4)), ValueError)))
def test_native_box_images_keep_exact_rank_integer_and_half_open_guards(boxes, error):
    multi, engines = composed()
    engines["phase"].boxes = (boxes,)
    with pytest.raises(error):
        public(multi).local_boxes("population")


@pytest.mark.parametrize("index", (True, -1, 1.0, np.int64(0)))
def test_invalid_index_refuses_before_owner_selection_or_native_call(index):
    multi, engines = composed()
    with pytest.raises(TypeError, match="non-negative integer"):
        public(multi).local_state("population", index)
    assert all(not engine.calls for engine in engines.values())


@pytest.mark.parametrize("rank", (0, 1))
def test_single_layout_backward_observation_identity_and_native_axis_order(rank):
    old, new = NativeOwnerProbe((4, 3), rank), NativeOwnerProbe((4, 3), rank)
    old_view, new_view = public(old, legacy), public(new)
    assert old_view.local_boxes("U") == new_view.local_boxes("U")
    assert old_view.spatial_shape() == new_view.spatial_shape() == (4, 3)
    if rank == 0:
        assert old_view.local_state("U", 0) is old.array
        assert new_view.local_state("U", 0) is new.array
        assert old.array.dtype == new.array.dtype and old.array.shape == new.array.shape
        assert old.array.tobytes() == new.array.tobytes()
    assert old.calls == new.calls


def test_ambiguous_global_geometry_still_refuses():
    multi, engines = composed()
    with pytest.raises(ValueError, match="executor_for_layout"):
        public(multi).spatial_shape()
    assert all(not engine.calls for engine in engines.values())


def test_pinned_unrelated_observation_bodies_are_byte_identical():
    for filename, names in ((FILES[0], ("get_state", "state_global", "_executor_for_block")),
                            (FILES[1], ("executor_for_block", "executor_for_layout", "state_global", "get_state", "set_state", "spatial_shape"))):
        def bodies(commit, filename=filename, names=names):
            text = source(commit, filename)
            return {node.name: hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
                    for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef) and node.name in names}
        assert bodies(PARENT) == bodies(CANDIDATE)
