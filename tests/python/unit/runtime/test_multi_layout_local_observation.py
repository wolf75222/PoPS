"""Actual RuntimeInstance/composite routing with an explicit child-authority seam.

The child providers below substitute only native storage/queries. There is no
compile, MPI collective, native install, physical evolution or checkpoint claim.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._multi_layout_executor import _MultiLayoutUniformExecutor
from pops.runtime._runtime_instance import RuntimeInstance


class LocalProvider:
    def __init__(self, block, shape, boxes, *, width=3):
        self.block, self.shape, self.boxes = block, shape, boxes
        self.calls = []
        self.pieces = []
        for lower, upper in boxes:
            extents = tuple(hi - lo for lo, hi in zip(lower, upper, strict=True))
            native_shape = (width, *reversed(extents))
            self.pieces.append(np.arange(np.prod(native_shape), dtype=np.float64).reshape(native_shape))

    def _require_block(self, block):
        self.calls.append(("block", block))
        if block != self.block:
            raise KeyError("native child does not own block " + block)

    def spatial_shape(self):
        self.calls.append(("shape",))
        return self.shape

    def local_boxes(self, block):
        self._require_block(block)
        self.calls.append(("boxes", block))
        return self.boxes

    def local_state(self, block, index):
        self._require_block(block)
        self.calls.append(("state", block, index))
        return self.pieces[index]


def bound_observation_seam(providers):
    # These are the production classes at their published-child routing seam.
    # Binding/transfer installation itself is covered by the authority tests.
    composite = object.__new__(_MultiLayoutUniformExecutor)
    composite._engines = {"layout::" + name: child for name, child in providers.items()}
    composite._block_layouts = {name: "layout::" + name for name in providers}
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = composite
    return runtime, composite


@pytest.mark.parametrize("block,shape,boxes,width", [
    ("x", (11,), (((2,), (5,)), ((8,), (11,))), 1),
    ("xv", (11, 7), (((1, 2), (5, 6)),), 3),
    ("xyz", (8, 9, 10), (((2, 1, 3), (7, 8, 9)),), 5),
])
def test_exact_owner_dimension_and_native_piece_identity(block, shape, boxes, width):
    child = LocalProvider(block, shape, boxes, width=width)
    unrelated = LocalProvider("other", (2,), (((0,), (2,)),))
    runtime, composite = bound_observation_seam({"other": unrelated, block: child})
    assert runtime.local_boxes(block) == boxes
    assert composite.local_boxes(block) is boxes
    for index, piece in enumerate(child.pieces):
        assert runtime.local_state(block, index) is piece
        assert composite.local_state(block, index) is piece
        assert piece.dtype == np.float64
        assert piece.shape == (width, *reversed(tuple(b-a for a,b in zip(*boxes[index], strict=True))))
    assert unrelated.calls == []
    with pytest.raises(ValueError, match="multi-layout geometry requires"):
        runtime.spatial_shape()


def test_runtime_chooses_the_child_once_for_shape_and_boxes():
    first = LocalProvider("fluid", (7, 9), (((1, 2), (6, 8)),))
    second = LocalProvider("fluid", (9,), (((0,), (9,)),))
    selections = []
    def select(block):
        selections.append(block)
        return first if len(selections) == 1 else second
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = SimpleNamespace(executor_for_block=select)
    assert runtime.local_boxes("fluid") == first.boxes
    assert selections == ["fluid"]
    assert second.calls == []


@pytest.mark.parametrize("method", ["local_boxes", "local_state"])
def test_unknown_block_and_missing_layout_fail_without_fallback(method):
    child = LocalProvider("fluid", (7,), (((0,), (7,)),))
    runtime, composite = bound_observation_seam({"fluid": child})
    args = () if method == "local_boxes" else (0,)
    for target in (runtime, composite):
        with pytest.raises(KeyError, match="unknown RuntimeInstance block ghost"):
            getattr(target, method)("ghost", *args)
    composite._block_layouts["fluid"] = "layout::absent"
    with pytest.raises(KeyError, match="unknown RuntimeInstance block fluid"):
        getattr(runtime, method)("fluid", *args)
    assert child.calls == []


@pytest.mark.parametrize("shape,error", [((7.0,9), TypeError), ((True,9), TypeError),
    ((0,9), TypeError), ((7,-1), TypeError), ((), TypeError), ((1,2,3,4), TypeError), (None, TypeError)])
def test_selected_shape_is_validated_before_local_storage_query(shape, error):
    child = LocalProvider("fluid", (7, 9), (((0, 0), (7, 9)),))
    child.shape = shape
    runtime, _ = bound_observation_seam({"fluid": child})
    with pytest.raises(error):
        runtime.local_boxes("fluid")
    assert child.calls == [("shape",)]


@pytest.mark.parametrize("boxes,error,message", [
    ((((0,), (7,)),), TypeError, "exact rank 2"),
    ((((0,0), (7,9), (8,10)),), TypeError, "exact rank 2"),
    ((((0.0,0), (7,9)),), TypeError, "plain integer"),
    ((((False,0), (7,9)),), TypeError, "plain integer"),
    ((((0,0), (0,9)),), ValueError, "greater than"),
    ((((3,0), (2,9)),), ValueError, "greater than"),
])
def test_selected_box_bounds_keep_exact_existing_validation(boxes,error,message):
    child = LocalProvider("fluid", (7,9), ())
    child.boxes = boxes
    runtime, _ = bound_observation_seam({"fluid":child})
    with pytest.raises(error, match=message):
        runtime.local_boxes("fluid")


def test_empty_owner_has_no_rank_zero_piece_or_global_fallback():
    empty = LocalProvider("empty", (7,9), ())
    peer = LocalProvider("peer", (7,9), (((0,0),(7,9)),))
    runtime, composite = bound_observation_seam({"empty":empty,"peer":peer})
    assert runtime.local_boxes("empty") == ()
    assert composite.local_boxes("empty") == ()
    with pytest.raises(IndexError):
        runtime.local_state("empty", 0)
    assert peer.calls == []


@pytest.mark.parametrize("index", [True, -1, 0.0, "0"])
def test_bad_piece_index_fails_before_child_selection(index):
    child = LocalProvider("fluid", (7,), (((0,), (7,)),))
    runtime, _ = bound_observation_seam({"fluid":child})
    with pytest.raises(TypeError, match="non-negative integer"):
        runtime.local_state("fluid", index)
    assert child.calls == []


def test_child_exception_identity_and_single_layout_passthrough():
    poison = RuntimeError("actual child observation failure")
    def fail(*_args):
        raise poison
    child = SimpleNamespace(local_boxes=fail, local_state=fail, spatial_shape=lambda:(7,))
    runtime, _ = bound_observation_seam({"fluid":child})
    for method,args in (("local_boxes",()),("local_state",(0,))):
        with pytest.raises(RuntimeError) as raised:
            getattr(runtime, method)("fluid", *args)
        assert raised.value is poison
    single = LocalProvider("fluid", (7,9), (((1,2),(6,8)),))
    runtime._executor = single
    assert runtime.spatial_shape() == (7,9)
    assert runtime.local_boxes("fluid") == single.boxes
    assert runtime.local_state("fluid",0) is single.pieces[0]


@pytest.mark.parametrize("method", ["local_boxes", "local_state"])
def test_missing_child_observation_is_not_sourced_from_another_child(method):
    missing = SimpleNamespace(spatial_shape=lambda:(7,))
    peer = LocalProvider("peer", (7,), (((0,),(7,)),))
    runtime, _ = bound_observation_seam({"missing":missing,"peer":peer})
    args = () if method == "local_boxes" else (0,)
    with pytest.raises(NotImplementedError, match="does not expose rank-owned local"):
        getattr(runtime, method)("missing", *args)
    assert peer.calls == []


def test_two_blocks_on_one_layout_keep_distinct_native_routes_and_widths():
    boxes = {"fluid": (((0,0),(7,9)),), "load": (((1,2),(6,8)),)}
    pieces = {"fluid": np.arange(63.,dtype=np.float64).reshape(1,9,7),
              "load": np.arange(150.,dtype=np.float64).reshape(5,6,5)}
    calls = []
    def local_boxes(block):
        calls.append(("boxes",block))
        return boxes[block]
    def local_state(block,index):
        calls.append(("state",block,index))
        assert index == 0
        return pieces[block]
    child = SimpleNamespace(spatial_shape=lambda:(7,9),local_boxes=local_boxes,local_state=local_state)
    composite = object.__new__(_MultiLayoutUniformExecutor)
    composite._engines = {"shared-layout":child}
    composite._block_layouts = {"fluid":"shared-layout","load":"shared-layout"}
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = composite
    for block in ("load","fluid"):
        assert runtime.local_boxes(block) == boxes[block]
        assert runtime.local_state(block,0) is pieces[block]
    assert calls == [("boxes","load"),("state","load",0),("boxes","fluid"),("state","fluid",0)]


def test_wrong_child_does_not_invent_block_ownership_or_fallback():
    owner = LocalProvider("fluid",(7,),(((0,),(7,)),))
    peer = LocalProvider("peer",(7,),(((0,),(7,)),))
    runtime,composite = bound_observation_seam({"fluid":owner,"peer":peer})
    composite._block_layouts["fluid"] = "layout::peer"
    with pytest.raises(KeyError,match="native child does not own block fluid"):
        runtime.local_boxes("fluid")
    with pytest.raises(KeyError,match="native child does not own block fluid"):
        runtime.local_state("fluid",0)
    assert owner.calls == []
