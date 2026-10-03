"""Saved-data counterexamples to falsely receiving consumer-cursor identity."""
import json

import numpy as np
import pytest

from tests.review.sol61_checkpoint_cursor_join import join_checkpoint_cursors


def saved(tmp_path, value):
    path = tmp_path / "outer.npz"
    np.savez(path, runtime_consumer_cursors=value)
    return path


def test_cursor_join_preserves_distinct_layout_and_stage_receipts(tmp_path):
    expected = {"cursors": {"alpha": [0, 2], "omega": {"frame": 3}}, "schema": 1}
    path = saved(tmp_path, np.asarray(json.dumps(expected)))
    result = join_checkpoint_cursors(path, expected)
    assert result["consumer_cursors_CP_joined"] is True
    assert result["consumer_cursors"] == expected


@pytest.mark.parametrize("altered", [True, 0.0, 1, "0"])
def test_cursor_join_refuses_type_or_value_substitution(tmp_path, altered):
    path = saved(tmp_path, np.asarray('{"frame":0}'))
    with pytest.raises(ValueError, match="differ"):
        join_checkpoint_cursors(path, {"frame": altered})


def test_cursor_join_refuses_duplicate_checkpoint_keys(tmp_path):
    path = saved(tmp_path, np.asarray('{"frame":1,"frame":0}'))
    with pytest.raises(ValueError, match="duplicate"):
        join_checkpoint_cursors(path, {"frame": 0})


def test_cursor_join_refuses_nonscalar_carrier(tmp_path):
    path = saved(tmp_path, np.asarray(['{"frame":0}']))
    with pytest.raises(ValueError, match="Unicode scalar"):
        join_checkpoint_cursors(path, {"frame": 0})


def test_cursor_join_refuses_missing_authority(tmp_path):
    path = tmp_path / "outer.npz"
    np.savez(path, other=np.asarray('{}'))
    with pytest.raises(ValueError, match="missing"):
        join_checkpoint_cursors(path, {"frame": 0})


def test_cursor_join_refuses_nonfinite_json(tmp_path):
    path = saved(tmp_path, np.asarray('{"frame":NaN}'))
    with pytest.raises(ValueError, match="finite"):
        join_checkpoint_cursors(path, {"frame": float("nan")})
