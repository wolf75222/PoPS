"""Filesystem/JSON protocol primitives, not synthetic physical/native receipts."""
import os
from pathlib import Path
import subprocess

import pytest

from tests.review import sol61_t5_portable_archive_checker as checker
from tests.review import test_sol61_integral_feedback_offline_contract as historical


@pytest.mark.parametrize("value", ["", ".", "..", "../a", "/a", "a/../b", "a//b", "a/./b", "a/", "C:/a", "a:b", "a\\b", "a\x00b"])
def test_nonportable_or_aliased_paths_refused(value):
    with pytest.raises(ValueError, match="nonportable"):
        checker.portable_path(value)


def test_canonical_archive_path_preserved():
    name = "t5-sdk7b/positive/restart-n8/accepted-state.npz"
    assert checker.portable_path(name) == name


@pytest.mark.parametrize("raw", ['{"a":1,"a":1}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_duplicate_and_nonfinite_json_refused(raw):
    with pytest.raises(ValueError):
        checker.strict_json(raw)


def test_inventory_preserves_empty_directories_for_exact_contract(tmp_path):
    (tmp_path / "ordinary.txt").write_text("protocol primitive")
    (tmp_path / "empty").mkdir()
    files, directories = checker.regular_tree(tmp_path.absolute())
    assert files == {"ordinary.txt"} and directories == {"empty"}


def test_internal_same_content_symlink_is_never_a_regular_archive_file(tmp_path):
    target = tmp_path / "ordinary.txt"
    target.write_text("protocol primitive")
    (tmp_path / "alias.txt").symlink_to(target.name)
    with pytest.raises(ValueError, match="archive symlink entry"):
        checker.regular_tree(tmp_path.absolute())


def test_archive_root_itself_cannot_be_a_symlink(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target.name, target_is_directory=True)
    with pytest.raises(ValueError, match="archive root is a symlink"):
        checker.regular_tree(alias.absolute())


def test_fifo_is_refused_without_opening_it(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="archive nonregular entry"):
        checker.regular_tree(tmp_path.absolute())


def test_external_seal_mandatory_even_before_directory_access():
    with pytest.raises(ValueError, match="external archive seal"):
        checker.receive(Path("missing-data"), "")


def test_frozen_historical_source_and_oracle_are_not_auto_rebaselined():
    assert checker.FROZEN["oracle/sol61_integral_feedback_offline_oracle.py"] == "21041d64af5378a35b5a8717cc0300b6a399d95cc2c9209e66d66e5e2ed54f89"
    assert checker.OWNER_SHA == "7a6d07c39e434b256e08792ff6c89d991c8525068eb0df0c0a37e43c12521bc2"
    assert len(checker.EXTRAS) == 8


def _historical_fixture():
    return subprocess.check_output([
        "git", "show", "634cba3511fef957e65a21fcae3ab257f164b360:"
        "tests/python/integration/runtime/test_public_integral_feedback.py"], cwd=historical.ROOT).decode()


def test_fixture_extensions_do_not_redefine_the_historical_scientific_seams():
    frozen = _historical_fixture()
    extended = frozen.replace("_case(8,proposed_dt=.5)", "_case(8,proposed_dt=.5,physical_global=True)")
    extended += "\n# An additional reception does not change historical equations.\n"
    assert extended != frozen
    historical._assert_scientific_fixture_seams(extended, frozen)


def test_narrow_fixture_authentication_still_refuses_changed_physics():
    frozen = _historical_fixture()
    altered = frozen.replace("reaction = previous*(1.-GAMMA*q*dt)",
                             "reaction = previous*(1.-2*GAMMA*q*dt)")
    assert altered != frozen
    with pytest.raises(AssertionError):
        historical._assert_scientific_fixture_seams(altered, frozen)
