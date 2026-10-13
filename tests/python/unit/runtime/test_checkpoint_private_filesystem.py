"""Private checkpoint POSIX lifecycle; public paths retain strict atomic quarantine."""

import os
import stat
from pathlib import Path

import pytest

from pops.output._restart_provider import _CheckpointTransactionReceipt
from pops.output._writers import common


def _inode(path):
    value = Path(path).lstat()
    return int(value.st_dev), int(value.st_ino)


def test_private_entry_reseal_and_cleanup_without_rename_flags(tmp_path, monkeypatch):
    def unsupported(*_args, **_kwargs):
        raise OSError(22, "GPFS RENAME_NOREPLACE unsupported")

    monkeypatch.setattr(common, "_rename_no_replace", unsupported)
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    native = receipt.take_native_entry()
    receipt.quarantine_entry_at(native, phase="native replacement")
    entry = receipt.create_unique_at(suffix=".runtime-instance.tmp")
    os.write(entry.fileno(), b"resealed checkpoint")
    old_name = entry.name
    receipt.rename_no_replace_at(entry, "native.npz")
    assert entry.name == "native.npz"
    assert not (receipt.directory / old_name).exists()
    assert (receipt.directory / entry.name).read_bytes() == b"resealed checkpoint"
    assert os.fstat(entry.fileno()).st_nlink == 1
    receipt.quarantine_entry_at(entry, phase="completed staging")
    receipt.cleanup_empty()
    (container,) = tuple(tmp_path.iterdir())
    assert container.name.startswith(".pops-restart-cleanup-")
    assert container.is_dir() and not tuple(container.iterdir())
    assert stat.S_IMODE(container.stat().st_mode) == 0o700


@pytest.mark.parametrize("name", ("../escape.npz", "/absolute.npz", ".", "..", "", "bad\x00name"))
def test_private_entry_operations_reject_namespace_escape_before_mutation(tmp_path, name):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    native = receipt.take_native_entry()
    for operation in (
        lambda: receipt.created_at(name),
        lambda: receipt.open_candidate_at(name),
        lambda: receipt.rename_no_replace_at(native, name),
    ):
        with pytest.raises(ValueError, match="one local name"):
            operation()
    assert os.listdir(receipt.directory_fileno()) == ["native.npz"]
    assert _inode(receipt.staging_path) == native.owner
    receipt.quarantine_entry_at(native, phase="invalid name cleanup")
    receipt.cleanup_empty()


@pytest.mark.parametrize("same_inode", (False, True))
def test_private_reseal_collision_never_grants_destination_ownership(
    tmp_path, same_inode,
):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    native = receipt.take_native_entry()
    receipt.quarantine_entry_at(native, phase="native replacement")
    entry = receipt.create_unique_at(suffix=".runtime-instance.tmp")
    os.write(entry.fileno(), b"owned temporary")
    source = receipt.directory / entry.name
    target = receipt.directory / "native.npz"
    if same_inode:
        os.link(source, target)
    else:
        target.write_bytes(b"foreign collision")
    owner = _inode(target)
    content = target.read_bytes()
    with pytest.raises(FileExistsError):
        receipt.rename_no_replace_at(entry, "native.npz")
    assert source.is_file()
    assert _inode(target) == owner
    assert target.read_bytes() == content
    assert entry.name == source.name
    receipt.quarantine_entry_at(entry, phase="failed reseal")
    with pytest.raises(RuntimeError, match="not empty"):
        receipt.cleanup_empty()
    assert _inode(target) == owner


def test_private_cleanup_retains_detected_replacement_and_descriptor(tmp_path):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    entry = receipt.take_native_entry()
    original = receipt.directory / entry.name
    saved = receipt.directory / "saved-original.npz"
    os.rename(original, saved)
    original.write_bytes(b"foreign replacement")
    replacement_owner = _inode(original)
    with pytest.raises(RuntimeError, match="private recovery namespace retained"):
        receipt.quarantine_entry_at(entry, phase="injected replacement", close_entry=False)
    assert entry.is_open
    assert _inode(saved) == entry.owner
    assert _inode(original) == replacement_owner
    assert original.read_bytes() == b"foreign replacement"
    entry.close()
    with pytest.raises(RuntimeError, match="not empty"):
        receipt.cleanup_empty()
    assert saved.exists() and original.exists()


def test_private_reseal_authenticates_inode_captured_by_link(tmp_path, monkeypatch):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    native = receipt.take_native_entry()
    receipt.quarantine_entry_at(native, phase="native replacement")
    entry = receipt.create_unique_at(suffix=".runtime-instance.tmp")
    source = receipt.directory / entry.name
    saved = receipt.directory / "saved-original.npz"
    target = receipt.directory / "native.npz"
    real_link = os.link

    def replace_before_link(source_name, destination_name, **kwargs):
        os.rename(source, saved)
        source.write_bytes(b"foreign replacement at link")
        return real_link(source_name, destination_name, **kwargs)

    monkeypatch.setattr(os, "link", replace_before_link)
    with pytest.raises(RuntimeError, match="entry was replaced"):
        receipt.rename_no_replace_at(entry, "native.npz")
    assert entry.name == "native.npz"
    assert _inode(saved) == entry.owner
    assert os.path.samefile(source, target)
    assert target.read_bytes() == b"foreign replacement at link"
    with pytest.raises(RuntimeError, match="refuses to delete replaced transaction entry"):
        receipt.quarantine_entry_at(entry, phase="failed reseal")
    with pytest.raises(RuntimeError, match="not empty"):
        receipt.cleanup_empty()
    assert saved.is_file() and target.is_file()


@pytest.mark.parametrize("change", ("uid", "mode"))
def test_private_namespace_authority_is_required_for_cleanup(tmp_path, monkeypatch, change):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    entry = receipt.take_native_entry()
    if change == "uid":
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
    else:
        os.chmod(receipt.directory, 0o755)
    with pytest.raises(RuntimeError, match="private recovery namespace retained"):
        receipt.quarantine_entry_at(entry, phase="changed namespace")
    assert (receipt.directory / "native.npz").is_file()
    receipt.close()


def test_directory_substitution_at_detachment_is_retained_without_unlink(tmp_path, monkeypatch):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    receipt.quarantine_entry_at(receipt.take_native_entry(), phase="empty staging")
    original = receipt.directory
    saved = tmp_path / "saved-owned-directory"
    real_rename = os.rename

    def replace_directory_before_detachment(source, destination, **kwargs):
        if source == receipt.directory_name:
            real_rename(original, saved)
            original.mkdir()
            (original / "foreign.data").write_bytes(b"foreign directory content")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(os, "rename", replace_directory_before_detachment)
    with pytest.raises(RuntimeError, match="replacement retained at"):
        receipt.cleanup_empty()
    assert _inode(saved) == receipt.owner
    (retained,) = tuple(tmp_path.glob(".pops-restart-cleanup-*/transaction/foreign.data"))
    assert retained.read_bytes() == b"foreign directory content"


def test_cleanup_namespace_collision_is_retained_before_detachment(tmp_path, monkeypatch):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    receipt.quarantine_entry_at(receipt.take_native_entry(), phase="empty staging")
    original = receipt.directory
    real_open = os.open

    def occupy_private_destination_before_authentication(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".pops-restart-cleanup-"):
            foreign = real_open(
                "transaction", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(foreign, b"foreign destination collision")
            finally:
                os.close(foreign)
        return descriptor

    monkeypatch.setattr(os, "open", occupy_private_destination_before_authentication)
    with pytest.raises(RuntimeError, match="cleanup namespace authority changed"):
        receipt.cleanup_empty()
    assert original.is_dir()
    assert _inode(original) == receipt.owner
    (retained,) = tuple(tmp_path.glob(".pops-restart-cleanup-*/transaction"))
    assert retained.read_bytes() == b"foreign destination collision"


@pytest.mark.parametrize("directory_collision", (False, True))
def test_directory_detachment_never_clobbers_destination_content(
    tmp_path, monkeypatch, directory_collision,
):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    receipt.quarantine_entry_at(receipt.take_native_entry(), phase="empty staging")
    original = receipt.directory
    real_rename = os.rename

    def occupy_destination_before_detachment(source, destination, **kwargs):
        destination_fd = kwargs["dst_dir_fd"]
        if directory_collision:
            os.mkdir(destination, 0o700, dir_fd=destination_fd)
            content_fd = os.open(destination, receipt._directory_flags(), dir_fd=destination_fd)
            content_name = "foreign.data"
        else:
            content_fd = os.dup(destination_fd)
            content_name = destination
        try:
            descriptor = os.open(
                content_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                dir_fd=content_fd,
            )
            try:
                os.write(descriptor, b"concurrent destination collision")
            finally:
                os.close(descriptor)
        finally:
            os.close(content_fd)
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(os, "rename", occupy_destination_before_detachment)
    with pytest.raises(OSError):
        receipt.cleanup_empty()
    assert _inode(original) == receipt.owner
    suffix = "transaction/foreign.data" if directory_collision else "transaction"
    (retained,) = tuple(tmp_path.glob(".pops-restart-cleanup-*/" + suffix))
    assert retained.read_bytes() == b"concurrent destination collision"


def test_completed_cleanup_never_removes_a_substituted_public_container(tmp_path, monkeypatch):
    receipt = _CheckpointTransactionReceipt.created(tmp_path)
    receipt.quarantine_entry_at(receipt.take_native_entry(), phase="empty staging")
    real_rename, real_close, real_rmdir = os.rename, os.close, os.rmdir
    evidence = {}
    saved = tmp_path / "saved-empty-owned-container"

    def record_detachment(source, destination, **kwargs):
        evidence["cleanup_fd"] = kwargs["dst_dir_fd"]
        return real_rename(source, destination, **kwargs)

    def substitute_container_before_descriptor_close(descriptor):
        if descriptor == evidence.get("cleanup_fd") and "foreign" not in evidence:
            (container,) = tuple(tmp_path.glob(".pops-restart-cleanup-*"))
            real_rename(container, saved)
            container.mkdir(mode=0o700)
            foreign = container / "foreign.data"
            foreign.write_bytes(b"foreign public container replacement")
            evidence["foreign"] = foreign
        return real_close(descriptor)

    def forbid_public_rmdir(path, **kwargs):
        assert path == "transaction", "cleanup tried to remove a public directory name"
        return real_rmdir(path, **kwargs)

    monkeypatch.setattr(os, "rename", record_detachment)
    monkeypatch.setattr(os, "close", substitute_container_before_descriptor_close)
    monkeypatch.setattr(os, "rmdir", forbid_public_rmdir)
    receipt.cleanup_empty()
    assert saved.is_dir() and not tuple(saved.iterdir())
    assert evidence["foreign"].read_bytes() == b"foreign public container replacement"


def test_public_checkpoint_cleanup_still_requires_strict_atomic_quarantine(tmp_path, monkeypatch):
    target = tmp_path / "published.npz"
    target.write_bytes(b"public checkpoint")
    owner = _inode(target)

    def unsupported(*_args, **_kwargs):
        raise OSError(22, "GPFS RENAME_NOREPLACE unsupported")

    monkeypatch.setattr(common, "_rename_no_replace", unsupported)
    with pytest.raises(OSError, match="unsupported"):
        common._StagedOutputFile._quarantine_owned_path(
            target, owner, replaced_message="public rollback refused",
        )
    assert _inode(target) == owner
    assert target.read_bytes() == b"public checkpoint"
    assert tuple(tmp_path.iterdir()) == (target,)
