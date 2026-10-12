"""Strict fixture oracle for portable checkpoint namespace cleanup."""
import os
import re
import stat


def require_no_checkpoint_staging(root):
    """Authenticate only the empty private tombstones required by portable cleanup.

    Production retains these names to avoid deleting a substituted public directory.
    No transaction entry, checkpoint data, symlink or unowned namespace may survive.
    """
    tombstones = []
    for path in sorted(root.rglob("*.pops-*")):
        assert re.fullmatch(r"\.pops-restart-cleanup-[0-9a-f]{32}", path.name), path
        named = path.lstat()
        assert stat.S_ISDIR(named.st_mode), path
        assert stat.S_IMODE(named.st_mode) == 0o700 and named.st_uid == os.geteuid(), path
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            retained = os.fstat(descriptor)
            owner = (named.st_dev, named.st_ino)
            assert (retained.st_dev, retained.st_ino) == owner, path
            assert retained.st_uid == os.geteuid() and stat.S_IMODE(retained.st_mode) == 0o700, path
            assert retained.st_dev == path.parent.stat().st_dev, path
            assert not os.listdir(descriptor), "checkpoint staging entries may not survive: %s" % path
            current = path.lstat()
            assert stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == owner, path
            assert current.st_uid == retained.st_uid and stat.S_IMODE(current.st_mode) == 0o700, path
            tombstones.append(dict(path=str(path.relative_to(root)), owner=list(owner),
                                   uid=retained.st_uid, mode="0700", entries=[]))
        finally:
            os.close(descriptor)
    return tombstones
