"""CF1 (spec 0.3, section 16.2; FC6b, FC8): adding a case changes no path of the frozen engine.

For every conformance case, probe and the final witness, ``git diff <base>..<case> -- <core paths>
<native component tree>`` is empty. In CI the case's files are added on top of the integration head
and the tree hashes of the core paths and of the native component tree are compared.

A tree hash is the id of the git tree that holds exactly the files of one part of the frozen engine
(``cf0_paths.toml``: the core paths, or the native component tree), at their repository paths; the
empty tree when the part has no file, as the native component tree at b6e64c1d. Two trees have the
same hash exactly when the part has the same files with the same contents and modes. FC8 records
these two hashes at a freeze.

Trees are built with a temporary index in a temporary object directory that borrows the
repository's objects, so a check writes nothing into the repository.

    python -m tests.python.architecture.frozen_engine.cf1 --all     # every case of cases.toml
    python -m tests.python.architecture.frozen_engine.cf1 --case M01
    python -m tests.python.architecture.frozen_engine.cf1 --ref <case commit> [--base <commit>]

``--ref`` takes the files that a case branch adds, changes or deletes since ``--base`` (by
default its merge base with ``--head``) and applies them on top of ``--head`` (default ``HEAD``).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

from . import cf8, scan


class ObjectStore:
    """A temporary git object directory and index, with the repository's objects as alternates."""

    def __init__(self, root: pathlib.Path = scan.REPO_ROOT):
        self.root = root
        self._tmp = tempfile.TemporaryDirectory(prefix="pops-cf1-")
        objects = pathlib.Path(self._tmp.name) / "objects"
        objects.mkdir()
        repository_objects = self.git("rev-parse", "--git-path", "objects",
                                      own=False).decode().strip()
        self.env = dict(os.environ,
                        GIT_OBJECT_DIRECTORY=str(objects),
                        GIT_ALTERNATE_OBJECT_DIRECTORIES=str((root / repository_objects).resolve()),
                        GIT_INDEX_FILE=str(pathlib.Path(self._tmp.name) / "index"))

    def __enter__(self) -> ObjectStore:
        return self

    def __exit__(self, *exc) -> None:
        self._tmp.cleanup()

    def git(self, *args: str, stdin: bytes | None = None, own: bool = True) -> bytes:
        completed = subprocess.run(["git", "-C", str(self.root), *args], input=stdin,
                                   capture_output=True, env=self.env if own else None,
                                   check=False)
        if completed.returncode:
            raise RuntimeError("git %s failed: %s" % (" ".join(args),
                                                      completed.stderr.decode(errors="replace")))
        return completed.stdout

    # -----------------------------------------------------------------------------------------
    def tree_of(self, rev: str) -> str:
        return self.git("rev-parse", "--verify", rev + "^{tree}").decode().strip()

    def entries(self, tree: str) -> dict[str, tuple[str, str]]:
        """path -> (mode, object id) of every file of a tree, submodules included."""
        out = self.git("ls-tree", "-r", "-z", "--full-tree", tree)
        found = {}
        for record in out.split(b"\0"):
            if record:
                meta, path = record.split(b"\t", 1)
                mode, _, oid = meta.decode().split(" ")
                found[path.decode()] = (mode, oid)
        return found

    def write_tree(self, entries: dict[str, tuple[str, str]]) -> str:
        self.git("read-tree", "--empty")
        if entries:
            self.git("update-index", "-z", "--index-info", stdin=b"".join(
                b"%s %s\t%s\0" % (mode.encode(), oid.encode(), path.encode())
                for path, (mode, oid) in sorted(entries.items())))
        return self.git("write-tree").decode().strip()

    def blob(self, content: bytes) -> str:
        return self.git("hash-object", "-w", "--stdin", stdin=content).decode().strip()

    def worktree_file(self, rel: str) -> tuple[str, str]:
        """(mode, blob id) of a file of the checkout as it is on disk."""
        path = self.root / rel
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            return "120000", self.blob(os.readlink(path).encode())
        mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
        oid = self.git("hash-object", "-w", "--path=" + rel, str(path)).decode().strip()
        return mode, oid

    def commit(self, tree: str, parent: str, message: str = "cf1 probe") -> str:
        """A commit object in the temporary store (no ref moves), for --ref checks of a probe."""
        env = dict(GIT_AUTHOR_NAME="cf1", GIT_AUTHOR_EMAIL="cf1@invalid",
                   GIT_COMMITTER_NAME="cf1", GIT_COMMITTER_EMAIL="cf1@invalid")
        saved = dict(self.env)
        self.env.update(env)
        try:
            return self.git("commit-tree", tree, "-p", parent, "-m", message).decode().strip()
        finally:
            self.env = saved


@dataclasses.dataclass(frozen=True)
class FrozenHashes:
    core: str
    native: str


def frozen_hashes(store: ObjectStore, entries: dict[str, tuple[str, str]],
                  engine: scan.FrozenEngine | None = None) -> FrozenHashes:
    """The tree hashes of the core paths and of the native component tree of a set of entries."""
    engine = engine or scan.load_cf0()
    core = {path: entry for path, entry in entries.items() if engine.is_core(path)}
    native = {path: entry for path, entry in entries.items() if engine.is_native(path)}
    return FrozenHashes(store.write_tree(core), store.write_tree(native))


@dataclasses.dataclass
class Check:
    """The outcome of adding a case on top of a head."""
    head: str
    head_hashes: FrozenHashes  # the integration head
    with_case: FrozenHashes  # the head with the case's files added on top
    without_case: FrozenHashes | None  # the head without the case's files, a tree that lacks it
    case_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]  # case paths that lie in the frozen engine (FC6b)

    @property
    def unchanged(self) -> bool:
        return (self.head_hashes == self.with_case and not self.frozen_paths
                and self.without_case in (None, self.with_case))

    def describe(self, label: str) -> str:
        lines = ["CF1 %s: %s" % (label, "frozen engine unchanged" if self.unchanged else
                                 "the case changes the frozen engine"),
                 "  head %s, %d case file(s)" % (self.head, len(self.case_paths)),
                 "  core tree   %s -> %s" % (self.head_hashes.core, self.with_case.core),
                 "  native tree %s -> %s" % (self.head_hashes.native, self.with_case.native)]
        if self.without_case is not None:
            lines.append("  without the case: core %s, native %s"
                         % (self.without_case.core, self.without_case.native))
        lines.extend("  frozen path in the case: %s" % path for path in self.frozen_paths)
        return "\n".join(lines)


def apply_case(store: ObjectStore, head: str, changes: dict[str, tuple[str, str] | None], *,
               lacking: bool, engine: scan.FrozenEngine | None = None) -> Check:
    """Add @p changes (path -> (mode, object id), or None to delete the path) on top of @p head and
    compare the frozen-engine hashes. With @p lacking, also build the tree that lacks the case (the
    head without the case's paths), from which adding the case must change no frozen path either."""
    engine = engine or scan.load_cf0()
    head_entries = store.entries(store.tree_of(head))
    with_case = {path: entry for path, entry in head_entries.items() if path not in changes}
    without = dict(with_case) if lacking else None
    with_case.update({path: entry for path, entry in changes.items() if entry is not None})
    touched = tuple(sorted(changes))
    return Check(head, frozen_hashes(store, head_entries, engine),
                 frozen_hashes(store, with_case, engine),
                 None if without is None else frozen_hashes(store, without, engine),
                 touched, tuple(path for path in touched if engine.is_frozen(path)))


def check_case(case: dict, *, head: str = "HEAD", store: ObjectStore | None = None) -> Check:
    """CF1 for one [[case]] of cases.toml, with the case's files as they are in the checkout."""
    own = store is None
    store = store or ObjectStore()
    try:
        return apply_case(store, head, {rel: store.worktree_file(rel) for rel in case["files"]},
                          lacking=True)
    finally:
        if own:
            store.__exit__()


def check_ref(case_rev: str, *, base: str | None = None, head: str = "HEAD",
              store: ObjectStore | None = None) -> Check:
    """CF1 for a case branch: the files that it adds, changes or deletes since @p base, applied on
    top of @p head."""
    own = store is None
    store = store or ObjectStore()
    try:
        base = base or store.git("merge-base", head, case_rev).decode().strip()
        diff = store.git("diff", "--name-status", "--no-renames", "-z", base, case_rev).split(b"\0")
        case_entries = store.entries(store.tree_of(case_rev))
        changes: dict[str, tuple[str, str] | None] = {}
        for status, path in zip(diff[0::2], diff[1::2], strict=False):  # a trailing NUL
            if status:
                name = path.decode()
                changes[name] = None if status.startswith(b"D") else case_entries[name]
        return apply_case(store, head, changes, lacking=False)
    finally:
        if own:
            store.__exit__()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every [[case]] of cases.toml")
    which.add_argument("--case", action="append", help="a case id of cases.toml")
    which.add_argument("--ref", help="a case commit, checked with the files it changes")
    parser.add_argument("--base", help="with --ref: the commit the case branch started from")
    parser.add_argument("--head", default="HEAD", help="the integration head (default HEAD)")
    args = parser.parse_args(argv)

    failed = False
    with ObjectStore() as store:
        if args.ref:
            check = check_ref(args.ref, base=args.base, head=args.head, store=store)
            print(check.describe(args.ref))
            failed = not check.unchanged
        else:
            cases = {case["id"]: case for case in cf8.load_cases()}
            unknown = sorted(set(args.case or ()) - set(cases))
            if unknown:
                parser.error("unknown case id(s): %s" % ", ".join(unknown))
            for case_id in (sorted(cases) if args.all else args.case):
                check = check_case(cases[case_id], head=args.head, store=store)
                print(check.describe(case_id))
                failed = failed or not check.unchanged
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
