#!/usr/bin/env python3
"""Deterministic lint and freshness gate for the maintained PoPS documentation.

The check protects the active corpus and its executable/source dependencies:

  - retained active docs are listed in docs/docmap.toml;
  - docmap depends_on / tested_by paths must exist;
  - relative Markdown links and image paths must resolve;
  - em-dashes are rejected in active project docs.

Usage: python docs/check_docs.py [--freshness-warn-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None


ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCMAP = DOCS / "docmap.toml"
EM_DASH = "\u2014"

PROJECT_ROOT_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CHANGELOG.md",
]

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)[\"']")


def load_docmap(path: pathlib.Path = DOCMAP) -> dict:
    if tomllib is None:
        raise RuntimeError("Python >= 3.11 is required for docs/check_docs.py")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def md_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    files = [root / p.name for p in PROJECT_ROOT_DOCS if (root / p.name).exists()]
    for directory in ("docs", "examples", "benchmarks"):
        files.extend(sorted((root / directory).glob("**/*.md")))
    return sorted(set(files))


def active_docs(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    files = [root / p.name for p in PROJECT_ROOT_DOCS if (root / p.name).exists()]
    files.extend(sorted((root / "docs").glob("*.md")))
    return sorted(set(files))


def retained_evidence(data: dict, root: pathlib.Path, violations: list[str]) -> set[pathlib.Path]:
    """Authenticate opt-in raw archives, never reinterpret their captured prose.

    Exact catalog pins, every member's bytes/hash and complete bounded inventory
    are required. Active/mapped documentation cannot become an archived member.
    """
    retained: set[pathlib.Path] = set()
    active = set(active_docs(root)) | {root / name for name in data.get("docs", {})}

    def bounded(base: pathlib.Path, name: str) -> pathlib.Path:
        if type(name) is not str:
            raise ValueError("chemin d'archive non textuel")
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or str(relative) != name or ".." in relative.parts:
            raise ValueError("chemin d'archive non canonique ou hors perimetre")
        target = base / name
        if target.resolve() != target.absolute() or not target.resolve().is_relative_to(base.resolve()):
            raise ValueError("alias ou chemin d'archive hors perimetre")
        return target

    def digest(path: pathlib.Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    archives = data.get("retained_evidence", {})
    if not isinstance(archives, dict):
        violations.append("retained_evidence: declaration d'archives non conforme")
        return retained
    for scope, config in archives.items():
        try:
            archive = bounded(root, scope)
            if not scope.startswith("docs/") or "evidence" not in pathlib.PurePosixPath(scope).parts:
                raise ValueError("archive hors du perimetre documentaire evidence")
            if not archive.is_dir() or set(config) != {"members", "catalogs", "metadata"}:
                raise ValueError("declaration d'archive incomplete")
            if type(config["members"]) is not int or config["members"] <= 0:
                raise ValueError("nombre de membres d'archive invalide")
            catalogs, metadata = config["catalogs"], config["metadata"]
            if not isinstance(catalogs, dict) or not catalogs or not isinstance(metadata, dict):
                raise ValueError("catalogs/metadata d'archive invalides")
            controls, members = set(), set()
            for name, expected in {**catalogs, **metadata}.items():
                target = bounded(archive, name)
                if name in catalogs and name in metadata:
                    raise ValueError("controle d'archive duplique")
                if target.suffix != ".json" or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                    raise ValueError("controle d'archive sans empreinte JSON exacte")
                if not target.is_file() or digest(target) != expected:
                    raise ValueError("controle d'archive absent ou altere: " + name)
                controls.add(target)
            for name in catalogs:
                document = json.loads(bounded(archive, name).read_text(encoding="utf-8"))
                if not isinstance(document, dict) or type(document.get("schema")) is not int or document["schema"] != 1 \
                        or not isinstance(document.get("files"), dict) or not document["files"]:
                    raise ValueError("catalog d'archive non conforme: " + name)
                for member, evidence in document["files"].items():
                    target = bounded(archive, member)
                    if target in active or target in controls or target in members:
                        raise ValueError("page active, controle ou membre duplique dans archive: " + member)
                    if set(evidence) != {"source", "bytes", "sha256"} \
                            or type(evidence["source"]) is not str or not evidence["source"] \
                            or type(evidence["bytes"]) is not int or evidence["bytes"] < 0 \
                            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence["sha256"])):
                        raise ValueError("preuve d'archive non conforme: " + member)
                    if not target.is_file() or target.stat().st_size != evidence["bytes"] \
                            or digest(target) != evidence["sha256"]:
                        raise ValueError("membre d'archive absent ou altere: " + member)
                    members.add(target)
            paths = list(archive.rglob("*"))
            if any(path.is_symlink() for path in paths):
                raise ValueError("alias interdit dans archive")
            actual = {path for path in paths if path.is_file()}
            if len(members) != config["members"] or actual != members | controls:
                raise ValueError("inventaire d'archive incomplet ou membre non catalogue")
            retained.update(members)
        except (OSError, ValueError, KeyError, TypeError) as error:
            violations.append(f"{scope}: {error}")
    return retained


def relpath(path: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    return str(path.relative_to(root)).replace("\\", "/")



def mask_code(text: str) -> str:
    """Blank code blocks/spans while preserving offsets for line numbers."""

    def blank(match: re.Match) -> str:
        return "".join(c if c == "\n" else " " for c in match.group(0))

    text = re.sub(r"```.*?```", blank, text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", blank, text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", blank, text)


def local_target(raw: str) -> str | None:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        raw = raw[1:raw.index(">")]
    target = raw.split()[0] if raw.split() else ""
    path = target.split("#", 1)[0]
    if not path:
        return None
    if target.startswith(("http://", "https://", "mailto:", "data:", "#")):
        return None
    first = path.split("/", 1)[0]
    looks_like_host = "/" in path and "." in first and first.rsplit(".", 1)[-1].isalpha()
    if "://" in target or first.startswith("www.") or looks_like_host:
        return None
    return path


def check_links(
    path: pathlib.Path, text: str, violations: list[str], root: pathlib.Path = ROOT,
) -> None:
    masked = mask_code(text)
    rel = relpath(path, root)

    for regex in (LINK_RE, IMAGE_RE):
        for match in regex.finditer(masked):
            raw = next((group for group in match.groups() if group), "")
            target = local_target(raw)
            if target is None:
                continue
            if not (path.parent / target).resolve().exists():
                line = masked[: match.start()].count("\n") + 1
                kind = "image" if regex is IMAGE_RE else "lien"
                violations.append(f"{rel}:{line}: {kind} relatif introuvable : {target}")


def _git(args: list[str], root: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def last_commit(doc_rel: str, root: pathlib.Path) -> str | None:
    return _git(["log", "-1", "--format=%H", "--", doc_rel], root)


def commits_touching(ref: str, deps: list[str], root: pathlib.Path) -> list[str] | None:
    res = _git(["rev-list", "--no-merges", f"{ref}..HEAD", "--", *deps], root)
    if res is None:
        return None
    return [line for line in res.splitlines() if line]


def check_docmap(data: dict, root: pathlib.Path, violations: list[str]) -> None:
    docs_map = data.get("docs", {})
    mapped = set(docs_map)

    for path in active_docs(root):
        rel = relpath(path, root)
        if rel not in mapped:
            violations.append(f"{rel}: doc actif absent de docs/docmap.toml")

    for doc, meta in docs_map.items():
        doc_path = root / doc
        if not doc_path.exists():
            violations.append(f"{doc}: entree docmap sans fichier correspondant")
        for kind in ("depends_on", "tested_by"):
            for dep in meta.get(kind, []) or []:
                if not (root / dep).exists():
                    violations.append(f"{doc}: {kind} introuvable sur le disque : {dep}")


def check_freshness(
    data: dict,
    root: pathlib.Path,
    violations: list[str],
    warnings: list[str],
    warn_only: bool,
) -> None:
    for doc, meta in data.get("docs", {}).items():
        deps = meta.get("depends_on") or []
        if not deps:
            continue
        ref = meta.get("reviewed") or last_commit(doc, root)
        if not ref:
            warnings.append(f"{doc}: fraicheur ignoree, document jamais commite")
            continue
        bad = commits_touching(ref, deps, root)
        if bad is None:
            warnings.append(f"{doc}: fraicheur ignoree, reference git inconnue {ref[:12]}")
            continue
        if not bad:
            continue
        short = ", ".join(commit[:12] for commit in bad[:3])
        more = "" if len(bad) <= 3 else f" (+{len(bad) - 3} autre(s))"
        msg = f"{doc}: doc suspect, depends_on modifie depuis la relecture ({short}{more})"
        if warn_only or meta.get("mode") == "warning":
            warnings.append(msg)
        else:
            violations.append(msg)


def check(freshness_warn_only: bool = False, root: pathlib.Path = ROOT) -> int:
    violations: list[str] = []
    warnings: list[str] = []

    docmap = root / "docs" / "docmap.toml"
    if not docmap.exists():
        violations.append("docs/docmap.toml manquant")
        data: dict = {}
    else:
        data = load_docmap(docmap)

    archived = retained_evidence(data, root, violations)

    for path in md_files(root):
        if path in archived:
            continue  # Its exact retained bytes are checked above; it is not active prose.
        rel = relpath(path, root)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            violations.append(f"{rel}: document absent ou illisible: {error}")
            continue
        if EM_DASH in text:
            violations.append(f"{rel}: {text.count(EM_DASH)} em-dash (U+2014) interdits")
        check_links(path, text, violations, root)

    if data:
        check_docmap(data, root, violations)
        check_freshness(data, root, violations, warnings, freshness_warn_only)

    if warnings:
        print(f"DOC-LINT : {len(warnings)} avertissement(s)", file=sys.stderr)
        for warning in warnings:
            print("  " + warning, file=sys.stderr)

    if violations:
        print(f"DOC-LINT : {len(violations)} violation(s)", file=sys.stderr)
        for violation in violations:
            print("  " + violation, file=sys.stderr)
        return 1

    docs_map = data.get("docs", {}) if data else {}
    print(f"DOC-LINT : OK ({len(md_files(root))} fichiers .md verifies, {len(docs_map)} entrees docmap)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoPS documentation conformance lint.")
    parser.add_argument(
        "--freshness-warn-only",
        action="store_true",
        help="downgrade freshness violations to warnings",
    )
    args = parser.parse_args()
    sys.exit(check(freshness_warn_only=args.freshness_warn_only))
