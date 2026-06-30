#!/usr/bin/env python3
"""Transitional documentation lint for the adc_cpp documentation reset.

The old Sphinx/Doxygen site has been removed. This script only protects the
small retained corpus while the new documentation is rebuilt:

  - retained active docs are listed in docs/docmap.toml;
  - docmap depends_on / tested_by paths must exist;
  - relative Markdown links and image paths must resolve;
  - em-dashes are rejected in active project docs.

Usage: python docs/check_docs.py [--freshness-warn-only]
"""
from __future__ import annotations

import argparse
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
DOCGUIDE = DOCS / "docguide"
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
    files = [p for p in PROJECT_ROOT_DOCS if p.exists()]
    files.extend(sorted((root / "docs").glob("**/*.md")))
    return sorted(set(files))


def active_docs(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    files = [p for p in PROJECT_ROOT_DOCS if p.exists()]
    files.extend(sorted((root / "docs").glob("*.md")))
    return sorted(set(files))


def relpath(path: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def is_docguide(path: pathlib.Path) -> bool:
    try:
        path.relative_to(DOCGUIDE)
        return True
    except ValueError:
        return False


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


def check_links(path: pathlib.Path, text: str, violations: list[str]) -> None:
    masked = mask_code(text)
    rel = relpath(path)

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

    if not DOCMAP.exists():
        violations.append("docs/docmap.toml manquant")
        data: dict = {}
    else:
        data = load_docmap(DOCMAP)

    for path in md_files(root):
        text = path.read_text(encoding="utf-8")
        rel = relpath(path, root)
        if EM_DASH in text and not is_docguide(path):
            violations.append(f"{rel}: {text.count(EM_DASH)} em-dash (U+2014) interdits")
        check_links(path, text, violations)

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
    parser = argparse.ArgumentParser(description="Transitional adc_cpp documentation lint.")
    parser.add_argument(
        "--freshness-warn-only",
        action="store_true",
        help="downgrade freshness violations to warnings",
    )
    args = parser.parse_args()
    sys.exit(check(freshness_warn_only=args.freshness_warn_only))
