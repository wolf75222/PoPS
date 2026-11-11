#!/usr/bin/env python3
"""SU-T1 (spec 0.3, 2.6.7; SD79): the lint of the user surface.

The lint reads the syntax tree of a case or of a module of ``pops.lib`` and imports nothing. It
fails (``F``) on what SU-T1 names:

* ``lambda``: a lambda (SU5: a function of a case is a named ``def``);
* ``string-key``: a string key in a dictionary (SU8: a dictionary that PoPS receives is keyed by
  handles);
* ``word``: a word that the tables 2.6.2 to 2.6.4, 2.6.8 and 2.6.9 do not list: a name imported
  from a module of ``pops``, an attribute of ``pops``, a read of an entity that PoPS supplies
  (``cell.width``, ``step.state``), or a name that a module star-imports from ``pops.math`` and
  does not bind;
* ``star-import``: a star import of another module than ``pops.math``;
* ``math-rebinding``: a module-level rebinding of a word of ``pops.math``, or a parameter named
  like one;
* ``float``: arithmetic that Python computes on two numbers, each written as a literal or as a
  module-level name bound to one, whose result is a float (SU4, SD79);
* ``pow``: a call of Python's ``pow`` (SD79: ``**`` is the one spelling);
* ``membership``: ``in`` or ``.index`` on a list or a tuple of handles (SD79: they compare by
  ``==``, which builds a relation).

It warns (``W``) on an order of blocks other than that of SU6 (``order``), on ``vector(e)`` of one
component (``vector``, 2.6.3) and on ``lag`` of the state (``lag``, 2.6.8, SD55). An operator
outside SU4 applied to a PoPS value is refused by the trace at the user's line, and Python's
operators on Python data stay Python's (E4), so the lint does not look at operators.

The words are those of ``surface_lint_words.json``, extracted from the spec by
``--words SPEC``: the identifiers written in code spans in the sections 2.6.2 to 2.6.4, 2.6.8 and
2.6.9, outside the columns that give the spellings of earlier texts (``Replaces``), the elaborated
forms and the rules, and outside the code blocks. Until the namespace ``pops.surface`` becomes
``pops`` (§21.7), ``pops.surface`` reads as ``pops`` and ``pops.surface.math`` as ``pops.math``.

Usage::

    python scripts/surface_lint.py FILE...      # prints the findings; exit status 1 on a failure
    python scripts/surface_lint.py --words SPEC  # regenerates surface_lint_words.json from SPEC
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORDS = HERE / "surface_lint_words.json"
SECTIONS = ("2.6.2", "2.6.3", "2.6.4", "2.6.8", "2.6.9")
SKIPPED_COLUMNS = ("Replaces", "Elaborated form (spec 0.3)", "Rules")
TRANSITIONAL = {"pops.surface": "pops", "pops.surface.math": "pops.math"}
# The names that the column Words of 2.6.3 mentions without listing them as words of pops.math:
# two refusals, a form of the boundary policy (2.6.2) and a scheme of the library (2.6.9).
MENTIONED = ("AmbiguousElaboration", "NormalComponent", "SupportMismatch", "upwind")

# SU6: geometry, model, numerics, claims, methods, program, outputs, case and run.
LEVELS = ("geometry", "model", "numerics", "claims", "methods", "program", "outputs",
          "case and run")
LEVEL_OF = {"Box": 0, "product": 0, "Model": 1, "quantity": 1, "constant": 1, "parameter": 1,
            "data": 1, "define": 1, "flux": 1, "diffusive_flux": 1, "source": 1, "balance": 1,
            "interaction": 1, "inventory": 1, "constraint": 1, "relation": 1, "on": 1,
            "wave_speeds": 1, "Numerics": 2, "claim": 3, "method": 4, "multistep": 4, "assess": 4,
            "Advance": 5, "Program": 5, "Regrid": 5, "Output": 6, "Case": 7, "compile": 7,
            "bind": 7, "run": 7}
MODEL_WORDS = ("quantity", "constant", "parameter", "data", "define", "flux", "diffusive_flux",
               "source", "balance", "interaction", "inventory", "constraint", "relation", "on",
               "wave_speeds", "run")
# The entities of 2.6.4 whose reads are words; an axis, a depth, an interval and a payload are
# read as values.
READ_ENTITIES = ("cell", "face", "edge", "incident", "term", "step", "candidate", "region",
                 "problem")


class Finding:
    __slots__ = ("path", "line", "kind", "rule", "text")

    def __init__(self, path: str, line: int, kind: str, rule: str, text: str) -> None:
        self.path, self.line, self.kind, self.rule, self.text = path, line, kind, rule, text

    def key(self) -> tuple:
        return (self.path, self.line, self.kind, self.rule, self.text)

    def __str__(self) -> str:
        return "%s:%d: %s %s: %s" % (self.path, self.line, self.kind, self.rule, self.text)


# --- the words of the spec ---------------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_CODE = re.compile(r"`([^`]+)`")
_CELL = re.compile(r"(?<!\\)\|")


def extract_words(spec: Path) -> dict:
    """The words of the sections of SU-T1 in ``spec``, with the words of ``pops.math`` (2.6.3)
    and the entities (2.6.4) apart."""
    lines = spec.read_text(encoding="utf-8").splitlines()
    words: set = set()
    math: set = set()
    entities: list = []
    ranges = []
    section = None
    fenced = False
    header: list | None = None
    for number, line in enumerate(lines, start=1):
        heading = re.match(r"^(#{2,4}) (\S+)", line)
        if heading:
            section = heading.group(2) if heading.group(2) in SECTIONS else None
            if section is not None:
                ranges.append([section, number, number])
            header = None
            continue
        if section is None:
            continue
        ranges[-1][2] = number
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if not line.startswith("|"):
            header = None
            words.update(_identifiers(line))
            continue
        cells = [cell.strip() for cell in _CELL.split(line.strip().strip("|"))]
        if header is None:
            header = cells
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        for column, cell in zip(header, cells, strict=False):
            if column in SKIPPED_COLUMNS:
                continue
            found = _identifiers(cell)
            words.update(found)
            if section == "2.6.3" and column == "Words":
                math.update(_heads(cell))
            if section == "2.6.4" and column == "Entity":
                entities.extend(name for name in found if name not in entities)
    # The words of pops.math are the bare words and the heads of the calls that its column
    # writes, less the names that the column mentions without listing them (MENTIONED).
    math -= set(MENTIONED)
    return {"source": {"spec": spec.name,
                       "sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                       "sections": [{"section": s, "lines": [lo, hi]} for s, lo, hi in ranges],
                       "skipped_columns": list(SKIPPED_COLUMNS)},
            "math": sorted(math), "entities": entities, "words": sorted(words)}


def _identifiers(text: str) -> set:
    found: set = set()
    for span in _CODE.findall(text):
        found.update(_IDENTIFIER.findall(span))
    return found


def _heads(text: str) -> set:
    """The bare words of the code spans of ``text`` and the heads of the calls that they write."""
    found: set = set()
    for span in _CODE.findall(text):
        span = span.strip()
        if re.fullmatch(r"%s(\s+%s)*" % (_IDENTIFIER.pattern, _IDENTIFIER.pattern), span):
            found.update(span.split())
        else:
            call = re.match(r"(%s)\(" % _IDENTIFIER.pattern, span)
            if call:
                found.add(call.group(1))
    return found


def load_words(path: Path = WORDS) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"words": frozenset(data["words"]), "math": frozenset(data["math"]),
            "entities": tuple(data["entities"]), "source": data["source"]}


# --- the lint ---------------------------------------------------------------------------------

def _module(name: str | None) -> str:
    name = name or ""
    for old, new in sorted(TRANSITIONAL.items(), key=lambda item: -len(item[0])):
        if name == old or name.startswith(old + "."):
            return new + name[len(old):]
    return name


def _number(node: ast.AST, numbers: dict) -> object:
    """The Python value of a numeric expression of literals and module-level numbers, or None."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        return numbers.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(node.operand, numbers)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _number(node.left, numbers), _number(node.right, numbers)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow) and abs(right) <= 64 and abs(left) <= 2 ** 64:
                return left ** right
        except (ZeroDivisionError, OverflowError):
            return None
    return None


def _handles(node: ast.AST, sequences: dict) -> bool:
    """Whether ``node`` is a list or a tuple display of handles, or a module-level name of one."""
    if isinstance(node, ast.Name) and node.id in sequences:
        node = sequences[node.id]
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    return any(not isinstance(element, ast.Constant) for element in node.elts)


def _bound_names(tree: ast.AST) -> set:
    bound: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):
            if node.name != "*":
                bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _level(statement: ast.stmt, pops_names: set) -> int | None:
    levels = []
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for decorator in statement.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr in ("method", "multistep",
                                                                       "assess"):
                levels.append(LEVEL_OF[target.attr])
    else:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in LEVEL_OF:
                owner = node.func.value
                if (isinstance(owner, ast.Name) and owner.id in pops_names) \
                        or node.func.attr in MODEL_WORDS:
                    levels.append(LEVEL_OF[node.func.attr])
    return max(levels) if levels else None


def lint_source(source: str, path: str, words: dict | None = None) -> list:
    """The findings of SU-T1 on ``source``, sorted by line."""
    words = words or load_words()
    listed, math = words["words"], words["math"]
    tree = ast.parse(source, filename=path)
    found: list = []

    def report(node: ast.AST | int, kind: str, rule: str, text: str) -> None:
        line = node if isinstance(node, int) else getattr(node, "lineno", 0)
        found.append(Finding(path, line, kind, rule, text))

    pops_names: set = set()
    star_math = False
    numbers: dict = {}
    sequences: dict = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if _module(alias.name) == "pops":
                    pops_names.add(alias.asname or "pops")
                elif alias.asname is None and alias.name.split(".")[0] == "pops":
                    pops_names.add("pops")
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            value = _number(statement.value, numbers)
            if value is not None:
                numbers[name] = value
            if isinstance(statement.value, (ast.List, ast.Tuple)):
                sequences[name] = statement.value
        if isinstance(statement, ast.ImportFrom) and _module(statement.module) == "pops.math" \
                and any(alias.name == "*" for alias in statement.names):
            star_math = True

    # SU5, SU8, SD79: what each node shows.
    reported_floats: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            report(node, "F", "lambda", "a lambda; a function of a case is a named def (SU5)")
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    report(key, "F", "string-key", "the string key %r; a dictionary that PoPS "
                           "receives is keyed by handles (SU8)" % key.value)
        if isinstance(node, ast.ImportFrom):
            module = _module(node.module)
            for alias in node.names:
                if alias.name == "*":
                    if module != "pops.math":
                        report(node, "F", "star-import", "a star import of %s; only pops.math "
                               "is star-imported" % node.module)
                elif module.split(".")[0] == "pops" and alias.name not in listed:
                    report(node, "F", "word", "the word %s of %s, which 2.6.2 to 2.6.4, 2.6.8 "
                           "and 2.6.9 do not list" % (alias.name, node.module))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in pops_names and node.attr not in listed:
            report(node, "F", "word", "the word pops.%s, which 2.6.2 to 2.6.4, 2.6.8 and 2.6.9 "
                   "do not list" % node.attr)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            for argument in arguments:
                if argument.arg in math:
                    report(argument, "F", "math-rebinding", "a parameter named like %s, a word "
                           "of pops.math (SU13)" % argument.arg)
            entities = {argument.arg for argument in arguments} & set(READ_ENTITIES)
            if entities and not isinstance(node, ast.Lambda):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) \
                            and inner.value.id in entities and inner.attr not in listed:
                        report(inner, "F", "word", "the read %s.%s, which 2.6.2 to 2.6.4, 2.6.8 "
                               "and 2.6.9 do not list" % (inner.value.id, inner.attr))
        if isinstance(node, ast.BinOp) and id(node) not in reported_floats:
            value = _number(node, numbers)
            if isinstance(value, float):
                report(node, "F", "float", "%s is computed by Python as the float %r; write "
                       "the number exactly, with Fraction or integers (SU4, SD79)"
                       % (ast.unparse(node), value))
                reported_floats.update(id(inner) for inner in ast.walk(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "pow":
                report(node, "F", "pow", "a call of pow; ** is the one spelling (SD79)")
            if node.func.id == "vector" and len(node.args) == 1 and not node.keywords:
                report(node, "W", "vector", "vector(e) of one component; on a support of rank "
                       "one a vector is written as its scalar (2.6.3)")
            if node.func.id == "lag" and node.args and ast.unparse(node.args[0]).endswith(
                    ".state"):
                report(node, "W", "lag", "lag of the state; a past state is "
                       "step.previous(k).state (2.6.8, SD55)")
        if isinstance(node, ast.Compare):
            for operator, right in zip(node.ops, node.comparators, strict=True):
                if isinstance(operator, (ast.In, ast.NotIn)) and _handles(right, sequences):
                    report(node, "F", "membership", "in on a list or a tuple of handles compares "
                           "by ==, which builds a relation; use a set or a dictionary (SD79)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "index" and _handles(node.func.value, sequences):
            report(node, "F", "membership", ".index on a list or a tuple of handles compares by "
                   "==, which builds a relation (SD79)")

    # SU13: a word of pops.math is not rebound at module level.
    for statement in tree.body:
        names = []
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [statement.name]
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [
                statement.target]
            names = [node.id for target in targets for node in ast.walk(target)
                     if isinstance(node, ast.Name)]
        for name in names:
            if name in math:
                report(statement, "F", "math-rebinding", "a module-level rebinding of %s, a word "
                       "of pops.math (SU13)" % name)

    # A name of the star import that pops.math does not list and the module does not bind.
    if star_math:
        bound = _bound_names(tree) | set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                    and node.id not in bound and node.id not in math:
                report(node, "F", "word", "the name %s, which pops.math (2.6.3) does not list "
                       "and the module does not bind" % node.id)

    # SU6: the order of the blocks (a warning).
    top, top_line = -1, 0
    for statement in tree.body:
        level = _level(statement, pops_names)
        if level is None:
            continue
        if level < top and level != 3:   # a claim may follow the handles that it names
            report(statement, "W", "order", "a %s statement after the %s statement of line %d "
                   "(SU6)" % (LEVELS[level], LEVELS[top], top_line))
        elif level > top:
            top, top_line = level, statement.lineno

    unique = {finding.key(): finding for finding in found}
    return sorted(unique.values(), key=lambda finding: (finding.line, finding.rule, finding.text))


def lint_file(path: Path, words: dict | None = None, *, name: str | None = None) -> list:
    return lint_source(path.read_text(encoding="utf-8"), name or str(path), words)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--words", type=Path, metavar="SPEC",
                        help="regenerate %s from the spec" % WORDS.name)
    arguments = parser.parse_args(argv)
    if arguments.words is not None:
        data = extract_words(arguments.words)
        WORDS.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("%s: %d words, %d of pops.math, %d entities" % (
            WORDS, len(data["words"]), len(data["math"]), len(data["entities"])))
        return 0
    words = load_words()
    failures = 0
    for path in arguments.files:
        for finding in lint_file(path, words):
            print(finding)
            failures += finding.kind == "F"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
