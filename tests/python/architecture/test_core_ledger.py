"""The core ledger of spec 0.3 (FC7, CF15): the shape of every entry, until Q01 produces it (CF16).

An entry names its issue and the rule that admits it, what it adds, two scientifically distinct
structures with the builder of each case, the measurement that shows each one needs it, its
refusal set and the core paths it changes. A builder or a path that no longer exists fails here.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "tests/python/architecture/frozen_engine/core_ledger.toml"
KEYS = {"id", "issue", "rule", "pre_freeze", "what", "structures", "measurement", "refusal_set",
        "paths"}


def test_every_core_ledger_entry_names_its_issue_structures_measurement_and_paths():
    ledger = tomllib.loads(LEDGER.read_text(encoding="utf-8"))
    assert set(ledger) == {"mechanism"}
    entries = ledger["mechanism"]
    assert entries and len({entry["id"] for entry in entries}) == len(entries)
    builders = "\n".join(path.read_text(encoding="utf-8")
                         for path in sorted((ROOT / "examples/api_v02").glob("models_*.py")))
    for entry in entries:
        name = entry.get("id")
        assert set(entry) == KEYS, name
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), name
        assert re.fullmatch(r"ROM-[0-9]+", entry["issue"]), name
        assert re.fullmatch(r"[A-Z]+[0-9]+[a-z]?", entry["rule"]), name
        assert type(entry["pre_freeze"]) is bool, name
        assert all(type(entry[key]) is str and entry[key].strip()
                   for key in ("what", "measurement", "refusal_set")), name
        # CF15: two scientifically distinct structures, never one case.
        structures = entry["structures"]
        assert len({row["case"] for row in structures}) == len(structures) >= 2, name
        for row in structures:
            assert set(row) == {"case", "builder", "structure"}, (name, row)
            assert re.search(r"^def %s\(" % re.escape(row["builder"]), builders, re.M), row
        paths = entry["paths"]
        assert paths and len(set(paths)) == len(paths), name
        assert all((ROOT / path).is_file() for path in paths), (name, paths)
