#!/usr/bin/env python3
"""Afficher ou enregistrer le domaine utilise par les tutoriels d'advection."""
from pathlib import Path

from pops.domain import Rectangle


HERE = Path(__file__).resolve().parent
PREVIEW_FILE = HERE / "results" / "00_domain_preview.svg"

domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")

domain.show(path=PREVIEW_FILE)

print("Domain preview: %s" % PREVIEW_FILE)
