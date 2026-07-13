"""Builtin brick inspection over the generated component catalog."""
from __future__ import annotations

from ._generated_component_routes import (
    BRICK_CATALOG_ROWS,
    COMPONENT_CATALOG_SCHEMA_VERSION,
    COMPONENT_CATALOG_SEMANTIC_SHA256,
    COMPONENT_CATALOG_SHA256,
)


def _row_dict(row) -> dict:
    return {
        "category": row["category"],
        "id": row["id"],
        "route_index": row["route_index"],
        "native_entry": row["native_entry"],
        "parameters": list(row["parameters"]),
        "n_vars": row["n_vars"],
        "polar_ok": row["polar_ok"],
        "requirements": list(row["requirements"]),
        "limitations": list(row["limitations"]),
        "summary": row["summary"],
        "catalog_digest": COMPONENT_CATALOG_SHA256,
        "catalog_semantic_digest": COMPONENT_CATALOG_SEMANTIC_SHA256,
    }


def brick_catalog() -> list[dict]:
    """Return detached JSON-ready rows in stable generated route order."""
    return [_row_dict(row) for row in BRICK_CATALOG_ROWS]


def catalog_info() -> dict:
    """Return the versioned identity of the declaration authority."""
    return {
        "schema_version": COMPONENT_CATALOG_SCHEMA_VERSION,
        "digest": COMPONENT_CATALOG_SHA256,
        "semantic_digest": COMPONENT_CATALOG_SEMANTIC_SHA256,
    }


def catalog_ids(category: str) -> list[str]:
    return [row["id"] for row in BRICK_CATALOG_ROWS if row["category"] == category]


def resolve(category: str, id: str, context: str = "brick catalog") -> dict:
    for row in BRICK_CATALOG_ROWS:
        if row["category"] == category and row["id"] == id:
            return _row_dict(row)
    ids = catalog_ids(category)
    if not ids:
        categories = list(dict.fromkeys(row["category"] for row in BRICK_CATALOG_ROWS))
        raise ValueError(
            "%s: unknown category %r (valid: %s)" % (context, category, "|".join(categories)))
    raise ValueError(
        "%s: unknown %s brick %r (catalog: %s); the catalog never falls back to a default"
        % (context, category, id, "|".join(ids)))


__all__ = ["brick_catalog", "catalog_ids", "catalog_info", "resolve"]
