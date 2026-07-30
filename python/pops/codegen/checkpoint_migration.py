"""Explicit offline checkpoint migration entry points.

This module is deliberately not imported by :mod:`pops.runtime` or by restart loaders.
"""

from ._checkpoint_migration_uniform_v2 import (
    UNIFORM_V2_AUTHORITY_TRANSFERS,
    UNIFORM_V2_MIGRATION_PROTOCOL,
    UNIFORM_V2_MIGRATION_SCHEMA_VERSION,
    UNIFORM_V2_SOURCE_VERSION,
    UniformV2BlockMapping,
    UniformV2HistoryMapping,
    UniformV2MigrationMapping,
    UniformV2MigrationReport,
    migrate_uniform_v2_checkpoint,
)

__all__ = [
    "UNIFORM_V2_AUTHORITY_TRANSFERS",
    "UNIFORM_V2_MIGRATION_PROTOCOL",
    "UNIFORM_V2_MIGRATION_SCHEMA_VERSION",
    "UNIFORM_V2_SOURCE_VERSION",
    "UniformV2BlockMapping",
    "UniformV2HistoryMapping",
    "UniformV2MigrationMapping",
    "UniformV2MigrationReport",
    "migrate_uniform_v2_checkpoint",
]
