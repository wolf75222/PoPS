"""pops.external.manifests -- read + register a compiled-brick manifest (Spec 5 sec.5.17).

A manifest is the JSON ``pops_brick_manifest()`` exports under the strict v3 schema: every ABI and
brick compatibility field is explicit, and documentary ``annotations`` are carried losslessly.
It can be read from a ``.json`` file or from a ``.so`` (dlopened).
:func:`register` / :func:`register_manifest_file` register the ids in the in-process catalog owned by
:mod:`pops.descriptors`; :func:`read_manifest` is the read-only counterpart that returns the metadata
WITHOUT registering or executing anything. The strict parse (schema_version / required fields /
unknown-field refusal) lives ONCE in :func:`pops.descriptors.parse_brick_manifest`. Nothing here computes.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

from pops.descriptors import (
    BRICK_MANIFEST_SCHEMA_VERSION,
    _parse_brick_manifest_document,
    _register_manifest,
    load_cpp_library,
    parse_brick_manifest,
)


def register_manifest_file(path: Any) -> Any:
    """Register the bricks in a manifest ``.json`` file. Returns the count registered."""
    with open(str(path), encoding="utf-8") as handle:
        return _register_manifest(handle.read())


def register(path: Any) -> Any:
    """Register a manifest from a ``.json`` file or a brick ``.so`` (dlopen). Returns the count.

    A ``.json`` path is parsed directly; anything else is treated as a loadable ``.so`` and
    dlopened via :func:`pops.descriptors.load_cpp_library` (its static initializers register
    the bricks and the exported ``pops_brick_manifest()`` is read).
    """
    p = str(path)
    if p.endswith(".json"):
        return register_manifest_file(p)
    return load_cpp_library(p)


def register_and_capture(path):
    """Register a manifest AND return ``(records, abi_key, handle)`` for the ADC-544 gates.

    Like :func:`register` but exposes the parsed per-brick records, the manifest ``abi_key`` (for the
    G1 ABI gate) and the loaded ``.so`` handle (the ctypes ``CDLL`` for the G4 dlsym probe). A ``.json``
    path is parsed directly and yields ``handle=None`` -- there is no ``.so`` to probe, so a
    ``CompiledBrickRef`` over a ``.json`` honestly SKIPS G4. A ``.so`` path is dlopened (its static
    initializers register the bricks) and the SAME ``CDLL`` handle is returned so the G4 probe reads
    THIS brick ``.so``'s own symbols (never a process-global lookup; ADC-622 STB_GNU_UNIQUE caveat).
    The bricks are also registered in the in-process catalog (parity with :func:`register`)."""
    p = str(path)
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as fh:
            manifest_json = fh.read()
        records, abi_key = parse_brick_manifest(manifest_json)
        _register_manifest(manifest_json)
        return records, abi_key, None
    os.stat(p)  # exact FileNotFoundError before ctypes normalizes the dynamic-loader failure
    handle = ctypes.CDLL(p)  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("brick library %r does not export pops_brick_manifest(); it is not an "
                         "pops brick .so" % (p,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("brick library %r: pops_brick_manifest() returned NULL" % (p,))
    manifest_json = raw.decode("utf-8")
    records, abi_key = parse_brick_manifest(manifest_json)
    _register_manifest(manifest_json)
    return records, abi_key, handle


class CompiledManifest:
    """The read-only metadata of a compiled-brick manifest (Spec 5 sec.5.17).

    A plain value holding the parsed manifest: the ABI key (when the manifest carries one) and
        annotations and the per-brick records. It is inert -- it
    NEITHER registers the bricks in the in-process catalog NOR dlopens / executes anything, so a
    caller can inspect a third-party brick before deciding to load it. Use
    :func:`pops.external.register` to actually register the bricks.
    """

    def __init__(self, bricks: Any, *, abi_key: Any, annotations: Any) -> None:
        from copy import deepcopy

        self.bricks = deepcopy(list(bricks))
        self.abi_key = abi_key
        self.annotations = deepcopy(dict(annotations))

    @property
    def ids(self) -> list:
        """The brick ids in declaration order."""
        return [b["id"] for b in self.bricks]

    @property
    def categories(self) -> list:
        """The set of brick categories the manifest declares."""
        return sorted({b["category"] for b in self.bricks})

    def to_dict(self) -> dict:
        """The exact canonical v3 wire form, reversible through :meth:`from_dict`."""
        from copy import deepcopy

        csv_fields = (
            "requirements", "capabilities", "supported_layouts", "supported_platforms", "params",
            "options", "exported_symbols",
        )
        bricks = []
        for record in self.bricks:
            row = {key: record[key] for key in ("id", "category", "native_id")}
            row.update({key: ",".join(record[key]) for key in csv_fields})
            bricks.append(row)
        return {
            "schema_version": BRICK_MANIFEST_SCHEMA_VERSION,
            "abi_key": self.abi_key,
            "annotations": deepcopy(self.annotations),
            "bricks": bricks,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CompiledManifest:
        """Strictly reconstruct the inert value from its canonical JSON-ready wire form."""
        import json

        if not isinstance(data, dict):
            raise TypeError("CompiledManifest.from_dict expects a dict")
        records, abi_key, annotations = _parse_brick_manifest_document(
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return cls(records, abi_key=abi_key, annotations=annotations)

    def __repr__(self) -> str:
        return "CompiledManifest(ids=%r, abi_key=%r)" % (self.ids, self.abi_key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CompiledManifest) and self.to_dict() == other.to_dict()


def _parse_manifest_metadata(manifest_json: Any) -> Any:
    """Parse manifest JSON into a :class:`CompiledManifest` WITHOUT registering it, under the STRICT
    versioned schema (ADC-611).

    Delegates to :func:`pops.descriptors.parse_brick_manifest` (the single strict parser: schema_version
    check, required fields, unknown-field refusal, each error naming the offending field), building an
    inert value instead of mutating the in-process catalog. Any schema violation raises ``ValueError``.
    """
    records, abi_key, annotations = _parse_brick_manifest_document(manifest_json)
    return CompiledManifest(records, abi_key=abi_key, annotations=annotations)


def read_manifest(path: Any) -> Any:
    """Read a brick manifest (``.json`` or ``.so``) for INSPECTION only (Spec 5 sec.5.17).

    Returns a :class:`CompiledManifest` with the manifest's ids / categories / requirements /
    capabilities (and the ``abi_key`` when present). It does NOT register the bricks in the
    in-process catalog and does NOT execute any brick code: a ``.json`` path is parsed directly;
    a ``.so`` path is dlopened ONLY to read the exported ``pops_brick_manifest()`` string (its
    static initializers run as a side effect of any dlopen, but no brick is registered or
    invoked). Use :func:`register` when you actually want to register the bricks.
    """
    p = str(path)
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as handle:
            return _parse_manifest_metadata(handle.read())
    os.stat(p)  # exact FileNotFoundError for missing shared-object manifests
    handle = ctypes.CDLL(p)  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("brick library %r does not export pops_brick_manifest(); it is not an "
                         "pops brick .so" % (p,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("brick library %r: pops_brick_manifest() returned NULL" % (p,))
    return _parse_manifest_metadata(raw.decode("utf-8"))


__all__ = ["register", "register_manifest_file", "register_and_capture", "read_manifest",
           "CompiledManifest", "BRICK_MANIFEST_SCHEMA_VERSION"]
