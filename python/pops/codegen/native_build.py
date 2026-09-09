"""One authenticated native source staging path for model and Program compilers."""

from __future__ import annotations

import os
from pathlib import Path

from pops._ir.native_call import native_functions


def model_native_roots(model):
    """One formula inventory for native source discovery and artifact authentication."""
    return {
        name: getattr(model, name, None)
        for name in (
            "_flux",
            "_eig",
            "_wave_speeds",
            "_ws_jacobian",
            "prim_defs",
            "_source",
            "_source_terms",
            "_flux_terms",
            "_proj",
            "cons_from",
            "_linear_sources",
            "_local_transforms",
            "_stab_speed",
            "_stab_dt",
            "_src_freq",
            "_src_jac",
            "_roe_rows",
            "_roe_jacobian",
            "_riemann_hook_forms",
            "_elliptic",
            "_elliptic_fields",
        )
    }


def model_native_components(model):
    """Discover source providers from emitted physical formula roots, never from names."""
    components = {}
    for function in native_functions(model_native_roots(model)):
        components.setdefault(function.component.manifest_sha256, function.component)
    return tuple(components.values())


def stage_native_components(components, directory, pops_include_root):
    """Stage verified trees and reject ambiguous or SDK-shadowing header ownership."""
    flags, roots, authorities, header_owners = [], {}, [], {}
    for component in components:
        component.verify_builtin_headers(pops_include_root)
        for header in component.files:
            prior = header_owners.get(header.path)
            if prior is not None and prior != header.sha256:
                raise ValueError(
                    "prepared native components provide conflicting header %r" % header.path
                )
            if os.path.isfile(os.path.join(pops_include_root, header.path)):
                raise ValueError(
                    "prepared native component %r shadows PoPS SDK header %r"
                    % (component.component_id, header.path)
                )
            header_owners[header.path] = header.sha256
        staged = component.stage_verified(Path(directory) / component.manifest_sha256)
        if staged is not None:
            flags.extend(("-I", staged))
            roots[staged] = component.component_id
            authorities.append((component, staged))
    return flags, roots, tuple(authorities)
