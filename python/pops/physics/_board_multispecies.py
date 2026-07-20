"""Board-facade mixin: multi-species lowering and inspection.

Splits the multi-species authoring (``species`` promotion, ``coupled_rate`` and
block-owned ``field_provider`` declarations) and the inspection/dump helpers out of the board
:class:`pops.physics.board.Model` so neither file exceeds the Spec-4 500-line
bound. Methods only; they operate on the board ``Model`` instance attributes
(``_multi_module`` / ``_species`` / ``_states`` / ``_dsl`` / ...). Lowers to the
operator-first multi-block IR (:mod:`pops.model`); codegen-free, ``_pops``-free.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._board_contract import (atomic_attrs, normalize_components, normalize_roles,
                              normalize_sequence, require_name)
from .board_handles import StateHandle, _canon_role, _safe_name

if TYPE_CHECKING:
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


class _MultiSpeciesMixin(_BoardModel):
    """Multi-species promotion, coupled rates, field providers, and inspection dumps."""

    def _promote_to_multispecies(self, extra: Any = None) -> Any:
        """Build the multi-block :class:`pops.model.Module` and migrate the first species into it.

        The single-state dsl model authored the first species; multi-species mode realizes every
        species as a typed StateSpace on a shared Module so N >= 2 species lower to the existing
        operator-first multi-block IR (N spaces + coupled rates + block-owned field providers), not a
        second runtime. Promotion replaces registry metadata with a new, equal immutable handle;
        an earlier user reference retains the same qualified identity and expression components."""
        if self._multi_module is not None:
            if extra is None:
                return None
            return self._add_species(extra[0], components=extra[1], roles=extra[2])
        from .. import model as _model
        candidate = _model.Module(self.name, owner=self.owner_path)
        # Promotion changes the state-space view, not parameter ownership.
        candidate._param_registry = self._dsl._param_registry
        promoted = {}
        for nm, h in self._species.items():
            promoted[nm] = self._declare_species_on(
                candidate, nm, h.components, dict(h.roles))
        result = None
        if extra is not None:
            name, components, roles = extra
            result = self._declare_species_on(candidate, name, components, roles)
            promoted[name] = result
        self._migrate_first_species_local_transforms(candidate, promoted)
        self._multi_module = candidate
        self._species.update(promoted)
        self._states.update(promoted)
        return result

    def _add_species(self, name: Any, components: Any = (), roles: Any = None) -> Any:
        """Add one typed StateSpace to the multi-block Module atomically."""
        name = require_name(name, "species name")
        comps = normalize_components(components, "species %s state" % name)
        role_map = normalize_roles(roles, comps, "species %s" % name)
        if name in self._species:
            raise ValueError("species %r is already declared" % name)
        module = self._multi_module
        with atomic_attrs((module, "_state_spaces"), (self, "_species"), (self, "_states")):
            handle = self._declare_species_on(module, name, comps, role_map)
            self._species[handle.name] = handle
            self._states[handle.name] = handle
        return handle

    def _declare_species_on(self, module: Any, name: Any, components: Any,
                            roles: Any) -> StateHandle:
        """Build one complete typed species on an unpublished/guarded Module."""
        name = require_name(name, "species name")
        comps = normalize_components(components, "species %s state" % name)
        role_map = normalize_roles(roles, comps, "species %s" % name)
        canon = {component: _canon_role(role) for component, role in role_map.items()}
        space = module.state_space(name, comps, roles=canon)
        vars_ = module.state_symbols(space)
        return StateHandle(
            name, comps, vars_, role_map, owner=self.owner_path, space=space)

    def _local_transform_body(
        self, name: Any, *, on: StateHandle, expressions: Any, valid_if: Any,
    ) -> dict[str, Any]:
        """Validate one exact block-local symbolic map without mutating a registry."""
        from pops._ir import Var, _children, _wrap
        from pops.model.state_symbols import state_component_symbol

        name = require_name(name, "local_transform name")
        if not name.isidentifier():
            raise ValueError(
                "local_transform(%r): name must be a valid identifier "
                "(letters/digits/_, no leading digit)" % name)
        values = normalize_sequence(
            expressions, "local_transform %s expressions" % name, nonempty=True)
        if len(values) != len(on.components):
            raise ValueError(
                "local_transform(%r) on StateSpace %r has %d expression(s), expected %d"
                % (name, on.space.name, len(values), len(on.components)))
        wrapped = tuple(_wrap(self._to_expr(value)) for value in values)
        predicate = _wrap(self._to_expr(valid_if))

        state_spaces = self._multi_module.state_spaces()
        known_symbols = {
            state_component_symbol(space, component): (space.name, component)
            for space in state_spaces.values()
            for component in space.components
        }
        allowed_symbols = {
            state_component_symbol(on.space, component)
            for component in on.space.components
        }
        aux_reads = set()
        seen = set()
        stack = [*wrapped, predicate]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, Var):
                if node.kind in ("cons", "prim") and node.name not in allowed_symbols:
                    foreign = known_symbols.get(node.name)
                    if foreign is not None:
                        raise ValueError(
                            "local_transform(%r) on StateSpace %r reads component %r from "
                            "StateSpace %r; a local transform may read only its exact on= state"
                            % (name, on.space.name, foreign[1], foreign[0]))
                    raise ValueError(
                        "local_transform(%r) on StateSpace %r reads an unauthenticated %s "
                        "coordinate %r"
                        % (name, on.space.name, node.kind, node.name))
                if node.kind == "aux":
                    aux_reads.add(node.name)
            stack.extend(_children(node))
        if aux_reads:
            raise ValueError(
                "local_transform(%r) reads field coordinate(s) %s, but the board API has no "
                "implicit field selection for a multi-state transform; author an exact "
                "Module.operator signature (StateSpace, FieldSpace) instead"
                % (name, sorted(aux_reads)))
        return {"expressions": wrapped, "valid_if": predicate}

    def _register_multispecies_local_transform(
        self, name: Any, *, on: StateHandle, expressions: Any, valid_if: Any,
    ) -> Any:
        """Register one exact ``StateSpace -> same StateSpace`` transform atomically."""
        from .. import model as _model

        species = self._species_handle("local_transform", name, on)
        body = self._local_transform_body(
            name, on=species, expressions=expressions, valid_if=valid_if)
        module = self._multi_module
        registry = module.operator_registry()
        with atomic_attrs(
            (registry, "_by_name"),
            (registry, "_order"),
            (self, "_module_cache"),
        ):
            module.operator(
                name=name,
                kind="local_transform",
                signature=_model.Signature((species.space,), species.space),
                capabilities={
                    "local": True,
                    "supports_device": True,
                    "fail_closed": True,
                },
                expr=body,
            )
            self._invalidate_authoring_views()
            return self._registered_operator_handle(name)

    def _migrate_first_species_local_transforms(
        self, candidate: Any, promoted: Mapping[str, StateHandle],
    ) -> None:
        """Move pre-promotion transforms into the unpublished multi-state Module.

        A handle returned before the second species remains usable because operator identity is
        owner/name/kind based; its derived single-state signature is re-authenticated against this
        exact promoted registry when the Program calls it.
        """
        transforms = self._dsl._m._local_transforms
        if not transforms:
            return
        if len(self._species) != 1:
            raise RuntimeError(
                "local-transform promotion requires one exact pre-existing species")
        first_name = next(iter(self._species))
        first = promoted[first_name]
        previous_module = self._multi_module
        self._multi_module = candidate
        try:
            from .. import model as _model

            for name in sorted(transforms):
                transform = transforms[name]
                body = self._local_transform_body(
                    name,
                    on=first,
                    expressions=transform["expressions"],
                    valid_if=transform["valid_if"],
                )
                candidate.operator(
                    name=name,
                    kind="local_transform",
                    signature=_model.Signature((first.space,), first.space),
                    capabilities={
                        "local": True,
                        "supports_device": True,
                        "fail_closed": True,
                    },
                    expr=body,
                )
        finally:
            self._multi_module = previous_module

    # --- quantities ---

    def field_provider(self, name: Any, *, on: Any, into: Any, value: Any) -> Any:
        """Declare one species-owned contribution to a solved field RHS.

        Providers remain distinct from the Case-owned field solve: they state only which local
        source density a species contributes to the shared field space.
        """
        from .. import model as _model

        if self._multi_module is None:
            raise ValueError(
                "field_provider(%r) requires a multi-species Model" % name)
        species = self._species_handle("field_provider", name, on)
        if (not isinstance(into, _model.Handle) or into.owner_path != self.owner_path
                or self._fields.get(into.local_id) != into):
            raise ValueError(
                "field_provider(%r) into= must be a field declared by this Model" % name)
        fields = self._multi_module.field_spaces().get(into.local_id)
        if fields is None:
            raise ValueError(
                "field_provider(%r) output field %r has no typed FieldSpace"
                % (name, into.local_id))
        reg = _safe_name(name)
        self._multi_module.operator(
            name=reg,
            kind="field_operator",
            signature=_model.Signature((species.space,), fields),
            expr=self._to_expr(value),
        )
        self._invalidate_authoring_views()
        return self._registered_operator_handle(reg)

    def coupled_rate(self, name: Any, inputs: Any = (), outputs: Any = None, preserves: Any = None,
                     dissipates: Any = None) -> Any:
        """Declare a coupled rate over several species (collisions, ionization, radiation).

        ``inputs`` is the ordered list of participating species (:class:`StateHandle`); a species
        may appear as a READ-ONLY catalyst input without being an output block. ``outputs`` maps
        each output species to its per-component rate formulas (one expression per cons component,
        written over the input species' cons vars via ``e["ne"]``). Arbitrary arity: 2, 3, 4, ...
        inputs, no two-input limit.

        Lowers to the existing operator-first ``coupled_rate`` operator (the SAME kind #287/#300
        lower): a :class:`pops.model.RateBundle` signature over the input :class:`StateSpace` set,
        with the per-block component formulas as the operator body. ``preserves`` / ``dissipates``
        are recorded as capabilities (a generic invariant tag), not numerics. Requires multi-species
        mode (declare the species with :meth:`species`).
        """
        from .. import model as _model
        reg = _safe_name(name)
        if self._multi_module is None:
            raise ValueError(
                "coupled_rate(%r) needs at least two species; declare them with m.species(...)"
                % (name,))
        in_handles = self._as_species_list("coupled_rate", name, inputs)
        if not isinstance(outputs, Mapping) or len(outputs) == 0:
            raise ValueError("coupled_rate(%r) requires outputs={species: [per-component exprs]}"
                             % (name,))
        in_spaces = tuple(h.space for h in in_handles)
        output_specs = []
        for sp, comps in outputs.items():
            h = self._species_handle("coupled_rate", name, sp)
            if h not in in_handles:
                raise ValueError(
                    "coupled_rate(%r) output species %r must also be declared in inputs"
                    % (name, h.name))
            comp_values = normalize_sequence(
                comps, "coupled_rate %s output %s" % (reg, h.name), nonempty=True)
            if len(comp_values) != len(h.components):
                raise ValueError(
                    "coupled_rate(%r) output %r has %d component formula(s) but its state %r has %d"
                    % (name, h.name, len(comp_values), h.name, len(h.components)))
            output_specs.append((h, comp_values))
        caps = {}
        if preserves is not None:
            caps["preserves"] = preserves
        if dissipates is not None:
            caps["dissipates"] = dissipates
        registry = self._multi_module.operator_registry()
        hyp = self._dsl._m
        with atomic_attrs((registry, "_by_name"), (registry, "_order"),
                          (hyp, "aux_names"), (hyp, "aux_extra_names")):
            rate_entries = {handle.name: handle.space for handle, _ in output_specs}
            expr = {
                handle.name: [self._to_expr(value) for value in values]
                for handle, values in output_specs
            }
            bundle = _model.RateBundle(rate_entries)
            signature = _model.Signature(in_spaces, bundle)
            self._multi_module.operator(
                name=reg, kind="coupled_rate", signature=signature,
                capabilities=caps or None, expr=expr)
            result = self._registered_operator_handle(reg)
        return result

    def _as_species_list(self, op: Any, name: Any, items: Any) -> Any:
        """Resolve a list of species handles / names to StateHandles (multi-species mode)."""
        values = self._as_iter(items)
        if not values:
            raise ValueError("%s(%r) requires inputs=[species, ...]" % (op, name))
        handles = [self._species_handle(op, name, species) for species in values]
        if len(set(handles)) != len(handles):
            raise ValueError("%s(%r) inputs must not repeat a species" % (op, name))
        return handles

    def _species_handle(self, op: Any, name: Any, sp: Any) -> Any:
        """Authenticate one exact species handle; names never select dependencies."""
        if not isinstance(sp, StateHandle):
            raise TypeError(
                "%s(%r): species dependencies require exact StateHandle objects, not names"
                % (op, name))
        if sp.owner_path != self.owner_path:
            raise ValueError(
                "%s(%r): species handle %r belongs to another physics model"
                % (op, name, sp.name))
        handle = self._species.get(sp.name)
        if handle != sp:
            handle = None
        if handle is None:
            known = ", ".join(self._species) or "<none>"
            raise KeyError("%s(%r): unknown species %r (declared: %s)"
                           % (op, name, sp, known))
        return handle

    @staticmethod
    def _as_iter(x: Any) -> Any:
        """A list view of a single item or an iterable (so inputs=e and inputs=[e, i] both work)."""
        if x is None:
            return []
        if isinstance(x, StateHandle):
            return [x]
        return list(normalize_sequence(x, "species inputs"))


    def list_operators(self) -> Any:
        if self._multi_module is not None:
            return self._multi_module.list_operators()
        return self._dsl.list_operators()

    def operator_alias(self, name: Any) -> Any:
        """The registered operator name for a board role name (``operator(...)``)."""
        return self._aliases.get(name, name)

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
    def dump_physics(self) -> str:
        """A board-level view of what was declared (states, params, fields, fluxes,
        sources, operators) -- the layer-1 surface."""
        lines = ["# physics.Model %s" % self.name]
        lines.append("states: %s" % {n: list(h.components) for n, h in self._states.items()})
        lines.append("params: %s" % list(self._dsl.params))
        lines.append("fields: %s" % list(self._fields))
        lines.append("fluxes: %s" % list(self._fluxes))
        lines.append("sources: %s" % list(self._sources))
        lines.append("invariants: %s" % list(self._invariants))
        lines.append("operators: %s" % self.list_operators())
        return "\n".join(lines)

    def dump_module_ir(self) -> str:
        """The operator-first :class:`pops.model.Module` this model lowers to: the typed
        spaces and operators with signatures (layer 2)."""
        mod = self.module
        reg = mod.operator_registry()
        lines = ["# pops.model.Module %s" % mod.name]
        for n, s in mod.state_spaces().items():
            lines.append("StateSpace %s: %s" % (n, list(s.components)))
        for n, f in mod.field_spaces().items():
            lines.append("FieldSpace %s: %s" % (n, list(f.components)))
        for op in mod.list_operators():
            lines.append("Operator %s [%s]: %r" % (op, reg.get(op).kind, mod.operator_signature(op)))
        return "\n".join(lines)

    def dump_capabilities(self) -> str:
        """The requirements / capabilities declared by each typed operator."""
        mod = self.module
        lines = ["# capabilities / requirements of %s" % mod.name]
        for op in mod.list_operators():
            lines.append("%s: caps=%s reqs=%s"
                         % (op, mod.operator_capabilities(op), mod.operator_requirements(op)))
        return "\n".join(lines)
