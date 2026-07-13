"""Composable bricks: native brick descriptors and partial DSL bricks.

A NATIVE brick (:class:`NativeBrick`) describes a hand-written C++ brick to bake
into a hybrid composite; a compiled DSL brick (:class:`CompiledBrick` and its
hyperbolic / source / elliptic subclasses) carries the generated struct text. The
partial DSL bricks (:class:`HyperbolicBrick` / :class:`SourceBrick` /
:class:`EllipticBrick`) emit ONE brick struct via the ``HyperbolicModel`` codegen
wrappers (lazy). The composer :class:`pops.physics.hybrid.HybridModel` stitches
the three slots into a single ``.so``.

Import-graph rule (Spec 4): module-scope imports stay within :mod:`pops.physics`;
codegen runs only through the lazy ``HyperbolicModel`` wrappers used at compile().
"""
from __future__ import annotations

from typing import Any

from .aux import AUX_BASE_COMPS, aux_n_aux, roles_for
from ._model import HyperbolicModel
from ._scalars import (
    canonical_scalar_data,
    codegen_physics_scalar,
    exact_physics_scalar,
    physics_scalar_cpp,
)


class NativeBrick:
    """Descriptor of a NATIVE brick (include/pops/physics/) for hybrid composition.

    Carries the C++ type of the brick, the PARAMETERS to bake into the type (public field -> value) and,
    for a hyperbolic brick, the variable layout (conservative names, n_vars, primitives,
    gamma). emit(struct_name) returns the C++ text to sew into the composite .so: a derived struct that
    fixes the parameters (or a simple `using` alias if the brick has no parameter).

    - ``kind``: 'hyperbolic' | 'source' | 'elliptic' (target slot).
    - ``fields``: dict {public C++ field name -> value}; ORDER preserved (insertion).
    - ``var_names`` / ``n_vars`` / ``prim_names`` / ``gamma``: layout metadata (hyperbolic
      slot only).
    - ``min_vars``: minimal number of variables that a TEMPLATED brick (source/elliptic) requires;
      e.g. PotentialForce indexes s[1]/s[2] so it requires >= 3 variables. Checked by HybridModel.
    - ``n_aux``: width of the aux channel that the brick READS (>= 3 if it reads B_z/T_e)."""

    def __init__(self, cpp_type: Any, kind: Any, fields: Any = None, var_names: Any = None,
                 n_vars: Any = None, prim_names: Any = None, gamma: Any = None, min_vars: int = 1,
                 n_aux: Any = AUX_BASE_COMPS) -> None:
        if kind not in ("hyperbolic", "source", "elliptic"):
            raise ValueError("NativeBrick: kind 'hyperbolic' | 'source' | 'elliptic' (got %r)" % (kind,))
        self.cpp_type = cpp_type
        self.kind = kind
        self.fields = {
            name: codegen_physics_scalar(value, where="NativeBrick.fields[%r]" % name)
            for name, value in dict(fields or {}).items()
        }
        self.var_names = list(var_names) if var_names else None
        self.n_vars = n_vars
        self.prim_names = list(prim_names) if prim_names else (list(var_names) if var_names else None)
        self.gamma = (
            None if gamma is None
            else exact_physics_scalar(gamma, where="NativeBrick.gamma")
        )
        self.min_vars = min_vars
        self.n_aux = n_aux

    def emit(self, struct_name: Any, namespace: str = "pops_generated") -> str:
        """C++ text of the brick sewn into the composite .so. Without a parameter -> `using` alias
        (zero cost); with parameters -> a derived struct that fixes them in its host constructor
        (the values are WRITTEN HARD, like an inlined DSL constant)."""
        if not self.fields:
            return "namespace %s { using %s = %s; }\n" % (namespace, struct_name, self.cpp_type)
        sets = " ".join(
            "%s = pops::Real(%s);"
            % (name, physics_scalar_cpp(value, where="NativeBrick.fields[%r]" % name))
            for name, value in self.fields.items()
        )
        return ("namespace %s { struct %s : %s { %s() { %s } }; }\n"
                % (namespace, struct_name, self.cpp_type, struct_name, sets))

    def to_data(self) -> dict[str, Any]:
        """Detached JSON-ready descriptor retaining exact field identities."""
        data = {
            "cpp_type": self.cpp_type,
            "kind": self.kind,
            "fields": {
                name: canonical_scalar_data(value, where="NativeBrick.fields[%r]" % name)
                for name, value in self.fields.items()
            },
            "var_names": None if self.var_names is None else list(self.var_names),
            "n_vars": self.n_vars,
            "prim_names": None if self.prim_names is None else list(self.prim_names),
            "min_vars": self.min_vars,
            "n_aux": self.n_aux,
        }
        if self.gamma is not None:
            data["gamma"] = canonical_scalar_data(self.gamma, where="NativeBrick.gamma")
        return data


class CompiledBrick:
    """Result of <partial DSL brick>.compile(): the C++ of ONE brick (the generated struct) + its
    metadata, ready to be sewn into a hybrid CompositeModel. The MACHINE compilation happens at the
    level of the composite (a single .so); this object carries the brick already GENERATED and frozen."""

    def __init__(self, kind: Any, struct_src: Any, type_name: Any, n_vars: Any = None,
                 n_aux: Any = AUX_BASE_COMPS, cons_names: Any = None, cons_roles: Any = None,
                 prim_names: Any = None, gamma: Any = None, hash_part: str = "",
                 wave_speeds: bool = True) -> None:
        self.kind = kind                 # 'hyperbolic' | 'source' | 'elliptic'
        self.struct_src = struct_src     # C++ text of the struct (namespace pops_generated { struct ... })
        self.type_name = type_name       # qualified type to place in CompositeModel<...>
        self.n_vars = n_vars             # layout (hyperbolic) or declared number of variables (src/ell)
        self.n_aux = n_aux
        self.cons_names = list(cons_names) if cons_names else []
        self.cons_roles = list(cons_roles) if cons_roles else []
        self.prim_names = list(prim_names) if prim_names else []
        self.gamma = (
            None if gamma is None
            else exact_physics_scalar(gamma, where="CompiledBrick.gamma")
        )
        self.hash_part = hash_part       # stable hash slice (formulas) for the composite cache key
        # wave_speeds emitted by the struct (DSL hyperbolic brick: 'p' OR explicit pair); True by
        # default = unknown (native brick): we let the C++ requires-gate decide (historical).
        self.has_wave_speeds = bool(wave_speeds)

    def __repr__(self) -> str:
        return "CompiledBrick(kind=%r, type=%r, n_vars=%r)" % (self.kind, self.type_name, self.n_vars)


class CompiledHyperbolicBrick(CompiledBrick):
    """Compiled DSL hyperbolic brick (vars/flux/eigenvalues/conversions)."""
    def __init__(self, **kw: Any) -> None: super().__init__("hyperbolic", **kw)


class CompiledSourceBrick(CompiledBrick):
    """Compiled DSL source brick (apply(U, aux))."""
    def __init__(self, **kw: Any) -> None: super().__init__("source", **kw)


class CompiledEllipticBrick(CompiledBrick):
    """Compiled DSL elliptic right-hand side brick (rhs(U))."""
    def __init__(self, **kw: Any) -> None: super().__init__("elliptic", **kw)


class HyperbolicBrick:
    """PARTIAL hyperbolic DSL brick (variables/flux/eigenvalues/conversions), composable with
    native or DSL bricks for the source and the elliptic. Same surface as dsl.Model but limited
    to the hyperbolic slot. compile() -> CompiledHyperbolicBrick."""

    def __init__(self, name: Any) -> None:
        self._m = HyperbolicModel(name)

    @property
    def name(self) -> Any: return self._m.name

    def conservative_vars(self, *names: Any, roles: Any = None) -> Any:
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name: Any, expr: Any) -> Any:
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars: Any, roles: Any = None) -> None:
        """ORDERED layout of Prim (positional form, names/Var already defined)."""
        self._m.set_primitive_state(*vars, roles=roles)

    def aux(self, name: Any) -> Any: return self._m.aux(name)
    def flux(self, x: Any, y: Any) -> None: self._m.set_flux(x, y)
    def eigenvalues(self, x: Any, y: Any) -> None: self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x: Any, y: Any) -> None:
        """Explicit SIGNED wave speeds (smin, smax) per direction, WITHOUT requiring 'p' --
        same contract as Model.wave_speeds (the brick struct goes through emit_cpp_brick, which
        emits wave_speeds from the pair ; the hybrid CompositeModel forwards it to the HLL gate)."""
        self._m.set_wave_speeds(x, y)

    def conservative_from(self, exprs: Any) -> None: self._m.set_conservative_from(exprs)
    def gamma(self, value: Any) -> None: self._m.set_gamma(value)
    def check(self) -> Any: return self._m.check()

    def compile(self) -> Any:
        """Validate + emit the hyperbolic C++ struct (emit_cpp_brick) -> CompiledHyperbolicBrick."""
        self._m.check()
        struct_name = "Hyp" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_brick(name=struct_name)
        return CompiledHyperbolicBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, cons_names=list(self._m.cons_names),
            cons_roles=roles_for(self._m.cons_names, self._m.cons_roles),
            prim_names=list(self._m.prim_state), gamma=self._m.gamma,
            n_aux=aux_n_aux(self._m.aux_names), hash_part=self._m._model_hash(),
            wave_speeds=("p" in self._m.prim_defs or self._m._wave_speeds is not None))


class SourceBrick:
    """PARTIAL DSL brick for a source S(U, aux), composable with a native or DSL transport and
    elliptic. Declares its conservatives (the layout must match the transport) + its aux fields
    + the source formula. compile() -> CompiledSourceBrick."""

    def __init__(self, name: Any) -> None:
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names: Any, roles: Any = None) -> Any:
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name: Any, expr: Any) -> Any: return self._m.primitive(name, expr)
    def aux(self, name: Any) -> Any: return self._m.aux(name)
    def source(self, s: Any) -> None: self._m.set_source(s)

    def compile(self) -> Any:
        """Validate + emit the source C++ struct (emit_cpp_source) -> CompiledSourceBrick."""
        if self._m._source is None:
            raise ValueError("SourceBrick.compile: call source([...]) first")
        struct_name = "Src" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_source(name=struct_name)
        return CompiledSourceBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=aux_n_aux(self._m.aux_names),
            hash_part=self._m._model_hash())


class EllipticBrick:
    """PARTIAL DSL brick for an elliptic right-hand side rhs(U), composable with a native or DSL
    transport and source. Declares its conservatives (layout) + the right-hand side formula.
    compile() -> CompiledEllipticBrick."""

    def __init__(self, name: Any) -> None:
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names: Any, roles: Any = None) -> Any:
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name: Any, expr: Any) -> Any: return self._m.primitive(name, expr)
    def elliptic_rhs(self, e: Any) -> None: self._m.set_elliptic_rhs(e)

    def compile(self) -> Any:
        """Validate + emit the elliptic C++ struct (emit_cpp_elliptic) -> CompiledEllipticBrick."""
        if self._m._elliptic is None:
            raise ValueError("EllipticBrick.compile: call elliptic_rhs(...) first")
        struct_name = "Ell" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_elliptic(name=struct_name)
        return CompiledEllipticBrick(
            struct_src=struct_src, type_name="pops_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=AUX_BASE_COMPS, hash_part=self._m._model_hash())
