"""Immutable-authoring result object for a compiled coupled source."""
from __future__ import annotations

from typing import Any

from ._coupled_abi import (
    CS_ADD,
    CS_DIV,
    CS_MAX_PROG,
    CS_MAX_REG,
    CS_MAX_TERMS,
    CS_MUL,
    CS_NEG,
    CS_POW,
    CS_PUSHREG,
    CS_SQRT,
    CS_SUB,
    role_canonical,
)
from ._scalars import exact_physics_scalar, native_real, scalar_data_view


class CompiledCoupledSource:
    """Flat bytecode ABI plus exact authoring metadata and reference evaluators."""

    def __init__(
        self,
        name: Any,
        backend: Any,
        in_blocks: Any,
        in_roles: Any,
        consts: Any,
        out_blocks: Any,
        out_roles: Any,
        prog_ops: Any,
        prog_args: Any,
        prog_lens: Any,
        terms: Any,
        reg_order: Any,
        frequency: Any = 0,
        freq_prog_ops: Any = None,
        freq_prog_args: Any = None,
        frequency_expr: Any = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.frequency = exact_physics_scalar(
            frequency, where="CompiledCoupledSource.frequency")
        self.in_blocks = list(in_blocks)
        self.in_roles = list(in_roles)
        self.consts = [
            exact_physics_scalar(value, where="CompiledCoupledSource.consts")
            for value in consts
        ]
        self.out_blocks = list(out_blocks)
        self.out_roles = list(out_roles)
        self.prog_ops = list(prog_ops)
        self.prog_args = list(prog_args)
        self.prog_lens = list(prog_lens)
        self.freq_prog_ops = list(freq_prog_ops) if freq_prog_ops else []
        self.freq_prog_args = list(freq_prog_args) if freq_prog_args else []
        self._frequency_expr = frequency_expr
        self._terms = list(terms)
        self._reg_order = list(reg_order)

    def __repr__(self) -> str:
        return (
            "CompiledCoupledSource(name=%r, backend=%r, n_in=%d, n_const=%d, "
            "n_terms=%d, frequency=%r)"
            % (
                self.name,
                self.backend,
                len(self.in_blocks),
                len(self.consts),
                len(self.out_blocks),
                scalar_data_view(self.frequency, where="CompiledCoupledSource.frequency"),
            )
        )

    def to_data(self) -> dict[str, Any]:
        """Detached JSON-shaped inspection data without precision loss."""
        return {
            "name": self.name,
            "backend": self.backend,
            "inputs": [
                {"block": block, "role": role}
                for block, role in zip(self.in_blocks, self.in_roles, strict=True)
            ],
            "constants": [
                scalar_data_view(value, where="CompiledCoupledSource.consts")
                for value in self.consts
            ],
            "outputs": [
                {"block": block, "role": role, "program_length": length}
                for block, role, length in zip(
                    self.out_blocks, self.out_roles, self.prog_lens, strict=True)
            ],
            "frequency": {
                "constant": scalar_data_view(
                    self.frequency, where="CompiledCoupledSource.frequency"),
                "per_cell": bool(self.freq_prog_ops),
            },
        }

    def utilization(self) -> Any:
        return {
            "registers": {
                "count": len(self._reg_order) + len(self.consts),
                "limit": CS_MAX_REG,
            },
            "terms": {"count": len(self.out_blocks), "limit": CS_MAX_TERMS},
            "program": {"count": len(self.prog_ops), "limit": CS_MAX_PROG},
        }

    def _native_registers(self, fields: Any) -> list[Any]:
        env = {
            "%s::%s" % (block, role_canonical(role)): array
            for (block, role), array in fields.items()
        }
        missing = [key for key in self._reg_order if key not in env]
        if missing:
            raise KeyError("coupled-source reference fields missing: %s" % ", ".join(missing))
        registers = [env[key] for key in self._reg_order]
        registers.extend(
            native_real(value, where="CompiledCoupledSource.reference.constants[%d]" % index)
            for index, value in enumerate(self.consts)
        )
        return registers

    @staticmethod
    def _eval_program(ops: Any, args: Any, registers: Any) -> Any:
        import numpy as np

        stack = []
        for op, arg in zip(ops, args, strict=True):
            if op == CS_PUSHREG:
                stack.append(registers[arg])
            elif op == CS_NEG:
                stack.append(-stack.pop())
            elif op == CS_SQRT:
                stack.append(np.sqrt(stack.pop()))
            else:
                right, left = stack.pop(), stack.pop()
                if op == CS_ADD:
                    stack.append(left + right)
                elif op == CS_SUB:
                    stack.append(left - right)
                elif op == CS_MUL:
                    stack.append(left * right)
                elif op == CS_DIV:
                    stack.append(left / right)
                elif op == CS_POW:
                    stack.append(left ** right)
                else:
                    raise ValueError("unknown coupled-source opcode %r" % op)
        if len(stack) != 1:
            raise ValueError("invalid coupled-source bytecode stack depth %d" % len(stack))
        return stack[0]

    def reference_terms(self, fields: Any) -> Any:
        """Evaluate exactly the native bytecode after its explicit binary64 boundary."""
        registers = self._native_registers(fields)
        result, offset = [], 0
        for block, role, length in zip(
                self.out_blocks, self.out_roles, self.prog_lens, strict=True):
            end = offset + length
            result.append((block, role, self._eval_program(
                self.prog_ops[offset:end], self.prog_args[offset:end], registers)))
            offset = end
        return result

    def reference_frequency(self, fields: Any) -> Any:
        if self._frequency_expr is None:
            return None
        return self._eval_program(
            self.freq_prog_ops, self.freq_prog_args, self._native_registers(fields))


__all__ = ["CompiledCoupledSource"]
