"""Explicit geometric V-cycle algorithm controls, independent of equation regions."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundoffAwareResidual:
    """Explicit binary64 gamma bound for physical five-point residual evaluation.

    This changes the numerical stopping rule, independently of physics and time integration.
    Native reports retain both the requested tolerance and the applied evaluation bound.
    """
    __pops_ir_immutable__ = True

    def to_data(self) -> dict:
        return {"policy": "binary64-five-point-gamma-bound", "version": 1}


@dataclass(frozen=True, slots=True)
class RedBlackVcycle:
    """Red-black GS with optional constant correction injection and bottom pre-smoothing."""
    __pops_ir_immutable__ = True

    constant_injection: bool = True
    pre_smooth_bottom: bool = True
    residual_evaluation: RoundoffAwareResidual | None = None

    def __post_init__(self) -> None:
        if type(self.constant_injection) is not bool or type(self.pre_smooth_bottom) is not bool:
            raise TypeError("RedBlackVcycle controls must be exact booleans")
        if self.residual_evaluation is not None and type(self.residual_evaluation) is not RoundoffAwareResidual:
            raise TypeError("RedBlackVcycle residual_evaluation must be an exact RoundoffAwareResidual")

    def to_data(self) -> dict:
        result = {"schema_version": 1, "smoother": "red-black-gauss-seidel",
                "constant_injection": self.constant_injection,
                "pre_smooth_bottom": self.pre_smooth_bottom}
        if self.residual_evaluation is not None:
            result["schema_version"] = 2
            result["residual_evaluation"] = self.residual_evaluation.to_data()
        return result


__all__ = ["RedBlackVcycle", "RoundoffAwareResidual"]
