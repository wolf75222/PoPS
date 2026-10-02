"""Immutable public-result carrier for accepted-state-storage-observation@1."""
from dataclasses import dataclass
import math
import struct

@dataclass(frozen=True, slots=True)
class AcceptedStateStorageObservation:
    dimension: int
    time: float
    macro_step: int
    rank_local: bytes
    complete: bytes
    contract: str = "accepted-state-storage-observation@1"

    def __post_init__(self):
        type(self)._validate({"contract":self.contract,"dimension":self.dimension,
            "time":self.time,"macro_step":self.macro_step,
            "rank_local":self.rank_local,"complete":self.complete})

    @staticmethod
    def _validate(data):

        if type(data) is not dict or set(data) != {"contract","dimension","time","macro_step","rank_local","complete"}:
            raise TypeError("storage observation requires one exact native record")
        if data["contract"] != "accepted-state-storage-observation@1":
            raise ValueError("unknown storage observation contract")
        if type(data["dimension"]) is not int or data["dimension"] not in (1,2,3):
            raise ValueError("invalid native observation dimension")
        if type(data["time"]) is not float or not math.isfinite(data["time"]):
            raise ValueError("invalid accepted observation time")
        if type(data["macro_step"]) is not int or data["macro_step"] < 0:
            raise ValueError("invalid accepted observation macro step")
        for key in ("rank_local","complete"):
            if type(data[key]) is not bytes or not data[key].startswith(b"POPSCAR1"):
                raise ValueError("storage observation requires exact POPSCAR1 bytes")
            if len(data[key]) < 56:
                raise ValueError("truncated storage observation header")
            dimension, real_bits, ranks, shard, levels, blocks = struct.unpack_from("<QQQqQQ",data[key],8)
            if dimension != data["dimension"] or real_bits not in (32,64) or ranks < 1 or levels < 1 or blocks < 1:
                raise ValueError("storage observation wire authority differs")
            if (key == "complete" and shard != -1) or (key == "rank_local" and not 0 <= shard < ranks):
                raise ValueError("storage observation shard authority differs")
        if data["rank_local"][8:32] != data["complete"][8:32] or data["rank_local"][40:56] != data["complete"][40:56]:
            raise ValueError("storage observation shard/archive authority differs")

    @classmethod
    def from_native(cls, data):
        cls._validate(data)
        return cls(data["dimension"],data["time"],data["macro_step"],data["rank_local"],data["complete"])
