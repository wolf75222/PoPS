"""Independent POPSCAR1 and ExactContractBuilder decoder; no PoPS imports."""
import hashlib
import math
import struct
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(raw):
    require(type(raw) is np.ndarray and raw.dtype == np.uint8 and raw.ndim == 1, "carrier typed archive differs")
    data = raw.tobytes()
    require(data[:8] == b"POPSCAR1", "carrier magic differs")
    at = 8
    def word(signed=False):
        nonlocal at
        require(at + 8 <= len(data), "carrier truncated")
        value = int.from_bytes(data[at:at+8], "little", signed=signed)
        at += 8
        return value
    dim, real, ranks, shard, levels, count = word(), word(), word(), word(True), word(), word()
    require(dim in (1, 2, 3) and real in (32, 64) and 0 < ranks <= 2**63-1 and levels > 0
            and -1 <= shard < ranks and 0 < count <= (len(data)-at)//8, "carrier authority differs")
    blocks = []
    for _ in range(count):
        length = word()
        require(0 < length <= len(data)-at, "carrier name size differs")
        name = data[at:at+length].decode("utf-8", errors="strict")
        at += length
        require(name not in blocks, "carrier duplicate block")
        blocks.append(name)
    count = word()
    require(count <= (len(data)-at)//((6+4*dim)*8), "carrier row count exceeds bytes")
    patches = []
    for _ in range(count):
        block, level, patch, components, owner = word(), word(), word(), word(), word(True)
        axes = [tuple(word(True) for _ in range(4)) for _ in range(dim)]
        size = word()
        key = (block, level, patch)
        require(block < len(blocks) and level < levels and components > 0 and -1 <= owner < ranks
                and (shard == -1 or owner in (-1, shard))
                and (not patches or patches[-1]["key"] < key), "carrier patch authority/order differs")
        require(all(glo <= lo <= hi <= ghi for lo, hi, glo, ghi in axes), "carrier geometry differs")
        require(size == components*math.prod(ghi-glo+1 for lo, hi, glo, ghi in axes)
                and size <= (len(data)-at)//8, "carrier payload shape differs")
        bits = tuple(word() for _ in range(size))
        require(real == 64 or all(v <= 2**32-1 for v in bits), "carrier scalar width differs")
        patches.append(dict(key=key, components=components, owner=owner, axes=axes, bits=bits))
    require(at == len(data), "carrier trailing bytes")
    return dict(dim=dim, real=real, ranks=ranks, shard=shard, levels=levels, blocks=blocks, patches=patches)


def carrier_hash(archive, patch, local):
    """Match append_rank_local_carrier_rows, without float conversion."""
    def frame(kind, value):
        return kind + struct.pack(">Q", len(value)) + value
    def text(value):
        return frame(b"t", value.encode())
    def scalar(value, width, signed=False):
        return frame(b"s", (b"i" if signed else b"u") + bytes([width])
                     + value.to_bytes(width, "big", signed=signed))
    block, level, index = patch["key"]
    require(0 <= level < 2**31 and 0 < patch["components"] < 2**31
            and all(-2**31 <= v < 2**31 for axis in patch["axes"] for v in axis), "carrier native int32 geometry differs")
    image = text("pops.amr.rank-local-carrier-payload") + scalar(1, 4) + scalar(archive["dim"], 4, True)
    image += text("state") + text(archive["blocks"][block]) + scalar(level, 4, True)
    image += scalar(local, 8) + scalar(index, 8) + scalar(patch["components"], 4, True) + scalar(len(patch["bits"]), 8)
    for positions in ((0, 1), (2, 3)):
        for axis in patch["axes"]:
            for position in positions:
                image += scalar(axis[position], 4, True)
    for bits in patch["bits"]:
        image += frame(b"s", (b"d" if archive["real"] == 64 else b"f") + bits.to_bytes(archive["real"]//8, "big"))
    return "pops.amr.rank-local-carrier.v1:sha256:" + hashlib.sha256(image).hexdigest()


def receive_carriers(arrays, rows_by_rank, components):
    archive = decode(arrays["state_carriers_checkpoint"])
    require(archive["dim"] == 2 and archive["real"] == 64 and archive["shard"] == -1
            and archive["ranks"] == int(arrays["n_ranks"]) and archive["levels"] == int(arrays["n_levels"])
            and archive["blocks"] == list(arrays["blocks"]), "carrier scientific envelope differs")
    require(type(rows_by_rank) is list and len(rows_by_rank) == archive["ranks"], "carrier rank registry missing")
    boxes = np.asarray(arrays["patch_boxes"])
    expected = {}
    for block, name in enumerate(archive["blocks"]):
        for level in range(archive["levels"]):
            selected = boxes[boxes[:, 0] == level]
            owners = arrays[f"dmap_{level}"]
            mode = str(arrays[f"distribution_mode_{level}"].item())
            require(mode in ("replicated", "partitioned"), "carrier distribution differs")
            for index, row in enumerate(selected):
                owner = -1 if mode == "replicated" else int(owners[index])
                expected[(block, level, index)] = (owner, [(int(row[1]), int(row[3])), (int(row[2]), int(row[4]))])
    require(set(expected) == {p["key"] for p in archive["patches"]}, "carrier complete patch inventory differs")
    expected_rows = [[] for _ in rows_by_rank]
    local_counts = {}
    for patch in archive["patches"]:
        block, level, index = patch["key"]
        owner, valid = expected[patch["key"]]
        require(patch["owner"] == owner and [(a[0], a[1]) for a in patch["axes"]] == valid, "carrier owner/valid geometry differs")
        require(patch["components"] == components[archive["blocks"][block]], "carrier component declaration differs")
        state = arrays[f"state_{archive['blocks'][block]}_{level}"]
        size = int(round(math.sqrt(state.size/patch["components"])))
        require(state.dtype == np.float64 and state.size == patch["components"]*size*size, "carrier state component width differs")
        full = np.array(patch["bits"], dtype=np.uint64).reshape(patch["components"], patch["axes"][1][3]-patch["axes"][1][2]+1, patch["axes"][0][3]-patch["axes"][0][2]+1)
        xlo, xhi, gx, _ = patch["axes"][0]; ylo, yhi, gy, _ = patch["axes"][1]
        valid_bits = full[:, ylo-gy:yhi-gy+1, xlo-gx:xhi-gx+1]
        require(valid_bits.tobytes() == state.reshape(patch["components"], size, size).view(np.uint64)[:, ylo:yhi+1, xlo:xhi+1].tobytes(), "carrier valid scalar bits differ")
        for rank in (range(archive["ranks"]) if owner == -1 else (owner,)):
            local_key = (rank, block, level)
            local = local_counts.get(local_key, 0); local_counts[local_key] = local+1
            row = ["pops.amr.rank-local-carrier-manifest@1", "state", archive["blocks"][block], str(level), str(local), str(index), str(patch["components"])]
            row += [str(v) for axis in patch["axes"] for v in axis[:2]]
            row += [str(v) for axis in patch["axes"] for v in axis[2:]]
            row += [carrier_hash(archive, patch, local)]
            expected_rows[rank].append(row)
    for rank, rows in enumerate(rows_by_rank):
        actual = [row for row in rows if len(row) > 1 and row[1] == "state"]
        require(sorted(actual) == sorted(expected_rows[rank]), "carrier native registry/full ghost hash differs")
    return archive
