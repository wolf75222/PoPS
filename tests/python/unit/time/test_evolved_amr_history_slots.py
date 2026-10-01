"""Protocol-only probes; synthetic raw slots are not native/scientific evidence."""
import struct

import numpy as np
import pytest

from tests.python.support.evolved_stage_amr import DT, published_history_image


def sample(name="T0", level=1, step=2):
    # Independent literal encoder of the native codec's little-endian POPSHID1 layout.
    points = ((0., DT), ((step-1)*DT, DT))
    result = b"POPSHID1"+struct.pack("<Q", len(name.encode()))+name.encode()+struct.pack("<qQ", level, 2)
    for start, duration in points:
        result += struct.pack("<QddQ", 2, start, duration, 1)
    return result


@pytest.mark.parametrize("step", (1, 2))
def test_postpublication_labels_read_latest_one_and_previous_zero(step):
    previous, latest = np.array((11., 17.)), np.array((23., 29.))
    raw = (previous, latest)
    image = published_history_image("T0", 1, raw, (DT, DT), sample(step=step), step)
    np.testing.assert_array_equal(image["T0"], latest)
    np.testing.assert_array_equal(image["T0-previous"], previous)
    assert image["history_sample_identity_T0"].tobytes() == sample(step=step)
    assert not np.shares_memory(image["T0"], latest)
    assert not np.shares_memory(image["T0-previous"], previous)


@pytest.mark.parametrize("attack", ("name", "level", "swapped-points", "duration", "ordinal", "wire", "truncated", "outgoing-dt"))
def test_slot_labels_refuse_foreign_or_stale_publication_authority(attack):
    encoded = sample()
    durations = (DT, DT)
    if attack == "name":
        encoded = sample(name="z0")
    elif attack == "level":
        encoded = sample(level=0)
    elif attack == "swapped-points":
        encoded = encoded[:-64]+encoded[-32:]+encoded[-64:-32]
    elif attack == "duration":
        encoded = encoded[:-16]+struct.pack("<dQ", 2*DT, 1)
    elif attack == "ordinal":
        encoded = encoded[:-8]+struct.pack("<Q", 2)
    elif attack == "wire":
        encoded = b"POPSHID2"+encoded[8:]
    elif attack == "truncated":
        encoded = encoded[:-1]
    elif attack == "outgoing-dt":
        durations = (DT, 2*DT)
    with pytest.raises(ValueError, match="name/level/window/slot authority"):
        published_history_image("T0", 1, (np.array((1.,)), np.array((2.,))), durations, encoded, 2)
