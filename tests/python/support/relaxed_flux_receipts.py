"""Decode actual accepted face measures; this module does not execute a model."""

from fractions import Fraction
import math
import struct
import numpy as np
from tests.python.support.amr_accepted_state import accepted_tagging_hysteresis_span


def exact_weight(payload):
    """Decode the canonical EW1 bytes, independently of the C++ arithmetic."""
    assert len(payload) >= 20 and payload[:3] == b"EW1"
    sign = payload[3]
    numerator_count, denominator_count = struct.unpack_from("<QQ", payload, 4)
    assert sign in (0, 1) and denominator_count > 0
    assert len(payload) == 20 + 4 * (numerator_count + denominator_count)
    offset = 20 + 4 * numerator_count
    numerator = int.from_bytes(payload[20:offset], "little")
    denominator = int.from_bytes(payload[offset:], "little")
    assert denominator and (not numerator_count or payload[offset - 4 : offset] != bytes(4))
    assert payload[-4:] != bytes(4)
    assert numerator or (numerator_count == 0 and sign == 0 and denominator == 1)
    value = Fraction(-numerator if sign else numerator, denominator)
    assert abs(value.numerator) == numerator and value.denominator == denominator
    return value


def fragments(image, *, _include_interfaces=False, _with_cursor=False):
    assert image[:8] in (b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF")
    assert int.from_bytes(image[8:16], "little") == 2
    tagging, offset = accepted_tagging_hysteresis_span(image, dimension=2)
    cursor = offset + len(tagging)

    def take(n):
        nonlocal cursor
        assert 0 <= n <= len(image) - cursor
        value = image[cursor : cursor + n]
        cursor += n
        return value

    def word(signed=False):
        return int.from_bytes(take(8), "little", signed=signed)

    def real():
        (value,) = struct.unpack("<d", take(8))
        assert math.isfinite(value)
        return value

    def text():
        return take(word()).decode()

    def rational():
        n, d = word(True), word(True)
        value = Fraction(n, d)
        assert (value.numerator, value.denominator) == (n, d)
        return value

    def weight():
        return exact_weight(take(word())) if image[:8] in (b"POPSANDE", b"POPSANDF") else rational()

    assert text() and text()
    result = []
    for axis in range(2):
        count = word()
        assert count < len(image) // 200
        for _ in range(count):
            owner, state = text(), text()
            levels = word(True), word(True)
            centering, direction = word(), word(True)
            assert centering == 0 and direction == axis
            face = word(True), word(True)
            coarse_face = word(True), word(True)
            level, macro = word(True), word(True)
            phase, evaluation = rational(), real()
            family, stage, attempt = text(), text(), word()
            role, contribution = word(), word()
            assert contribution == 0
            stage_weight, begin, end = weight(), rational(), rational()
            duration, area = real(), real()
            affine = None
            if image[:8] in (b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF"):
                tag = word()
                values = tuple(real() for _ in range(4))
                assert tag in (0, 1)
                if tag:
                    affine = values
                else:
                    assert all(struct.pack("<d", value) == bytes(8) for value in values)
            payload = np.array([real() for _ in range(word())])
            result.append(
                dict(
                    owner=owner,
                    state=state,
                    levels=levels,
                    axis=axis,
                    face=face,
                    coarse_face=coarse_face,
                    level=level,
                    macro=macro,
                    phase=phase,
                    evaluation=evaluation,
                    family=family,
                    stage=stage,
                    attempt=attempt,
                    role=role,
                    weight=stage_weight,
                    begin=begin,
                    end=end,
                    duration=duration,
                    area=area,
                    affine=affine,
                    payload=payload,
                )
            )
    assert result
    if not _include_interfaces and not _with_cursor:
        return result

    def clock():
        return (word(True), word(True), rational(), real())

    interfaces = []
    count = word()
    assert count < len(image) // 256
    for _ in range(count):
        identity, epoch = text(), word()
        levels, evaluation = (word(True), word(True)), clock()
        stage, graph, rate, application = text(), text(), text(), text()
        interval = clock(), clock()
        orientation, left, right = word(), word(), word()
        stage_weight, area, duration, resolved = weight(), real(), real(), word()
        assert orientation in (0, 1) and left != right and resolved == 1
        affine = None
        if image[:8] in (b"POPSANDD", b"POPSANDE", b"POPSANDF"):
            tag = word()
            values = tuple(real() for _ in range(4))
            assert tag in (0, 1)
            if tag:
                affine = values
            else:
                assert all(struct.pack("<d", value) == bytes(8) for value in values)
        payload = np.array([real() for _ in range(word())])
        interfaces.append(
            dict(
                identity=identity,
                epoch=epoch,
                levels=levels,
                evaluation=evaluation,
                stage=stage,
                graph=graph,
                rate=rate,
                application=application,
                interval=interval,
                orientation=orientation,
                left=left,
                right=right,
                weight=stage_weight,
                area=area,
                duration=duration,
                affine=affine,
                payload=payload,
            )
        )
    return (result, interfaces, cursor) if _with_cursor else interfaces


def interface_fragments(image):
    return fragments(image, _include_interfaces=True)


def assert_scaled(original, transformed, gamma, *, requested_duration=1.0):
    a, b = fragments(original), fragments(transformed)
    assert len(a) == len(b)
    nonzero = False
    for left, right in zip(a, b, strict=True):
        # Artifact-qualified occurrence strings differ across the two authored
        # programs; geometric quadrature and actual evaluated payloads must not.
        for key in (
            "owner",
            "levels",
            "axis",
            "face",
            "coarse_face",
            "level",
            "macro",
            "phase",
            "evaluation",
            "role",
            "weight",
            "begin",
            "end",
            "duration",
            "area",
        ):
            assert left[key] == right[key], (key, left[key], right[key])
        np.testing.assert_array_equal(left["payload"], right["payload"])
        assert left["affine"] is None
        start, requested, reached, scale = right["affine"]
        assert start == 0 and requested == requested_duration
        assert abs(reached - gamma * requested_duration) < 1.0e-14
        assert scale == reached / requested
        integral = float(left["weight"]) * left["duration"] * left["area"] * left["payload"]
        np.testing.assert_allclose(scale * integral, gamma * integral, rtol=0, atol=1.0e-16)
        nonzero |= bool(np.any(abs(integral) > 0))
    assert nonzero, "a zero-flux receiver cannot qualify accepted face scaling"


def assert_interface_scaled(original, transformed, gamma):
    before, after = interface_fragments(original), interface_fragments(transformed)
    assert before and len(before) == len(after)
    nonzero, orientations = False, set()
    for left, right in zip(before, after, strict=True):
        for key in (
            "levels",
            "evaluation",
            "interval",
            "orientation",
            "left",
            "right",
            "weight",
            "area",
            "duration",
        ):
            assert left[key] == right[key], (key, left[key], right[key])
        np.testing.assert_array_equal(left["payload"], right["payload"])
        assert left["affine"] is None
        assert right["affine"] == (0.0, 1.0, gamma, gamma)
        integral = float(right["weight"]) * right["duration"] * gamma * right["payload"]
        nonzero |= bool(np.any(abs(integral) > 0))
        orientations.add(right["orientation"])
    assert nonzero, "a zero interface cannot qualify accepted shared-flux scaling"
    assert orientations == {0, 1}, "both coarse and fine interface consumers must publish"
