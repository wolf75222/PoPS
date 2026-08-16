"""Private scalar constructor compatibility stays ahead of exact native config setters."""

from __future__ import annotations

import pytest

from pops.runtime._private_config_compat import private_constructor_config


class _Config:
    def __init__(self, dimension: int):
        self.shape = (4,) * dimension
        self.lower = (0.0,) * dimension
        self.upper = (1.0,) * dimension
        self.periodicity = (False,) * dimension


def _config_type(dimension: int):
    return lambda: _Config(dimension)


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_scalar_geometry_aliases_follow_the_default_native_rank(dimension):
    config = private_constructor_config(
        _config_type(dimension),
        {"n": 12, "L": 3.5, "xlo": -2.0, "periodicity": (True,) * dimension},
        runtime="System",
        adaptive=False,
    )

    assert config.shape == (12,) * dimension
    assert config.lower == (-2.0,) + (0.0,) * (dimension - 1)
    assert config.upper == (1.5,) + (3.5,) * (dimension - 1)
    assert config.periodicity == (True,) * dimension


def test_amr_axis_one_aliases_only_override_axis_one_and_scalar_grid_is_ranked():
    config = private_constructor_config(
        _config_type(3),
        {
            "n": 12,
            "L": 3.0,
            "xlo": -2.0,
            "ny": 7,
            "Ly": 5.0,
            "ylo": 1.5,
            "coarse_max_grid": 6,
            "periodicity": (True, False, True),
        },
        runtime="AmrSystem",
        adaptive=True,
    )

    assert config.shape == (12, 7, 12)
    assert config.lower == (-2.0, 1.5, 0.0)
    assert config.upper == (1.0, 6.5, 3.0)
    assert config.coarse_max_grid == (6, 6, 6)


def test_exact_geometry_accepts_an_exact_amr_grid_tuple_unchanged():
    grid = (3, 5)
    config = private_constructor_config(
        _config_type(2),
        {
            "shape": (8, 6),
            "lower": (-1.0, 2.0),
            "upper": (3.0, 7.0),
            "coarse_max_grid": grid,
            "periodicity": (False, True),
        },
        runtime="AmrSystem",
        adaptive=True,
    )

    assert config.shape == (8, 6)
    assert config.lower == (-1.0, 2.0)
    assert config.upper == (3.0, 7.0)
    assert config.coarse_max_grid is grid


@pytest.mark.parametrize(
    ("keywords", "adaptive", "error", "match"),
    (
        ({"n": 8, "shape": (8, 8)}, False, ValueError, "cannot mix scalar aliases"),
        ({"periodicity": True}, False, TypeError, "exact bool tuple"),
        ({"periodicity": (True,)}, False, TypeError, "exact bool tuple"),
        ({"ny": 8}, False, TypeError, "no ny/Ly/ylo"),
        ({"ny": 8}, True, ValueError, "at least two"),
        ({"regrid_grow": 2}, True, NotImplementedError, "retired"),
        ({"regrid_margin": 2}, True, NotImplementedError, "retired"),
        ({"nz": 8}, True, TypeError, "no z-axis"),
    ),
)
def test_private_alias_conflicts_rank_errors_and_refusals_are_explicit(
    keywords, adaptive, error, match
):
    dimension = 1 if keywords == {"ny": 8} and adaptive else 2
    with pytest.raises(error, match=match):
        private_constructor_config(
            _config_type(dimension), keywords, runtime="AmrSystem" if adaptive else "System",
            adaptive=adaptive,
        )


def test_private_system_constructor_normalizes_before_the_native_bridge(monkeypatch):
    import pops.runtime._system as module

    class Config(_Config):
        def __init__(self):
            super().__init__(2)

    captured = []
    monkeypatch.setattr(module, "SystemConfig", Config)
    monkeypatch.setattr(module, "_System", lambda config: captured.append(config) or object())
    monkeypatch.setattr(module._threading, "_first_system_built", False)

    module.System(n=9, L=2.0, xlo=-1.0, periodicity=(True, False))

    assert len(captured) == 1
    assert captured[0].shape == (9, 9)
    assert captured[0].lower == (-1.0, 0.0)
    assert captured[0].upper == (1.0, 2.0)


def test_private_amr_constructor_normalizes_before_the_native_bridge(monkeypatch):
    import pops.runtime._amr_system as module

    class Config(_Config):
        def __init__(self):
            super().__init__(2)
            self.regrid_every = 0

    captured = []
    monkeypatch.setattr(module, "AmrSystemConfig", Config)
    monkeypatch.setattr(module, "_AmrSystem", lambda config: captured.append(config) or object())
    monkeypatch.setattr(module._threading, "_first_system_built", False)

    module.AmrSystem(n=9, L=2.0, ny=5, Ly=4.0, ylo=-1.0, coarse_max_grid=3)

    assert len(captured) == 1
    assert captured[0].shape == (9, 5)
    assert captured[0].lower == (0.0, -1.0)
    assert captured[0].upper == (2.0, 3.0)
    assert captured[0].coarse_max_grid == (3, 3)
