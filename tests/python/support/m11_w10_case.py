"""Load the standalone public W10 example and its independent NumPy oracle."""

import importlib.util
from pathlib import Path


def load(name):
    path = Path(__file__).resolve().parents[3] / "examples/migration/scientific" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


case_module = load("api040_m11_w10")
oracle = load("api040_m11_w10_oracle")
