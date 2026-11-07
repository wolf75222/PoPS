"""Source authoring and independent finite oracle, no native substitution."""
import importlib.util
from pathlib import Path
import sys


def load(name):
    path = Path(__file__).resolve().parents[3]/"examples/migration/scientific"/(name+".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


case_module = load("api040_m09_finite_native")
oracle = load("api040_m09_hoffart_oracle")
