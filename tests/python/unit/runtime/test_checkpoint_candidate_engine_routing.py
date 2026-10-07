"""Real Uniform checkpoint-candidate routing regression after Native installation."""
import inspect

import pytest

from pops.runtime._runtime_instance import RuntimeInstance


@pytest.mark.native_loader
def test_native_uniform_candidate_probe_preserves_required_path_capture():
    # Run only with the rebuilt/installed ABI12 package. Import the actual
    # classes and their mandatory Native module; no reconstructed class or mock.
    from pops.runtime._system import System
    from pops.runtime._system_io import _SystemIO
    from tests.python.integration.runtime.test_runtime_inspection_reports import _system_config

    uniform = System(_system_config(4))
    assert System._prepare_checkpoint_capture is _SystemIO._prepare_checkpoint_capture
    assert System._checkpoint_precreated_inode is _SystemIO._checkpoint_precreated_inode
    assert inspect.signature(uniform._prepare_checkpoint_capture).parameters[
        "path"].default is inspect.Parameter.empty
    runtime = RuntimeInstance.__new__(RuntimeInstance)
    runtime._executor = uniform
    # The old noargs call to capture(path) raises TypeError on this real class.
    assert runtime._prepare_checkpoint_candidate() is None
