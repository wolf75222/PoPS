from copy import deepcopy
import pytest

from pops.identity import make_identity
from pops.runtime._run_manifest import RunManifest


def _manifest_controls():
    return {"t_end": 3., "max_steps": 1, "output_mode": "disabled", "step_transaction": {}}


def _manifest(controls):
    return RunManifest(bind_identity=make_identity("bind", {"test": "single-step"}),
                       start_time=0., start_macro_step=0, controls=controls)


def test_step_request_identity_differs_while_legacy_run_controls_stay_unchanged():
    controls = _manifest_controls()
    original = deepcopy(controls)
    run = _manifest(controls)
    step = _manifest({**controls, "stop_condition": "one_accepted_step"})
    assert controls == original
    assert set(run.controls) == set(original)
    assert run.run_identity != step.run_identity
    assert RunManifest.from_dict(step.to_dict()).run_identity == step.run_identity
    assert RunManifest.from_dict(run.to_dict()).run_identity == run.run_identity


@pytest.mark.parametrize("changes", ({"max_steps": 0}, {"max_steps": 2},
    {"max_steps": True}, {"t_end": 0.}, {"stop_condition": "finish_early"}))
def test_step_manifest_cannot_misrepresent_its_terminal_condition(changes):
    controls = {**_manifest_controls(), "stop_condition": "one_accepted_step", **changes}
    with pytest.raises(ValueError, match="one_accepted_step"):
        _manifest(controls)
