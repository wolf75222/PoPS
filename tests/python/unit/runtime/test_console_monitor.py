from __future__ import annotations

from pops.output import ConsoleSample
from pops.output._console_monitor import ConsolePresentation


_HANDLED = []


def _handle(sample):
    _HANDLED.append(sample)


def test_console_template_formats_scalar_snapshot_and_unavailable_value(capsys):
    sample = ConsoleSample(
        time=0.25,
        step=4,
        dt=0.05,
        values={
            "tracer.integral": 1.25,
            "tracer.step_change_l2": None,
            "integral": 1.25,
            "step_change_l2": None,
            "dU_L2": None,
        },
        unavailable={
            "tracer.step_change_l2": "AMR regrid",
            "step_change_l2": "AMR regrid",
            "dU_L2": "AMR regrid",
        },
    )
    presentation = ConsolePresentation(
        template=(
            "step={step} t={time:.2f} dU={tracer.step_change_l2:.3e} "
            "mass={tracer.integral:.4f}"
        ),
        handler=None,
    )

    presentation.emit(sample)

    assert capsys.readouterr().out == (
        "step=4 t=0.25 dU=n/a (AMR regrid) mass=1.2500\n")
    assert sample["dU_L2"] is None
    assert sample.diagnostics == "integral=1.250000e+00 | dU_L2=n/a (AMR regrid)"


def test_console_handler_receives_the_exact_immutable_sample(capsys):
    _HANDLED.clear()
    sample = ConsoleSample(
        time=0.5,
        step=8,
        dt=0.1,
        values={"tracer.integral": 2.0, "integral": 2.0},
        unavailable={},
    )
    presentation = ConsolePresentation(template=None, handler=_handle)

    presentation.emit(sample)

    assert _HANDLED == [sample]
    assert capsys.readouterr().out == ""
