"""Observe existing native controller boundaries. No solver or guard expression is replaced."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
import pops
from pops._native_selector import select_native_dimension
from pops.codegen import program_graph_lowering as lowering
from pops.time import ErrorControlledDt
from pops.time._program.control import AdvanceControl, ADVANCE_ASSESSMENT_CONTRACT
from tests.python.integration.runtime import test_native_retry_transactions as euler
from tests.python.integration.runtime import test_public_diffusion_field_late_refusal as field

profile, receipt = sys.argv[1:]
selected = select_native_dimension(2)
print("native", selected.__file__, sha256(Path(selected.__file__).read_bytes()).hexdigest(), flush=True)
norm = "norm_inf" if profile == "euler" else "norm2"
emit = lowering.emit_program_graph


def observe_emission(*args, **kwargs):
    source = emit(*args, **kwargs)
    candidates = list(re.finditer(r"const pops::Real (\w+) = ctx\." + norm + r"\([^;]+\);", source))
    assert len(candidates) == 1, [(m[0], m.start()) for m in candidates]
    hit, = candidates
    observed = (
        '\n    std::fprintf(stderr, "ROM114_OBSERVED ' + norm + ' %a\\n", '
        'static_cast<double>(' + hit[1] + '));'
    )
    modified = "#include <cstdio>\n" + source[:hit.end()] + observed + source[hit.end():]
    if profile == "field-pi":
        def observe_proposal(match):
            return match[0] + ('\n    std::fprintf(stderr, "ROM114_PROPOSAL '
                               + match[1] + ' %a\\n", static_cast<double>(' + match[2] + '));')
        modified, count = re.subn(r'ctx.record_scalar\("(pi_prior_[ab])", (\w+)\);',
                                  observe_proposal, modified)
        assert count == 2
    path = Path(receipt)
    path.with_suffix(".stock.cpp").write_text(source)
    path.with_suffix(".observed.cpp").write_text(modified)
    return modified


if profile == "euler":
    original_guard = pops.Program._acceptance_guard_value
    original_strategy = pops.Program.step_strategy

    def check_candidate(program, name, value, condition, action, **kwargs):
        assert name == "dt_error_estimate"
        return original_guard(program, name, value, condition, action,
                              assessment=ADVANCE_ASSESSMENT_CONTRACT, **kwargs)

    def advance_strategy(program, strategy, **kwargs):
        assert type(strategy) is ErrorControlledDt
        program._set_advance_control(AdvanceControl(
            assessments=(("dt_error_estimate", "error_estimate"),),
            max_attempts=strategy.max_rejections + 1, max_transforms=0,
            first_dt=strategy.dt_init,
        ))
        return original_strategy(program, strategy, **kwargs)

    with patch.object(lowering, "emit_program_graph", observe_emission), \
            patch.object(pops.Program, "_acceptance_guard_value", check_candidate), \
            patch.object(pops.Program, "step_strategy", advance_strategy):
        runtime = euler.bind_case()
    before = euler.envelope(runtime)
    native = runtime._executor._s
    assert native._advance_max_attempts() == 2
    result = native._run_advance(0.125, 0.125, 1)
    assert euler.envelope(runtime) == before
    assert result["steps"] == 0 and result["attempts"] == result["rejections"] == 2
    assert result["rejection_phase"] == "assessment"
    print("euler native outcome", result, flush=True)
else:
    assert profile in ("field", "field-pi")
    world, identity = field.context()
    original_guard = pops.Program._acceptance_guard_value

    def read_pi_inputs(program, name, value, condition, action, **kwargs):
        if profile == "field-pi":
            assert name == "late_field_norm_refusal"
            error = condition.inputs[0]
            assert error.op == "reduce" and error.attrs["kind"] == "norm2"
            h = program.requested_dt()
            for label, prior in (("pi_prior_a", .025), ("pi_prior_b", .1)):
                proposal = program.scalar_expression(
                    label, (.9 * h) * ((.1 / error) ** .5) * ((program.float64(prior) / .1) ** .25))
                program.record_scalar(label, proposal)
        return original_guard(program, name, value, condition, action, **kwargs)

    with patch.object(lowering, "emit_program_graph", observe_emission), \
            patch.object(pops.Program, "_acceptance_guard_value", read_pi_inputs):
        authored, artifact = field.build(world, "euler", retry=True)
    runtime = field.bind(world, authored, artifact, 0.0)
    native = runtime._executor._s
    before = field.snapshot(world, runtime)
    diagnostics_before = native.program_diagnostics()
    time = runtime.time(), runtime.macro_step()
    assert not hasattr(native, "_run_advance")
    try:
        native.step(2 * float(field.DT))
    except Exception as error:
        assert "late_field_norm_refusal" in str(error)
        payload = getattr(error, "typed_failure_payload", b"")
        failure = {name: getattr(error, name, None) for name in ("phase", "status", "typed_failure_schema")}
        print("field native refusal", type(error).__name__, str(error), failure, repr(payload), flush=True)
    else:
        raise AssertionError("expected native guard refusal")
    assert before == field.snapshot(world, runtime)
    assert diagnostics_before == native.program_diagnostics()
    assert (runtime.time(), runtime.macro_step()) == time == (0.0, 0)
    accepted = pops.run(runtime, t_end=2 * float(field.DT), max_steps=2, console=False)
    assert accepted.accepted_steps == 2 and accepted.rejected_steps == 1
    assert all(native.history_fill_count(name, 0) == 2 for name in runtime.history_names())
    print("field legacy control", accepted.to_data(), flush=True)
