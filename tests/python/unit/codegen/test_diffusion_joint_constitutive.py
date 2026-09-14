"""One typed joint native constitutive application reaches the existing face consumer."""
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.time import FixedDt
from tests.python.unit.codegen.test_native_constitutive import constitutive_function


def joint_diffusion_case(tmp_path):
    function = constitutive_function(tmp_path / "external")
    frame = Rectangle("joint-domain", lower=(0.,0.), upper=(1.,1.)).frame(Cartesian2D())
    model = pops.Model("joint-diffusion", frame=frame)
    state = model.state("U", components=("q",))
    from dataclasses import replace
    from pops.model import Signature
    function = replace(function, signature=Signature((state.space,), function.signature.output))
    application = function((state[0],), occurrence="physical-constitutive-law")
    w = application.transform[0]
    ax, ay = application.diffusivity
    flux = model.diffusive_flux("joint-flux", state=state,
        value=math.CoeffGradient(w, ((ax,0),(0,ay))))
    rate = model.rate("joint-rate", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("joint-case")
    block = case.block("joint", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1e-5))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(16,16), periodic=PeriodicAxes(frame.axes)))
    return case, layout, model


def test_joint_native_endpoint_captures_primal_derivative_and_consumes_status(tmp_path):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_kernels import _prepared_native_components
    case, layout, model = joint_diffusion_case(tmp_path)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    value = next(value for value in resolved.time._values if value.op == "diffusive_rhs")
    assert len(value.attrs["native_functions"]) == 2
    assert len(_prepared_native_components(resolved.time)) == 1
    code = emit_cpp_program(resolved.time, model=lower_and_validate(model)[0])
    assert code.count("constitutive::evaluate(") == 1
    assert code.count("constitutive::jacobian(") == 1
    assert "DiffusiveLawResult<pops::kNativeDimension> result;" in code
    assert "if (result.evaluation_status == 0) result.values" in code
    assert "error.status()" in code and "error.reason()" in code
    assert "#include <constitutive.hpp>" in code
