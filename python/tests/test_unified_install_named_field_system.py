"""Named elliptic fields on the uniform System install through ``sim.install(...)``.

These host tests avoid a real native engine by using a local recording System subclass. They still
enter through the public install API and assert the resulting solver routing, instance binding, and
validation behavior.
"""
import sys

try:
    import pops
    from pops.codegen.loader import CompiledModel
    from pops.solvers import GeometricMG
except Exception as exc:  # noqa: BLE001
    msg = "skip test_unified_install_named_field_system (pops unavailable: %s)" % exc
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    print(msg)
    sys.exit(0)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _compiled_model(fields=()):
    return CompiledModel(
        so_path="/fake-model.so",
        backend=pops.codegen.Production(),
        adder="add_native_block",
        cons_names=["rho"],
        cons_roles=["Density"],
        prim_names=["rho"],
        n_vars=1,
        gamma=None,
        n_aux=0,
        params={},
        caps={},
        abi_key="k",
        model_hash="h",
        cxx="c++",
        std="c++20",
        elliptic_field_names=list(fields),
    )


class _NoArguments:
    instances = {}
    params = {}
    aux = {}
    solvers = {}


class _CompiledProblem:
    def __init__(self, model):
        self.so_path = "/fake-problem.so"
        self.model = model

    def arguments(self):
        return _NoArguments()


class _RecordingSystem(pops.System):
    """Local fake System: public install entry point, recorded native side effects."""

    def __init__(self):
        self.calls = []
        self._aux_field_index = {}
        self._program_cadence_cfl = None
        self._output_policies = []

    def block_names(self):
        return []

    def _set_poisson(self, **kw):
        self.calls.append(("poisson", kw))

    def _resolve_instance_model(self, model):
        return model

    def _lower_spatial(self, spatial):
        return spatial

    def _validate_riemann_capability(self, model, spatial):
        return None

    def _add_equation(self, name, model, spatial=None, time=None):
        self.calls.append(("add", name, tuple(getattr(model, "elliptic_field_names", []))))

    def _set_state(self, name, state):
        self.calls.append(("initial", name, state))

    def _install_aux(self, field_name, field):
        self.calls.append(("aux", field_name, field))

    def _install_params(self, resolved_models, params, reject_unknown=True):
        return set()

    def _install_problem_so(self, so_path):
        self.calls.append(("program", so_path))

    def _install_problem_params(self, compiled, params):
        self.calls.append(("program_params", dict(params)))

    def _install_cadence(self, cadence):
        self.calls.append(("cadence", cadence))


def _poisson_calls(sim):
    return [call[1] for call in sim.calls if call[0] == "poisson"]


def test_compiled_model_carries_elliptic_field_names():
    cm = _compiled_model(fields=("psi", "chi"))
    _check(cm.elliptic_field_names == ["psi", "chi"],
           "CompiledModel keeps the declared names")
    _check(_compiled_model().elliptic_field_names == [],
           "no declared field -> empty list")
    print("ok test_compiled_model_carries_elliptic_field_names")


def test_install_routes_default_poisson_field():
    sim = _RecordingSystem()
    sim.install(None, solvers={"phi": GeometricMG()})
    calls = _poisson_calls(sim)
    _check(calls and calls[0]["solver"] == "geometric_mg",
           "default field routes through sim.install")
    print("ok test_install_routes_default_poisson_field")


def test_install_routes_declared_named_field_from_compiled_handle():
    sim = _RecordingSystem()
    compiled = _CompiledProblem(_compiled_model(fields=("psi",)))
    sim.install(compiled, solvers={"psi": GeometricMG()})
    calls = _poisson_calls(sim)
    _check(calls and calls[0]["solver"] == "geometric_mg",
           "compiled handle declared field routes through sim.install")
    _check(("program", compiled.so_path) in sim.calls,
           "compiled Program handle is installed")
    print("ok test_install_routes_declared_named_field_from_compiled_handle")


def test_install_routes_declared_named_field_from_instance_model():
    sim = _RecordingSystem()
    model = _compiled_model(fields=("theta",))
    sim.install(
        None,
        instances={"plasma": {"model": model, "initial": [1.0]}},
        solvers={"theta": GeometricMG()},
    )
    calls = _poisson_calls(sim)
    _check(calls and calls[0]["solver"] == "geometric_mg",
           "instance model declared field routes through sim.install")
    _check(("add", "plasma", ("theta",)) in sim.calls,
           "instance model is bound through sim.install")
    _check(("initial", "plasma", [1.0]) in sim.calls,
           "initial state is routed by instance name")
    print("ok test_install_routes_declared_named_field_from_instance_model")


def test_install_rejects_undeclared_field():
    sim = _RecordingSystem()
    try:
        sim.install(
            None,
            instances={"plasma": {"model": _compiled_model(fields=("psi",))}},
            solvers={"psii": GeometricMG()},
        )
        raise AssertionError("an undeclared field name must raise")
    except ValueError as exc:
        _check("psii" in str(exc), "the reject names the offending field")
        _check("psi" in str(exc), "the reject names the declared set")
        _check("elliptic_field" in str(exc), "the reject points at m.elliptic_field")
    _check(not sim.calls, "no runtime mutation on a rejected field")
    print("ok test_install_rejects_undeclared_field")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
