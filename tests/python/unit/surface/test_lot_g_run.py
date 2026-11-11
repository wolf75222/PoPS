"""Lot G of N15 (ROM-98, OD31): the run constructs of the surface, without the native runtime.

``pops.record``, ``pops.every``, ``pops.Output``, ``pops.Case(outputs=[...])``, ``pops.compile``,
``pops.OpenMP``, ``pops.CUDA``, ``pops.Resources`` and ``pops.bind`` check their arguments and
refuse each option of a later lot with ``NotYetElaborated`` and that lot (2.6.2, 2.6.6, CC14);
an Output is lowered to the Output of N8 that the elaborated Advance carries; the arrays of a bind
and of a result are in the order of the support's axes (Q3); ``pops.lib.output.npz`` writes the
series of an Output in one archive and refuses a file that another run wrote. The compiled run
itself is checked by ``tests/python/integration/runtime/test_surface_lot_g_run_native.py``.
"""
import types

import numpy
import pytest

import pops.surface as surface
from pops.lib.numerics.schemes import two_point, upwind
from pops.lib.output import Npz, OutputExists, hdf5, npz
from pops.lib.time.methods import ssprk2
from pops.surface import _run
from pops.surface.math import ddt, div, grad
from pops.time._program import outputs as n8_outputs


def _model():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    advance = surface.Advance(ssprk2(rate=model), dt=1e-3, attempts=2)
    program = surface.Program(advance=advance, initial={u: u0})
    return types.SimpleNamespace(domain=domain, model=model, u=u, nu=nu, u0=u0,
                                 numerics=numerics, program=program)


def _case(parts, outputs=None, cells=16):
    return surface.Case(model=parts.model, numerics=parts.numerics, program=parts.program,
                        layout={parts.domain: surface.Uniform(cells=cells)}, outputs=outputs)


@pytest.fixture
def parts():
    return _model()


# --- record, every, Output ----------------------------------------------------------------------

def test_a_record_reads_its_fields_by_name(parts):
    value = surface.record(u=parts.u, n=3)
    assert value.u is parts.u and value.n == 3
    assert [name for name, _ in value.items()] == ["u", "n"]
    with pytest.raises(AttributeError, match="has the fields u, n"):
        value.w
    with pytest.raises(TypeError, match="at least one field"):
        surface.record()


def test_every_takes_a_time_or_a_number_of_steps():
    assert surface.every(time=0.5).time == 0.5
    assert surface.every(steps=4).steps == 4
    with pytest.raises(TypeError, match="one of them"):
        surface.every(time=0.5, steps=4)
    with pytest.raises(ValueError, match="positive finite"):
        surface.every(time=0.0)
    with pytest.raises(ValueError, match="k >= 1"):
        surface.every(steps=0)


def test_an_output_publishes_quantities_on_a_schedule(parts):
    output = surface.Output(surface.record(u=parts.u), when=[0.5, 0.25, 0.5])
    assert output.times == (0.25, 0.5)
    assert surface.Output(surface.record(u=parts.u), when=0.5).times == (0.5,)
    with pytest.raises(surface.AmbiguousElaboration, match="has no schedule"):
        surface.Output(surface.record(u=parts.u))
    with pytest.raises(TypeError, match="a tuple is a point"):
        surface.Output(surface.record(u=parts.u), when=(0.5,))
    with pytest.raises(TypeError, match="a number is the metadata= of a writer"):
        surface.Output(surface.record(u=parts.u, n=3), when=0.5)
    with pytest.raises(TypeError, match="a record of what the run publishes"):
        surface.Output(parts.u, when=0.5)
    with pytest.raises(TypeError, match="takes a writer of pops.lib.output"):
        surface.Output(surface.record(u=parts.u), when=0.5, write="u.npz")


def test_an_output_refuses_the_options_of_a_later_lot(parts):
    with pytest.raises(surface.NotYetElaborated, match="predicates of the when bodies"):
        surface.Output(surface.record(u=parts.u), when=lambda state: True)
    with pytest.raises(surface.NotYetElaborated, match="definitions of R16"):
        surface.Output(surface.record(nu=parts.nu), when=0.5)
    with pytest.raises(surface.NotYetElaborated, match="a list of schedules"):
        surface.Output(surface.record(u=parts.u), when=[0.5, surface.every(steps=2)])
    with pytest.raises(surface.SymbolNotInSurface):
        surface.Output(surface.record(u=parts.u), at=(0.5,))


def test_an_output_is_lowered_to_the_output_of_n8(parts):
    at_times = surface.Output(surface.record(u=parts.u), when=[0.25, 0.5])
    by_steps = surface.Output(surface.record(u=parts.u), when=surface.every(steps=4))
    by_time = surface.Output(surface.record(u=parts.u), when=surface.every(time=0.125))
    case = _case(parts, outputs=[at_times, by_steps, by_time])
    lowered = case.elaborated.outputs
    assert len(lowered) == 3 and all(type(item) is n8_outputs.Output for item in lowered)
    assert case.outputs == (at_times, by_steps, by_time)
    # the start of the first run: a list of times keeps it when it lists that time
    assert not at_times.keeps(0.0, True, 0) and at_times.keeps(0.25, True, 1)
    assert by_steps.keeps(0.0, True, 0) and by_time.keeps(0.0, True, 0)
    # the start of a later run is the end of the run before it
    assert not at_times.keeps(0.25, False, 0) and not by_steps.keeps(0.25, False, 0)
    assert at_times.keeps(0.5, False, 1) and by_steps.keeps(0.5, False, 1)


def test_a_case_takes_a_list_of_distinct_outputs(parts):
    output = surface.Output(surface.record(u=parts.u), when=0.5)
    assert _case(parts).outputs == ()
    with pytest.raises(TypeError, match="outputs=\\[output\\]"):
        _case(parts, outputs=output)
    with pytest.raises(TypeError, match="a list of pops.Output"):
        _case(parts, outputs=[parts.u])
    with pytest.raises(ValueError, match="each output once"):
        _case(parts, outputs=[output, output])


# --- compile, the targets, the resources ---------------------------------------------------------

def test_compile_names_its_target(parts):
    case = _case(parts)
    with pytest.raises(surface.AmbiguousElaboration, match="target=pops.OpenMP\\(\\)"):
        surface.compile(case)
    with pytest.raises(TypeError, match="takes a pops.Case"):
        surface.compile(parts.model, target=surface.OpenMP())
    with pytest.raises(TypeError, match="takes pops.OpenMP\\(\\)"):
        surface.compile(case, target="openmp")
    with pytest.raises(surface.NotYetElaborated, match="lowering classes of N7"):
        surface.compile(case, target=surface.OpenMP(), schedules={})


def test_the_targets_and_resources_of_a_later_lot_are_refused():
    with pytest.raises(surface.NotYetElaborated, match="GPU run of Q03a"):
        surface.CUDA()
    with pytest.raises(surface.NotYetElaborated, match="distributed runs of N10"):
        surface.OpenMP(mpi=True)
    with pytest.raises(surface.NotYetElaborated, match="distributed runs of N10"):
        surface.Resources(communicator=object())
    with pytest.raises(surface.NotYetElaborated, match="restart of G07"):
        surface.Resources(history=2)
    with pytest.raises(ValueError, match="n >= 1"):
        surface.Resources(threads=0)
    assert surface.Resources().threads == 1


# --- bind, before the native runtime -------------------------------------------------------------

def _artifact(case, subjects=("u",)):
    """An artifact whose compiled plan names the initial bindings only: what bind reads before
    it calls the native bind."""
    bindings = tuple(types.SimpleNamespace(subject=subject) for subject in subjects)
    compiled = types.SimpleNamespace(plan=types.SimpleNamespace(
        initial_condition_plan=types.SimpleNamespace(bindings=bindings)))
    return _run.Artifact(case, compiled, surface.OpenMP())


def test_bind_is_keyed_by_handles(parts):
    artifact = _artifact(_case(parts))
    cells = numpy.zeros(16)
    with pytest.raises(TypeError, match="what pops.compile returns"):
        surface.bind(_case(parts))
    with pytest.raises(TypeError, match="not by the string 'nu'"):
        surface.bind(artifact, params={"nu": 0.1}, data={parts.u0: cells})
    with pytest.raises(TypeError, match="not by the string 'u0'"):
        surface.bind(artifact, data={"u0": cells})
    with pytest.raises(TypeError, match="keyed by the data of the case's model"):
        surface.bind(artifact, data={parts.u: cells})
    with pytest.raises(TypeError, match="takes a number"):
        surface.bind(artifact, params={parts.nu: "0.1"}, data={parts.u0: cells})
    with pytest.raises(TypeError, match="pops.Resources\\(threads=n\\)"):
        surface.bind(artifact, data={parts.u0: cells}, resources={"threads": 2})


def test_bind_names_what_is_missing_and_what_is_later(parts):
    artifact = _artifact(_case(parts))
    with pytest.raises(surface.AmbiguousElaboration, match="data=\\{u0: array\\}"):
        surface.bind(artifact)
    with pytest.raises(ValueError, match="the cells of its support are \\(16,\\)"):
        surface.bind(artifact, data={parts.u0: numpy.zeros(15)})
    with pytest.raises(surface.NotYetElaborated, match="restart of G07"):
        surface.bind(artifact, data={parts.u0: numpy.zeros(16)}, restart="run.h5")


def test_the_layout_of_an_array_is_the_order_of_the_axes():
    cells = (4, 3)
    value = numpy.arange(12.0).reshape(cells)       # value[i, j] is the cell (x_i, y_j)
    stored = _run.to_layout(value, cells, "data")
    assert stored.shape == (1, 3, 4)                # b6: the component, then y, then x
    assert stored[0, 2, 1] == value[1, 2]
    assert numpy.array_equal(_run.from_layout(stored, cells), value)
    assert numpy.array_equal(_run.from_layout(stored.ravel(), cells), value)
    with pytest.raises(ValueError, match="in the order of its axes"):
        _run.to_layout(value.T, cells, "data")


# --- the writer npz -------------------------------------------------------------------------------

def _series(*times):
    return tuple(_run.Publication(time, step, {"u": numpy.full((2, 3), time)})
                 for step, time in enumerate(times))


def test_npz_checks_its_arguments(parts):
    with pytest.raises(TypeError, match="takes a file path"):
        npz(3)
    with pytest.raises(ValueError, match="relative to the directory= of pops.bind"):
        npz("/tmp/u.npz")
    with pytest.raises(TypeError, match="a pops.record of numbers"):
        npz("u.npz", metadata={"nx": 16})
    with pytest.raises(TypeError, match="metadata are numbers"):
        npz("u.npz", metadata=surface.record(name="t02"))
    with pytest.raises(ValueError, match="name the metadata otherwise"):
        npz("u.npz", metadata=surface.record(time=1.0))
    with pytest.raises(TypeError, match="True or False"):
        npz("u.npz", overwrite=1)
    with pytest.raises(surface.NotYetElaborated, match="writers of pops.lib.output"):
        hdf5("u.h5")
    assert isinstance(npz("u.npz"), Npz)


def test_npz_writes_the_series_of_an_output(tmp_path, parts):
    writer = npz("out/u.npz", metadata=surface.record(nx=3, length=1.0))
    target = writer.write_series(str(tmp_path), surface.record(u=parts.u), _series(0.0, 0.5),
                                 first=True)
    assert target == str(tmp_path / "out" / "u.npz")
    with numpy.load(target) as archive:
        assert sorted(archive.files) == ["length", "nx", "step", "time", "u"]
        assert archive["u"].shape == (2, 2, 3)
        assert archive["time"].tolist() == [0.0, 0.5]
        assert archive["step"].tolist() == [0, 1]
        assert archive["nx"] == 3 and archive["length"] == 1.0
    # the same simulation writes it again with the longer series; a new one is refused
    writer.write_series(str(tmp_path), surface.record(u=parts.u), _series(0.0, 0.5, 1.0),
                        first=False)
    with numpy.load(target) as archive:
        assert archive["time"].tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(OutputExists, match="overwrite=True"):
        writer.write_series(str(tmp_path), surface.record(u=parts.u), _series(0.0), first=True)
    npz("out/u.npz", overwrite=True).write_series(str(tmp_path), surface.record(u=parts.u),
                                                  _series(0.0), first=True)
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["u.npz"]


def test_npz_refuses_a_metadata_named_as_a_field_of_the_record(tmp_path, parts):
    """F2 of the review of lot G: a metadata u would replace the series u."""
    writer = npz("u.npz", metadata=surface.record(u=3.0))
    with pytest.raises(ValueError, match="u is both a field of the record and a metadata"):
        surface.Output(surface.record(u=parts.u), when=0.5, write=writer)
    with pytest.raises(ValueError, match="u is both a field of the record and a metadata"):
        writer.write_series(str(tmp_path), surface.record(u=parts.u), _series(0.0), first=True)
    assert not (tmp_path / "u.npz").exists()
    surface.Output(surface.record(u=parts.u), when=0.5,
                   write=npz("u.npz", metadata=surface.record(nx=3)))


def test_from_layout_takes_the_layout_of_b6_or_a_flat_array():
    """O1 of the review of lot G: an array in another order is refused, not reshaped."""
    cells = (4, 3)
    value = numpy.arange(12.0).reshape(cells)
    stored = _run.to_layout(value, cells, "data")
    with pytest.raises(ValueError, match="has the shape \\(1, 3, 4\\) or is flat"):
        _run.from_layout(numpy.transpose(stored[0])[numpy.newaxis, ...], cells)


def test_npz_refuses_a_field_named_as_the_series(tmp_path, parts):
    with pytest.raises(ValueError, match="name the field otherwise"):
        npz("u.npz").write_series(str(tmp_path), surface.record(time=parts.u), (), first=True)
