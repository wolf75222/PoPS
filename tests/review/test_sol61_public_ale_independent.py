"""Independent source/host probes; never installs or runs a native PoPS artifact."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pops
import pytest
from pops.analytic import coordinate, time
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_moving import _source, emit_moving_helpers, moving_updates
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt, MovingFieldProjection


def declared(*, components=("q",), velocity=.3, source_weights=(1, 0),
             face_weights=(Fraction(1, 2), Fraction(1, 2)), initial_shift=0,
             selected_flux=None):
    frame = CartesianDomain("physical_domain", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("independent_body", frame=frame)
    state = model.state("parcel", components=components)
    axis = frame.axes[0]
    physical = model.flux("signed_original", frame=frame, state=state,
        components={axis: tuple(velocity * item for item in state)},
        waves={axis: (velocity,) * len(components)})
    offsets = {"q": 1, "a": 1, "b": 2, "c": 3}
    source_body = tuple(offsets[name] + .2 * item
                        for name, item in zip(components, state, strict=True))
    source = model.source("original_once", on=state, value=source_body)
    equation = model.rate("balance", equation=ddt(state) == -div(physical) + source)
    physical_rate, source_rate = equation.select(physical), equation.select(source)
    case = pops.Case("independent_ale")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(physical_rate, FiniteVolume(flux=physical,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=selected_flux or riemann.Rusanov()))
    plan.rates.add(source_rate, StateStorage())
    case.numerics(plan, block=block)
    program = pops.Program("reynolds")
    temporal = program.state(block[state])
    x = coordinate(frame, axis)
    evolution = GeometryEvolution((x + initial_shift + .7 * time(program.clock),))
    geometry = program.geometry_state(temporal.n, evolution=evolution)
    candidate = program.reynolds_update(geometry, physical_rate=physical_rate(temporal.n),
        source_rate=source_rate(temporal.n),
        projection=MovingFieldProjection(face_weights, source_weights),
        geometry_tolerance=0., at=temporal.next.point)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(8,),
        periodic=PeriodicAxes(frame.axes))), evolution=evolution)
    return case, layout


def resolved(**kwargs):
    case, layout = declared(**kwargs)
    result = pops.resolve(pops.validate(case), layout=layout)
    return result, ProgramModelGraph.from_resolved_blocks(result.blocks)


@pytest.mark.parametrize("weights", [(1, 0), (0, 1), (Fraction(1, 3), Fraction(2, 3))])
def test_noncentral_face_policy_refuses_at_real_public_prepare(weights):
    with pytest.raises(NotImplementedError, match="requires face_weights"):
        resolved(face_weights=weights)
    # The generic descriptor still represents other traces; the installed
    # Rusanov connector refuses its own missing realization.
    assert MovingFieldProjection(weights, (1, 0)).face_weights == weights


def test_unrealized_initial_projection_and_flux_refuse_before_emission():
    with pytest.raises(NotImplementedError, match="initial coordinate map"):
        resolved(initial_shift=.125)
    with pytest.raises(NotImplementedError, match="FaceContext provider"):
        resolved(selected_flux=riemann.HLL())


@pytest.mark.parametrize("components", [("q",), ("a", "b", "c"), ("c", "a", "b")])
@pytest.mark.parametrize("velocity", [.3, -.3])
@pytest.mark.parametrize("weights", [(1, 0), (0, 1), (Fraction(1, 2), Fraction(1, 2)), (2, -1)])
def test_actual_emitted_relative_model_and_original_source_on_host(
        tmp_path, components, velocity, weights):
    compiler = shutil.which("clang++")
    assert compiler, "the independent host probe requires clang++; no skip or JIT fallback"
    result, authority = resolved(components=components, velocity=velocity, source_weights=weights)
    emitted = emit_cpp_program(result.time, model_graph=authority)
    assert "ctx.commit_moving_interval(" in emitted
    assert "ctx.commit_many(" not in emitted and "ctx.rhs_into(" not in emitted
    helpers = emit_moving_helpers(result.time, authority)
    name = re.search(r"struct (PoPSMoving_\d+)Relative", helpers).group(1)
    update, = moving_updates(result.time)
    source_lines, source_values, invalid = _source(update, authority)
    actual_measure, = re.findall(r"const auto measure=[^;]+;", emitted)
    for component, expression in enumerate(source_values):
        assert "result[%d]=time.duration*measure*(%s);" % (component, expression) in emitted
    count = len(components)
    source_function = "\n".join([
        "std::array<double,%d> original(const std::array<double,%d>& u) {" % (count, count),
        "const pops::RuntimeParams params{};", *source_lines,
        'if (%s) throw std::runtime_error("nonfinite source");' % invalid,
        "return {%s}; }" % ",".join(source_values)])
    # Only math/type scaffolding is mocked. Both complete physical model and
    # relative override below are the actual resolved emitter's bytes.
    harness = """
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#define POPS_HD
namespace Kokkos { using std::max; using std::abs; using std::isfinite; }
namespace pops { using Real=double; template<int N> using StateVec=std::array<double,N>;
using RuntimeParams=std::array<double,1>;
namespace nd { enum class StateConversionStatus { Success, NonFiniteState };
template<class S> struct StateConversion { S value; StateConversionStatus status; }; } }
""" + helpers + source_function + "\nint main() {\n"
    harness += f"{name}Relative model; model.mesh_speed=.7;\n"
    harness += "pops::StateVec<%d> left{}, right{};\n" % count
    harness += "for(int c=0;c<%d;++c) { left[c]=c+1; right[c]=4*(c+1); }\n" % count
    harness += "const auto fl=model.flux<0>(left), fr=model.flux<0>(right);\n"
    harness += "const auto alpha=std::max(model.max_wave_speed<0>(left),model.max_wave_speed<0>(right));\n"
    harness += "const auto src=original(left);\n"
    harness += "const std::array<double,%d> offsets{%s};\n" % (
        count, ",".join(str({"q": 1, "a": 1, "b": 2, "c": 3}[name]) for name in components))
    harness += "for(int c=0;c<%d;++c) {\n" % count
    harness += "const double relative=.5*(fl[c]+fr[c])-.5*alpha*(right[c]-left[c]);\n"
    harness += "const double expected=(%s-.7)*right[c];\n" % velocity.hex()
    harness += 'if(std::abs(relative-expected)>1e-14 || std::abs(alpha-std::abs(%s-.7))>1e-14) throw std::runtime_error("relative oracle");\n' % velocity.hex()
    harness += 'if(std::abs(src[c]-(offsets[c]+.2*left[c]))>1e-14) throw std::runtime_error("original source/permutation");\n'
    harness += "struct { double previous_measure=.125, measure=.15; } cell;\n"
    harness += actual_measure + "\n"
    harness += "const double integrated=.01*measure*src[c];\n"
    harness += 'if(std::abs(integrated-.01*(%s*.125+%s*.15)*(offsets[c]+.2*left[c]))>1e-14) throw std::runtime_error("source measure"); }\n' % tuple(float(x).hex() for x in weights)
    harness += f'if(std::abs({name}Law{{}}(.25,.1)-.32)>1e-14) throw std::runtime_error("endpoint law");\n'
    harness += "return 0; }\n"
    path = tmp_path / "actual_emitted_model.cpp"
    path.write_text(harness)
    binary = tmp_path / "actual_emitted_model"
    compiled = subprocess.run([compiler, "-std=c++20", "-O0", str(path), "-o", str(binary)],
        capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stderr


def observed():
    from tests.python.unit.output.test_post_commit_observers import _frame
    frame = _frame(cell_shape=(3,), component_names=("q",))
    nodes = np.array([[.05], [.3], [.675], [1.05]])
    geometry = replace(frame.snapshot.geometries[0],
        coordinate_system="pops://coordinates/moving-cartesian-1d@1",
        cell_measure="pops://cell-measures/endpoint-length@1",
        node_coordinates=nodes, cell_volumes=np.diff(nodes[:, 0]))
    return replace(frame, snapshot=replace(frame.snapshot, geometries=(geometry,))), nodes


def test_physical_coordinates_owned_detached_archived_and_projected():
    from pops.output.observers import detach_observer_frame
    from pops.output._observer_archive import encode_observer_frame, decode_observer_frame
    from pops.output._writers.common import piece_payload
    from pops.output._writers.paraview import _physical_point_coordinates
    frame, borrowed = observed()
    expected = borrowed.copy()
    borrowed[:] = 90.
    detached = detach_observer_frame(frame)
    assert not np.shares_memory(detached.snapshot.geometries[0].node_coordinates,
                                frame.snapshot.geometries[0].node_coordinates)
    archived = decode_observer_frame(encode_observer_frame(detached))
    for item in (frame, detached, archived):
        geometry = item.snapshot.geometries[0]
        np.testing.assert_array_equal(geometry.node_coordinates, expected)
        np.testing.assert_array_equal(geometry.cell_volumes, np.diff(expected[:, 0]))
        assert not geometry.node_coordinates.flags.writeable
        points = _physical_point_coordinates(geometry, (np.arange(4),))
        np.testing.assert_array_equal(points[:, 0], expected[:, 0])
    arrays, datasets, _ = piece_payload(archived.snapshot, archived.request)
    declaration = next(iter(datasets["geometries"].values()))
    np.testing.assert_array_equal(arrays[declaration["node_coordinates"]], expected)


@pytest.mark.parametrize("bad", [None, np.array([[0.], [1.], [2.]]),
    np.array([[0.], [.1], [.1], [1.]]), np.array([[0.], [.3], [.2], [1.]]),
    np.array([[0.], [.3], [np.nan], [1.]]), np.array([[0.], [.3], [.6], [np.inf]])])
def test_observer_refuses_malformed_physical_nodes(bad):
    frame, _ = observed()
    with pytest.raises(ValueError):
        replace(frame.snapshot.geometries[0], node_coordinates=bad)


def test_observer_refuses_rounded_or_reference_measures():
    frame, _ = observed()
    geometry = frame.snapshot.geometries[0]
    altered = geometry.cell_volumes.copy()
    altered[0] = np.nextafter(altered[0], np.inf)
    with pytest.raises(ValueError, match="endpoint differences"):
        replace(geometry, cell_volumes=altered)


@pytest.mark.parametrize("metadata", [
    {"coordinate_system": "pops://coordinates/moving-cartesian-1d@1"},
    {"cell_measure": "pops://cell-measures/endpoint-length@1"},
])
def test_versioned_one_dimensional_geometry_metadata_refuses_rank_two(metadata):
    from tests.python.unit.output.test_post_commit_observers import _frame
    original = _frame(cell_shape=(2, 2)).snapshot.geometries[0]
    with pytest.raises(ValueError):
        replace(original, **metadata, node_coordinates=np.zeros((3, 3, 2)))


def test_legacy_ledger_writer_source_is_exactly_unchanged():
    root = Path(__file__).resolve().parents[2]
    path = "include/pops/runtime/program/accepted_exchange.hpp"
    previous = subprocess.run(["git", "show", "2a741eff:" + path], cwd=root,
        capture_output=True, text=True, check=True).stdout
    current = (root / path).read_text()
    begin = "  std::vector<std::uint8_t> checkpoint(bool force_extended=false) const {"
    end = "  static AcceptedExchangeLedger from_checkpoint("
    assert previous.split(begin, 1)[1].split(end, 1)[0] == current.split(begin, 1)[1].split(end, 1)[0]
