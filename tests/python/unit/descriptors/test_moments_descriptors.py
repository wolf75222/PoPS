#!/usr/bin/env python3
"""Spec 5 sec.6 (ADC-498): the route-choosing pops.moments objects are typed descriptors.

The moment toolkit exposes a mix of objects. A handful CHOOSE a math algorithm -- the
wave-speed strategy (:class:`ExactSpeeds`), the realizability-floor strategy
(:class:`RealizabilityProjection`), the magnetic-source binding
(:class:`MagneticMomentSource`) and the closure variant (:class:`HyQMOM15Closure`). Spec 5
sec.6 requires every such route chooser to be an inert, inspectable
:class:`pops.descriptors.Descriptor` that declares its options / capabilities and answers
``available(context)`` with an explainable status.

The rest only construct or hold structure (``MomentModel`` / ``MomentBasis`` / the binomial
transforms / ``MomentOrdering`` / ``VlasovSources`` / ``MomentHierarchy``); they make no
algorithm choice, so they stay lightweight handles and are NOT descriptors. ``MomentOrdering``
in particular has a single forced layout, so it is a handle, not a route chooser.

Pure Python: this imports only the public descriptor surface; nothing here builds or runs a
native moment model.
"""
import sys

import pytest

from pops import moments  # noqa: E402
from pops.descriptors import Availability, Descriptor, DescriptorProtocol  # noqa: E402
from pops.params import ConstParam, RuntimeParam  # noqa: E402
from pops.physics import Model  # noqa: E402


def test_exact_speeds_descriptor_contract():
    speeds = moments.ExactSpeeds(moments.ExactSpeeds.ROE_DISSIPATION)
    assert isinstance(speeds, (Descriptor, DescriptorProtocol))
    assert speeds.name == "ExactSpeeds"
    assert speeds.category == "wave_speed"
    assert speeds.options()["kind"] == moments.ExactSpeeds.ROE_DISSIPATION
    caps = speeds.capabilities().to_dict()
    assert caps["exact_speeds"] is True and caps["roe"] is True
    assert speeds.available().ok
    assert speeds.validate() is True
    # The BOUNDED strategy turns the engine flags off (still a valid, available route).
    bounded = moments.ExactSpeeds(moments.ExactSpeeds.BOUNDED)
    assert bounded.capabilities().to_dict() == {"exact_speeds": False, "roe": False}
    # from_flags round-trips to the same descriptor kind.
    assert moments.ExactSpeeds.from_flags(True, True).options()["kind"] == \
        moments.ExactSpeeds.ROE_DISSIPATION
    with pytest.raises(ValueError):
        moments.ExactSpeeds("nope")


def test_realizability_projection_descriptor_contract():
    proj = moments.RealizabilityProjection(eps_m00=1e-10, eps_cov=1e-9, robust=False)
    assert isinstance(proj, (Descriptor, DescriptorProtocol))
    assert proj.name == "RealizabilityProjection"
    assert proj.category == "realizability"
    opts = proj.options()
    assert opts["eps_m00"] == 1e-10 and opts["eps_cov"] == 1e-9 and opts["robust"] is False
    assert proj.capabilities().to_dict()["guard_level"] == "bare"
    assert moments.RealizabilityProjection().capabilities().to_dict()["guard_level"] == "smooth"
    assert proj.validate() is True
    # The .none() preset is the bare guard-free route.
    assert moments.RealizabilityProjection.none().options()["robust"] is False


def test_magnetic_moment_source_descriptor_contract():
    src = moments.MagneticMomentSource(q_over_m="my_q", b_field="my_b")
    assert isinstance(src, (Descriptor, DescriptorProtocol))
    assert src.name == "MagneticMomentSource"
    assert src.category == "moment_source"
    assert src.options() == {"q_over_m": "my_q", "b_field": "my_b"}
    assert src.capabilities().to_dict()["provides"] == "magnetic_lorentz"
    assert src.validate() is True
    # The builder side stays: as_sources() returns a (m, M) -> list callable.
    assert callable(src.as_sources(2.0))


def test_hyqmom15_closure_descriptor_contract():
    closure = moments.HyQMOM15Closure()
    assert isinstance(closure, (Descriptor, DescriptorProtocol))
    assert closure.name == "HyQMOM15Closure"
    assert closure.category == "closure"
    assert closure.order == 4
    opts = closure.options()
    assert opts["order"] == 4
    assert opts["local_operator"]["kind"] == "local_moment_closure"
    assert closure.capabilities().to_dict()["provides"] == "order_4_standardized_moments"
    assert closure.validate() is True
    # The descriptor is still the closure callable (Spec 5 sec.6 does not change its role).
    standardized = {"S11": 0.1, "S20": 1.0, "S02": 1.0, "S30": 0.0, "S21": 0.0,
                    "S12": 0.0, "S03": 0.0, "S40": 3.0, "S31": 0.0, "S22": 1.0,
                    "S13": 0.0, "S04": 3.0}
    out = closure(standardized)
    assert set(out) == {"S%d%d" % (p, 5 - p) for p in range(6)}
    # User physics goes through @closure(4), never a reserved string variant.
    with pytest.raises(TypeError):
        moments.HyQMOM15Closure(variant="custom")


def test_route_choosers_available_is_explainable():
    # Every moments descriptor answers available() with an Availability, never a bare bool.
    for descriptor in (moments.ExactSpeeds(), moments.RealizabilityProjection(),
                       moments.MagneticMomentSource(), moments.HyQMOM15Closure()):
        status = descriptor.available()
        assert isinstance(status, Availability)
        assert not isinstance(status, bool)
        assert status.ok is True


def test_handles_are_not_descriptors():
    # The builders / handles construct or hold; they choose no route, so they are not
    # descriptors. MomentOrdering is a single forced layout -> a handle, not a route chooser.
    handles = (
        moments.MomentOrdering(),
        moments.MomentBasis(order=2),
        moments.CenteredTransform(order=2),
        moments.StandardizedTransform(order=2),
        moments.CartesianVelocityMoments(order=2),
        moments.CartesianVelocityMoments(order=2).hierarchy(),
        moments.VlasovSources,
    )
    for handle in handles:
        name = getattr(handle, "__name__", type(handle).__name__)
        assert not isinstance(handle, Descriptor), (
            "%s is a builder/handle and must not be a Descriptor" % name)


def test_moment_model_has_no_transport_noop_surface():
    specification = moments.CartesianVelocityMoments(order=2)
    assert not hasattr(specification, "add_transport")


def test_hierarchy_snapshot_exposes_inspectable_descriptors():
    # The MomentHierarchy snapshot carries the speeds / projection descriptors; they remain
    # inspectable route choosers even when reached through the snapshot.
    model = (moments.CartesianVelocityMoments(order=2)
             .add_numerics(roe=True)
             .set_realizability(moments.RealizabilityProjection.none()))
    snapshot = model.hierarchy()
    assert isinstance(snapshot.speeds, Descriptor)
    assert snapshot.speeds.options()["kind"] == moments.ExactSpeeds.ROE_DISSIPATION
    assert isinstance(snapshot.projection, Descriptor)
    assert snapshot.projection.capabilities().to_dict()["guard_level"] == "bare"


def test_moment_coefficients_preserve_typed_storage_and_numeric_values():
    eps = RuntimeParam("eps_runtime", default=2.5)
    q_over_m = RuntimeParam("q_over_m_runtime", default=-3.0)
    specification = (moments.CartesianVelocityMoments(order=2)
                     .add_poisson_coupling(eps=eps)
                     .add_vlasov_electric_source("grad_x", "grad_y", q_over_m)
                     .add_magnetic_source(-0.25))
    first = specification.build("typed_moment_coefficients_a").module.params()
    second = specification.build("typed_moment_coefficients_b").module.params()

    assert eps.is_owned is False and q_over_m.is_owned is False
    for parameters in (first, second):
        assert parameters["eps_runtime"] == eps
        assert parameters["q_over_m_runtime"] == q_over_m
        assert parameters["eps_runtime"] is not eps
        assert parameters["q_over_m_runtime"] is not q_over_m
        assert isinstance(parameters["omega_c"], ConstParam)
        assert parameters["omega_c"].value == -0.25
    assert first["eps_runtime"] is not second["eps_runtime"]
    assert first["eps_runtime"].owner_identity != second["eps_runtime"].owner_identity


def test_moment_coefficients_refuse_implicit_string_and_bool_coercions():
    with pytest.raises(TypeError, match="q_over_m"):
        moments.CartesianVelocityMoments(2).add_vlasov_electric_source(
            "grad_x", "grad_y", "q_over_m")
    with pytest.raises(TypeError, match="eps"):
        moments.CartesianVelocityMoments(2).add_poisson_coupling(eps=True)
    with pytest.raises(TypeError, match="robust"):
        moments.CartesianVelocityMoments(2, robust=1)


def test_moment_coefficients_cannot_clone_another_registry_owner():
    already_owned = RuntimeParam("owned_eps", default=1.0)
    Model("coefficient_owner").param(already_owned)
    with pytest.raises(ValueError, match=r"already owned.*shared owner or tie"):
        moments.CartesianVelocityMoments(2).add_poisson_coupling(eps=already_owned)

    claimed_after_recording = RuntimeParam("late_owned_eps", default=1.0)
    specification = moments.CartesianVelocityMoments(2).add_poisson_coupling(
        eps=claimed_after_recording)
    Model("late_coefficient_owner").param(claimed_after_recording)
    with pytest.raises(ValueError, match=r"already owned.*shared owner or tie"):
        specification.build("late_owned_moment_model")


# ---------------------------------------------------------------------------------------------
# ADC-543: the generic construction vocabulary is additive and the four issue contracts hold.
# ---------------------------------------------------------------------------------------------
def test_realizability_projection_and_realizable_set_have_unique_names():
    assert not hasattr(moments, "MomentProjection")
    cone = moments.RealizableSet(4)
    assert isinstance(cone, (Descriptor, DescriptorProtocol))
    assert cone.name == "RealizableSet"
    assert cone.category == "realizability_set"
    assert cone.options() == {"order": 4}
    caps = cone.capabilities()
    assert caps.to_dict()["constraints"] == "m00_positive,cov_psd,schur"
    for descriptor in (moments.RealizabilityProjection(), cone):
        status = descriptor.available()
        assert isinstance(status, Availability) and not isinstance(status, bool)
        assert status.ok is True
    assert cone.validate() is True
    with pytest.raises(ValueError):
        moments.RealizableSet(1)


def test_realizable_set_descriptor_protocol_round_trip():
    # DescriptorProtocol conformity: options / available / validate / lower / freeze round-trip
    # (the inert typed-descriptor contract the base guarantees).
    cone = moments.RealizableSet(4)
    assert isinstance(cone, DescriptorProtocol)
    lowered = cone.lower()
    assert lowered.name == "RealizableSet" and lowered.category == "realizability_set"
    assert lowered.options() == {"order": 4} if callable(getattr(lowered, "options", None)) \
        else lowered.options == {"order": 4}
    inspected = cone.inspect()
    assert inspected["category"] == "realizability_set"
    # freeze lifecycle: a post-freeze mutation raises (inherited from Descriptor).
    cone.freeze()
    with pytest.raises(RuntimeError):
        cone.order = 2


def test_wave_speed_capability_has_one_canonical_home():
    # ExactSpeeds is the MOMENT wave-speed axis (how exact speeds are computed): a typed
    # CapabilitySet, kept as the moments chooser.
    exact = moments.ExactSpeeds()
    assert isinstance(exact.capabilities().to_dict(), dict)
    assert exact.capabilities().to_dict()["exact_speeds"] is True
    from pops.numerics.riemann.waves import WaveSpeedProvider
    assert not hasattr(moments, "WaveSpeedProvider")
    provider = WaveSpeedProvider("jacobian")
    assert provider.capabilities().supports("signed_pair") is True
    assert WaveSpeedProvider("max_wave_speed").capabilities().supports("signed_pair") is False


def test_hyqmom15_model_is_inspectable_and_runtime_free():
    # The provided HyQMOM15 model builds to an authoring physics object whose Module lists the
    # typed transport + source operators and the 15 conservative names, with no runtime leakage
    # (mirrors the ADC-566 lib boundary: models lower runtime-free to pops.model.Module).
    from pops.lib.models.moments import HyQMOM15
    model = HyQMOM15.vlasov_poisson_magnetic(order=4)
    module = getattr(model, "module", model)
    assert hasattr(module, "operator_registry")
    op_names = module.operator_registry().names()
    assert {"flux_default", "electric", "magnetic_rotation", "transport"} <= set(op_names)
    # 15 conservative components (the order-4 hierarchy), canonical names.
    components = model._dsl._m.state_space().components
    assert len(components) == 15 and components[0] == "M00"
    # No compiled / runtime leakage on the authoring object.
    for runtime_attr in ("so_path", "abi_key"):
        assert not getattr(model, runtime_attr, None)


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this module
# so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
