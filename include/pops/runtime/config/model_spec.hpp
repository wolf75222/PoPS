#pragma once

#include <pops/runtime/numerical_defaults.hpp>

#include <stdexcept>
#include <string>
#include <utility>

/// @file
/// @brief Flat specification of a model: chosen bricks plus their parameters.
///
/// Describes a PhysicalModel as a composition of generic bricks (transport, source,
/// elliptic right-hand side) and carries their parameters. No named scenario: the
/// application (adc_cases) is the one that names a composition of these bricks.
/// Flat authoring value with guarded fields so C++ and Python share the same lifecycle contract.

namespace pops {

namespace detail {

struct ModelSpecFreezeTransactionAccess;

/// Publicly readable authoring value whose writes are guarded by its owning ModelSpec.
///
/// Keeping the assignment spelling (``spec.gamma = ...``) avoids a noisy native builder API,
/// while the underlying value remains inaccessible: unlike a raw public field, no C++ caller can
/// bypass ``ModelSpec::freeze()`` accidentally.  The proxy is permanently bound to its owner's
/// lifecycle bit; ModelSpec implements copy/move explicitly so that binding never aliases another
/// instance.
template <class T>
class FrozenModelValue {
 public:
  using value_type = T;

  FrozenModelValue(bool& frozen, const char* name, T value)
      : frozen_(&frozen), name_(name), value_(std::move(value)) {}

  FrozenModelValue(const FrozenModelValue&) = delete;
  FrozenModelValue(FrozenModelValue&&) = delete;

  FrozenModelValue& operator=(const FrozenModelValue& other) { return *this = other.value_; }
  // Never move from another proxy: its owner may already be frozen, and draining that source
  // would mutate a sealed ModelSpec through an rvalue reference. Copy the value into this owner.
  FrozenModelValue& operator=(FrozenModelValue&& other) { return *this = other.value_; }

  FrozenModelValue& operator=(const T& value) {
    require_mutable();
    value_ = value;
    return *this;
  }

  FrozenModelValue& operator=(T&& value) {
    require_mutable();
    value_ = std::move(value);
    return *this;
  }

  [[nodiscard]] const T& get() const noexcept { return value_; }
  [[nodiscard]] operator const T&() const noexcept { return value_; }

  /// String-like completeness checks used by the native routing contract. Instantiated only for
  /// value types that provide ``empty()``.
  [[nodiscard]] bool empty() const { return value_.empty(); }

 private:
  void require_mutable() const {
    if (*frozen_)
      throw std::runtime_error(std::string("ModelSpec is frozen; cannot modify '") + name_ + "'");
  }

  bool* frozen_;
  const char* name_;
  T value_;
};

}  // namespace detail

/// Brick composition of a block plus parameters. The fields are only read by the relevant
/// brick (see dispatch_model in model_factory.hpp).
///
/// CONTRACT (ADC-290): a ModelSpec carries NO silent physics default. `transport` and `elliptic`
/// are UNSET by default (empty string) and MUST be chosen explicitly; an unset tag is rejected with
/// a clear message by validate_model_spec (model_factory.hpp) instead of silently selecting a model
/// (the old defaults `transport="compressible"` / `elliptic="charge"` made a default-constructed
/// ModelSpec mean Euler + Poisson-charge by accident). `source` keeps the only default, "none" --
/// the EXPLICIT, neutral "no source" choice, not a physics selection. The numeric parameters keep
/// their defaults: each is read only once its brick has been chosen by a tag, so it can never inject
/// physics on its own. Historical shortcuts live at the Python edge (pops.Model(...)), which always
/// sets the three tags. See docs/adr/ADR-0001-genericity-contracts.md.
struct ModelSpec {
 private:
  friend struct detail::ModelSpecFreezeTransactionAccess;

  bool frozen_ = false;

 public:
  detail::FrozenModelValue<std::string> transport;  ///< REQUIRED: exb/compressible/isothermal
  detail::FrozenModelValue<std::string> source;     ///< none/potential/gravity/magnetic/...
                                                    ///< | "magnetic" | "potential_magnetic"
  detail::FrozenModelValue<std::string> elliptic;   ///< REQUIRED: charge/background/gravity

  detail::FrozenModelValue<double> B0;            ///< ExBVelocity: magnetic field
  detail::FrozenModelValue<double> gamma;         ///< CompressibleFlux: adiabatic index
  detail::FrozenModelValue<double> cs2;           ///< IsothermalFlux: sound speed squared
  detail::FrozenModelValue<double> vacuum_floor;  ///< IsothermalFlux: quasi-vacuum density floor
      ///< (ADC-77). Set from pops.FluidState(vacuum_floor=...)
  ///< INDEPENDENT of the spatial positivity_floor (deliberately decoupled --
  ///< coupling them shifts the CFL dt of existing positivity_floor runs).
  ///< 0 = off (bit-identical)
  detail::FrozenModelValue<double> qom;        ///< PotentialForce / MagneticLorentzForce: q/m
  detail::FrozenModelValue<double> q;          ///< ChargeDensity: charge q
  detail::FrozenModelValue<double> alpha;      ///< BackgroundDensity coupling
  detail::FrozenModelValue<double> n0;         ///< neutralizing background
  detail::FrozenModelValue<double> sign;       ///< +1 gravity, -1 electrostatic
  detail::FrozenModelValue<double> four_pi_G;  ///< coupling intensity
  detail::FrozenModelValue<double> rho0;       ///< GravityCoupling background

  ModelSpec()
      : transport(frozen_, "transport", std::string{}),
        source(frozen_, "source", std::string{"none"}),
        elliptic(frozen_, "elliptic", std::string{}),
        B0(frozen_, "B0", static_cast<double>(kPhysicalDefaultB0)),
        gamma(frozen_, "gamma", static_cast<double>(kPhysicalDefaultGamma)),
        cs2(frozen_, "cs2", static_cast<double>(kPhysicalDefaultFluidStateCs2)),
        vacuum_floor(frozen_, "vacuum_floor", static_cast<double>(kPhysicalDefaultVacuumFloor)),
        qom(frozen_, "qom", static_cast<double>(kPhysicalDefaultQOverM)),
        q(frozen_, "q", static_cast<double>(kPhysicalDefaultChargeQ)),
        alpha(frozen_, "alpha", static_cast<double>(kPhysicalDefaultAlpha)),
        n0(frozen_, "n0", static_cast<double>(kPhysicalDefaultBackgroundN0)),
        sign(frozen_, "sign", static_cast<double>(kPhysicalDefaultGravitySign)),
        four_pi_G(frozen_, "four_pi_G", static_cast<double>(kPhysicalDefaultFourPiG)),
        rho0(frozen_, "rho0", static_cast<double>(kPhysicalDefaultGravityRho0)) {}

  ModelSpec(const ModelSpec& other)
      : frozen_(other.frozen_),
        transport(frozen_, "transport", other.transport.get()),
        source(frozen_, "source", other.source.get()),
        elliptic(frozen_, "elliptic", other.elliptic.get()),
        B0(frozen_, "B0", other.B0.get()),
        gamma(frozen_, "gamma", other.gamma.get()),
        cs2(frozen_, "cs2", other.cs2.get()),
        vacuum_floor(frozen_, "vacuum_floor", other.vacuum_floor.get()),
        qom(frozen_, "qom", other.qom.get()),
        q(frozen_, "q", other.q.get()),
        alpha(frozen_, "alpha", other.alpha.get()),
        n0(frozen_, "n0", other.n0.get()),
        sign(frozen_, "sign", other.sign.get()),
        four_pi_G(frozen_, "four_pi_G", other.four_pi_G.get()),
        rho0(frozen_, "rho0", other.rho0.get()) {}

  ModelSpec(ModelSpec&& other) : ModelSpec(static_cast<const ModelSpec&>(other)) {}

  ModelSpec& operator=(const ModelSpec& other) {
    if (this == &other)
      return *this;
    require_mutable("ModelSpec assignment");
    transport = other.transport;
    source = other.source;
    elliptic = other.elliptic;
    B0 = other.B0;
    gamma = other.gamma;
    cs2 = other.cs2;
    vacuum_floor = other.vacuum_floor;
    qom = other.qom;
    q = other.q;
    alpha = other.alpha;
    n0 = other.n0;
    sign = other.sign;
    four_pi_G = other.four_pi_G;
    rho0 = other.rho0;
    frozen_ = other.frozen_;
    return *this;
  }

  ModelSpec& operator=(ModelSpec&& other) { return *this = static_cast<const ModelSpec&>(other); }

  /// Irreversibly seal Python authoring through the public lifecycle API.
  void freeze() noexcept { frozen_ = true; }

  /// Whether the authoring surface has been sealed.
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }

  /// Public lifecycle guard, also used by whole-object assignment and binding diagnostics.
  void require_mutable(const char* field) const {
    if (frozen_)
      throw std::runtime_error(std::string("ModelSpec is frozen; cannot modify '") + field + "'");
  }
};

}  // namespace pops
