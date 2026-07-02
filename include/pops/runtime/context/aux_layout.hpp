#pragma once

#include <pops/core/state/state.hpp>  // kAuxBaseComps, kAuxNamedBase, kAuxMaxComps (component truth)

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// @file
/// @brief Typed descriptor mapping stable field-output HANDLES ("phi", "E_x", a model-named
///        field) to the real ::pops::Aux components they land in. LIGHT host-only header: a
///        debug/report surface WRAPPING the component truth of core/state/state.hpp, it never
///        redeclares or changes the aux constants (kAuxBaseComps / kAuxNamedBase / kAuxMaxComps).
///
/// Motivation (ADC-588): today the "which aux component carries this output" contract is an
/// unwritten convention (phi=0, grad_x=1, grad_y=2, B_z=3, T_e=4, then model-named at
/// kAuxNamedBase+k). AuxLayout names that convention as a value a field problem can carry,
/// validate and report, WITHOUT touching the low-level fixed-component runtime (state.hpp is
/// deliberately the single source of the numeric indices; this wraps it).

namespace pops {

/// Role of a field-output channel, kept COARSE on purpose: it only distinguishes the
/// historical base contract (potential / its gradient) from everything a model names. It exists
/// so reports and validation can talk about a channel without re-encoding the physics; it is NOT
/// a physics enum and carries no numerics.
enum class FieldChannelRole {
  kPotential,  ///< phi (base component 0)
  kGradient,   ///< grad phi component (base components 1..2)
  kNamed,      ///< a canonical extra (B_z, T_e) or a model-named field (>= kAuxNamedBase)
  kCustom      ///< unclassified handle
};

/// One field output: a stable handle name bound to the ::pops::Aux component it occupies.
///
/// @c component is a REAL aux component in [0, kAuxMaxComps); it is derived from the
/// state.hpp layout, never a new numbering. Trivially copyable apart from the handle string
/// (host-only descriptor, not a device type).
struct AuxChannel {
  std::string handle;                            ///< stable name ("phi", "E_x", model field)
  int component = -1;                            ///< real pops::Aux component (0..kAuxMaxComps-1)
  FieldChannelRole role = FieldChannelRole::kCustom;
};

/// The manifest for ONE field problem: the ordered handle<->component map. Purely a
/// host-side descriptor / report surface; the numeric coupling still flows through the fixed
/// ::pops::Aux components. Constructing an AuxLayout does not allocate aux storage; it only
/// records how a field problem's outputs are laid out over the shared aux channel.
class AuxLayout {
 public:
  AuxLayout() = default;

  /// Base contract width (phi + grad phi), always the low three components. Named/extra
  /// channels land at or beyond @c base_width. Mirrors kAuxBaseComps and never redefines it.
  int base_width() const { return base_width_; }

  const std::vector<AuxChannel>& channels() const { return channels_; }

  /// Append an output channel. @p component must be a real aux component in
  /// [0, kAuxMaxComps); the layout refuses out-of-range or duplicate components/handles so a
  /// field problem cannot silently alias two outputs onto one aux slot.
  void add_channel(std::string handle, int component, FieldChannelRole role) {
    if (component < 0 || component >= kAuxMaxComps) {
      throw std::out_of_range("AuxLayout: component " + std::to_string(component) +
                              " for handle '" + handle + "' outside [0, kAuxMaxComps=" +
                              std::to_string(kAuxMaxComps) + ")");
    }
    if (find(handle) != nullptr) {
      throw std::invalid_argument("AuxLayout: duplicate output handle '" + handle + "'");
    }
    for (const auto& c : channels_) {
      if (c.component == component) {
        throw std::invalid_argument("AuxLayout: aux component " + std::to_string(component) +
                                    " already bound to handle '" + c.handle + "', cannot also bind '" +
                                    handle + "'");
      }
    }
    channels_.push_back({std::move(handle), component, role});
  }

  /// Width of the aux channel this layout occupies: one past the highest bound component (so it
  /// can size / bound-check the shared aux). At least the base contract width even when empty.
  int width() const {
    int w = base_width_;
    for (const auto& c : channels_)
      w = (c.component + 1 > w) ? c.component + 1 : w;
    return w;
  }

  /// Handle lookup for reports/debug. Returns nullptr on miss (no throw): callers that need a
  /// hard failure use @ref component_of.
  const AuxChannel* find(std::string_view handle) const {
    for (const auto& c : channels_)
      if (c.handle == handle)
        return &c;
    return nullptr;
  }

  /// Resolve a handle to its real aux component, throwing a MESSAGE that names the missing
  /// output and lists the known handles (the ADC-588 "structured error naming the output"
  /// contract). @p problem_id is woven into the message for field-problem context.
  int component_of(std::string_view handle, std::string_view problem_id = {}) const {
    if (const AuxChannel* c = find(handle))
      return c->component;
    std::string known;
    for (const auto& c : channels_) {
      if (!known.empty())
        known += ", ";
      known += c.handle;
    }
    std::string where = problem_id.empty() ? std::string{} : (" of field problem '" +
                                                              std::string(problem_id) + "'");
    throw std::out_of_range("AuxLayout: unknown output handle '" + std::string(handle) + "'" +
                            where + "; known outputs: [" + known + "]");
  }

 private:
  int base_width_ = kAuxBaseComps;  ///< == 3 (phi/grad in components 0..2); mirrors state.hpp
  std::vector<AuxChannel> channels_;
};

/// The default single-field Poisson layout: phi at component 0, grad phi at components 1..2
/// (the historical base contract). Used as the "phi" field problem's manifest so its layout,
/// like its numerics, is byte-identical to history.
inline AuxLayout default_poisson_layout() {
  AuxLayout layout;
  layout.add_channel("phi", 0, FieldChannelRole::kPotential);
  layout.add_channel("grad_x", 1, FieldChannelRole::kGradient);
  layout.add_channel("grad_y", 2, FieldChannelRole::kGradient);
  return layout;
}

}  // namespace pops
