#pragma once

#include "amr_fixture_criteria.hpp"

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/runtime/amr/generated_refinement_criterion.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace pops::test {

inline std::array<Real, 3> standalone_threshold_samples(Real threshold) {
  return {std::nextafter(threshold, Real(0)), threshold,
          std::nextafter(threshold, std::numeric_limits<Real>::infinity())};
}

struct StandaloneFixtureCriterion {
  const FixtureCriterion& criterion;
  const char* path;
  const char* receipt;

  std::string binary_identity() const {
    std::ifstream input(receipt);
    std::string token;
    if (!(input >> token))
      throw std::runtime_error("cannot read the built standalone refinement artifact identity");
    return token;
  }

  runtime::amr::StandaloneCriterionBinding binding() const {
    return {criterion.identity,
            criterion.clock,
            criterion.provider,
            0,
            {{criterion.state, criterion.block, {criterion.variable}, {criterion.component}}}};
  }

  template <class Runtime>
  void install(Runtime& system) const {
    system.install_standalone_refinement_criterion(path, binary_identity(), binding());
  }

  /// Read the exact built entry on three values taken from the real coarse state. This separate
  /// observation leaves the producer's raw bytes available before the runtime applies HOLD.
  template <int Dim>
  std::array<std::uint8_t, 3> marks(const std::array<Real, 3>& observed) const {
    const dynlib::AuthenticatedNativeFile file(path);
    if (file.binary_identity() != binary_identity())
      throw std::runtime_error("standalone fixture changed before its raw-byte observation");
    dynlib::UniqueHandle opened(dynlib::open_private_image(file.load_path()));
    if (!dynlib::valid(opened.get()))
      throw std::runtime_error("cannot open standalone fixture for its raw-byte observation");
    const auto entry = reinterpret_cast<runtime::amr::GeneratedCriterionEntry<Dim>>(
        dynlib::sym(opened.get(), runtime::amr::kGeneratedCriterionEntrySymbol));
    if (!entry)
      throw std::runtime_error("standalone fixture has no raw-byte entry");

    Index<Dim> high{};
    high[0] = 2;
    const Box<Dim> box{Index<Dim>{}, high};
    Fab<Dim> field(box, criterion.component + 1);
    auto host = field.create_host_mirror();
    field.copy_to_host(host);
    for (std::size_t i = 0; i < observed.size(); ++i)
      host(3 * static_cast<std::size_t>(criterion.component) + i) = observed[i];
    field.copy_from_host(host);
    const FieldView<const Real, Dim> view = std::as_const(field).view();
    std::vector<std::uint8_t, fab_allocator<std::uint8_t>> bytes(3);
    std::array<Real, Dim> spacing;
    spacing.fill(Real(1));
    const runtime::amr::GeneratedCriterionPatch<Dim> patch{box,     &view, 1,           spacing,
                                                           nullptr, 0,     bytes.data()};
    if (entry(&patch) != 0)
      throw std::runtime_error("standalone fixture rejected the raw-byte observation patch");
    device_fence();
    return {bytes[0], bytes[1], bytes[2]};
  }
};

}  // namespace pops::test
