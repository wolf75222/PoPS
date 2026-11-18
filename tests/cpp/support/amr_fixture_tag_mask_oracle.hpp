#pragma once

#include <gtest/gtest.h>

#include <pops/mesh/index/extent.hpp>

#include <cstddef>

namespace pops::test {

template <class Mask, int Dim, class Expected>
std::size_t expect_fixture_tag_mask(const Mask& mask, const Extent<Dim>& shape, Expected expected) {
  std::size_t checked = 0;
  for (const auto& patch : mask.patches()) {
    for (std::size_t local = 0; local < patch.tags.size(); ++local) {
      std::size_t remainder = local;
      std::size_t global = 0, stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        const auto extent = std::size_t(patch.box.hi[axis] - patch.box.lo[axis] + 1);
        global += (std::size_t(patch.box.lo[axis]) + remainder % extent) * stride;
        remainder /= extent;
        stride *= std::size_t(shape[axis]);
      }
      EXPECT_EQ(patch.tags[local] != 0, expected(global)) << "cell " << global;
      ++checked;
    }
  }
  return checked;
}

}  // namespace pops::test
