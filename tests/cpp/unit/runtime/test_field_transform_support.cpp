#include <pops/runtime/program/field_transform_support.hpp>
#include <gtest/gtest.h>

namespace {
using Support = pops::runtime::program::FieldTransformSupport;
const auto geometry = pops::Geometry<1>::from_bounds(
    {pops::Index<1>{-2}, pops::Index<1>{3}}, pops::RealVector<1>{-1}, pops::RealVector<1>{2});
constexpr std::uint64_t capacity = 1 << 20;
}  // namespace

TEST(FieldTransformSupport, OriginAndActualGeometryAreExactAndPartitionFree) {
  const Support old{"program", "point/n-1/q", "", 0, 1, 7, 9};
  const auto image = old.encode(geometry, 2, capacity);
  EXPECT_EQ(old.encode(geometry, 2, capacity), image);
  EXPECT_NE((Support{"program", "point/n/q", "", 0, 1, 7, 9}.encode(geometry, 2, capacity)), image);
  EXPECT_NE((Support{"program", "point/n-1/q", "", 1, 1, 7, 9}.encode(geometry, 2, capacity)),
            image);
  EXPECT_NE((Support{"program", "point/n-1/q", "", 0, 0, 7, 9}.encode(geometry, 2, capacity)),
            image);
  EXPECT_NE((Support{"program", "point/n-1/q", "", 0, 1, 8, 9}.encode(geometry, 2, capacity)),
            image);
  EXPECT_NE(old.encode(geometry, 1, capacity), image);
  const auto moved = pops::Geometry<1>::from_bounds(geometry.domain(), pops::RealVector<1>{-1},
                                                    pops::RealVector<1>{3});
  EXPECT_NE(old.encode(moved, 2, capacity), image);
  // No rank/patch allocator, evaluation ordinal or current time is an input:
  // an old, readable historical support stays the same on rematerialization.
}

TEST(FieldTransformSupport, IndependentProvidersAndInvalidAuthorityRejectExactly) {
  const Support first{"program", "point/n-1/phi", "problem/one", -1, 0, 2, 4};
  const Support second{"program", "point/n-1/phi", "problem/two", -1, 0, 2, 4};
  EXPECT_NE(first.encode(geometry, 1, capacity), second.encode(geometry, 1, capacity));
  EXPECT_THROW((Support{"program", "point", "provider", 0, 0}.encode(geometry, 1, capacity)),
               std::invalid_argument);
  EXPECT_THROW((Support{"program", "point", "", -1, 0}.encode(geometry, 1, capacity)),
               std::invalid_argument);
  EXPECT_THROW((Support{"program", "", "", 0, 0}.encode(geometry, 1, capacity)),
               std::invalid_argument);
  EXPECT_THROW(first.encode(geometry, 1, 1), std::length_error);
  EXPECT_NO_THROW(first.encode(geometry, 1, capacity));
}

TEST(FieldTransformSupport, FirstProductionBindsTheNativePointAndKeepsZeroTimeRemap) {
  using namespace pops::runtime::program;
  pops::runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = "clock";
  point.tick = 4;
  point.level = 1;
  point.substep = 2;
  point.stage = 7;
  point.stage_fraction = {1, 2};
  point.dt = 0.25;
  point.physical_time = 1.125;
  const auto source = field_transform_production_origin("type", 17, point, 1ULL << 20);
  const auto changed = [&](auto edit) {
    auto other = point;
    edit(other);
    EXPECT_NE(source, field_transform_production_origin("type", 17, other, 1ULL << 20));
  };
  changed([](auto& p) { ++p.tick; });
  changed([](auto& p) { ++p.substep; });
  changed([](auto& p) { p.physical_time = 1.25; });
  changed([](auto& p) { p.stage_fraction = {3, 4}; });
  EXPECT_NE(source, field_transform_production_origin("type", 18, point, 1ULL << 20));
  EXPECT_THROW(field_transform_production_origin("type", 17, point, 8), std::length_error);
  point.dt = 0;
  EXPECT_NO_THROW(field_transform_production_origin("type", 17, point, 1ULL << 20));
  point.dt = -1;
  EXPECT_THROW(field_transform_production_origin("type", 17, point, 1ULL << 20),
               std::invalid_argument);
}
