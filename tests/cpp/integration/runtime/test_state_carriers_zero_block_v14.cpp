#include <gtest/gtest.h>
#include <pops/runtime/checkpoint/state_carriers.hpp>

using namespace pops::runtime::checkpoint;

namespace {
template <int Dim>
void check_empty_archive() {
  const std::array<std::uint64_t, 1> levels{0};
  const std::vector<StateCarrierStorageCapacity<Dim>> blocks;
  const auto capacity = state_carriers_byte_capacity<Dim>(levels, blocks);
  ASSERT_EQ(capacity, 64u);
  for (const std::uint64_t ranks : {1u, 2u, 3u}) {
    std::vector<std::string> shards;
    for (std::uint64_t rank = 0; rank < ranks; ++rank) {
      StateCarrierArchive<Dim> image;
      image.real_bits = 64;
      image.ranks = ranks;
      image.levels = 1;
      image.shard = static_cast<std::int64_t>(rank);
      const auto bytes = encode_state_carriers(image);
      EXPECT_EQ(bytes.size(), capacity);
      const auto roundtrip = decode_state_carriers<Dim>(bytes);
      EXPECT_TRUE(roundtrip.blocks.empty());
      EXPECT_TRUE(roundtrip.patches.empty());
      EXPECT_EQ(roundtrip.levels, 1u);
      EXPECT_EQ(roundtrip.shard, image.shard);
      shards.emplace_back(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    const auto merged = merge_state_carrier_shards<Dim>(shards);
    EXPECT_NO_THROW(validate_complete_state_carriers(merged));
    EXPECT_TRUE(merged.blocks.empty());
    EXPECT_TRUE(merged.patches.empty());
    EXPECT_EQ(merged.shard, -1);
    EXPECT_EQ(merged.ranks, ranks);
    EXPECT_EQ(encode_state_carriers(merged).size(), capacity);
  }
}
}  // namespace

TEST(StateCarriersZeroBlockV14, EmptyArchiveHasExactNativeWireCapacity) {
  check_empty_archive<1>();
  check_empty_archive<2>();
  check_empty_archive<3>();
}

TEST(StateCarriersZeroBlockV14, ZeroCellsDoNotPermitMissingNonemptyCarrierStorage) {
  const std::array<std::uint64_t, 1> zero_cells{0};
  const std::array<std::uint64_t, 0> no_levels{};
  const std::vector<StateCarrierStorageCapacity<2>> no_blocks;
  const std::vector<StateCarrierStorageCapacity<2>> block{{"q", 1, {0, 0}}};
  EXPECT_THROW(state_carriers_byte_capacity<2>(zero_cells, block), std::invalid_argument);
  EXPECT_THROW(state_carriers_byte_capacity<2>(no_levels, no_blocks), std::invalid_argument);
}

TEST(StateCarriersZeroBlockV14, EmptyArchiveStillRequiresExactAuthorityAndNoOrphanPatch) {
  StateCarrierArchive<2> image;
  image.real_bits = 64;
  image.ranks = 2;
  image.levels = 1;
  image.shard = -1;
  auto bytes = encode_state_carriers(image);
  EXPECT_NO_THROW(validate_complete_state_carriers(decode_state_carriers<2>(bytes)));
  bytes.push_back(0);
  EXPECT_THROW(decode_state_carriers<2>(bytes), std::invalid_argument);
  image.patches.push_back({});
  EXPECT_THROW(encode_state_carriers(image), std::invalid_argument);
  EXPECT_THROW(validate_complete_state_carriers(image), std::invalid_argument);
  image.patches.clear();
  image.levels = 0;
  EXPECT_THROW(encode_state_carriers(image), std::invalid_argument);
  image.levels = 1;
  image.shard = 2;
  EXPECT_THROW(encode_state_carriers(image), std::invalid_argument);
}
