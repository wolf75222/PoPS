#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>

#include "component_abi_test_helpers.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

namespace abi = pops::component::test_support;

PopsComponentTableHeaderV1 header(std::size_t size, PopsNativeInterfaceIdV1 id) {
  return {static_cast<std::uint32_t>(size), POPS_COMPONENT_PROTOCOL_ABI_V1, id, 1,
          nullptr, nullptr};
}

PopsComponentStatusV1 ok() {
  return {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
}

TEST(test_amr_compiled_model, ExactTransferTablePreservesConservativeDataflow) {
  const std::array<double, 8> fine{1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0};
  std::array<double, 4> coarse{};
  const std::array<std::int32_t, 2> ratio{1, 2};
  const PopsTransferApiV1 transfer{
      header(sizeof(PopsTransferApiV1), POPS_NATIVE_INTERFACE_TRANSFER_V1),
      +[](void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {
        const auto* source = static_cast<const double*>(request->source.data);
        auto* destination = static_cast<double*>(request->destination.data);
        for (std::size_t point = 0; point < 4; ++point)
          destination[point] = 0.5 * (source[2 * point] + source[2 * point + 1]);
        *status = ok();
        return 0;
      }};
  const PopsTransferRequestV1 transfer_request{
      sizeof(PopsTransferRequestV1),
      abi::const_field_view(fine.data(), 1, 8),
      abi::field_view(coarse.data(), 1, 4),
      ratio.data(), 2, POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1,
      abi::host_execution_context()};
  PopsComponentStatusV1 status{};
  ASSERT_EQ(pops::component::apply_transfer(transfer, nullptr, transfer_request, status), 0);
  EXPECT_EQ(coarse, (std::array<double, 4>{1.0, 2.0, 4.0, 8.0}));

}

TEST(test_amr_compiled_model, TaggingAndClusteringUsePreparedMutableOutputs) {
  const std::array<double, 6> indicator{0.0, 2.0, 3.0, 0.5, 4.0, 0.0};
  std::array<std::uint8_t, 6> tags{};
  const PopsTaggerApiV1 tagger{
      header(sizeof(PopsTaggerApiV1), POPS_NATIVE_INTERFACE_TAGGER_V1),
      +[](void*, const PopsTaggerRequestV1* request, PopsComponentStatusV1* status) {
        const auto* state = static_cast<const double*>(request->states[0].values.data);
        for (std::size_t point = 0; point < request->refine_candidates.size; ++point)
          request->refine_candidates.data[point] =
              state[point] > request->program.leaves[0].threshold;
        *status = ok();
        return 0;
      }};
  const PopsQualifiedConstFieldV1 tag_states{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::indicator",
      abi::const_field_view(indicator.data(), 2, 3)};
  const PopsTaggingLeafV1 tag_leaf{
      sizeof(PopsTaggingLeafV1), 0, 0, 1, 1.0, POPS_TAGGING_NO_STENCIL_V1};
  const std::int32_t tag_op = 1, tag_arg = 0;
  std::array<std::uint8_t, 6> coarsen{}, refine_equalities{}, coarsen_equalities{};
  const PopsTaggerRequestV1 tag_request{
      sizeof(PopsTaggerRequestV1), 1, &tag_states,
      {sizeof(PopsTaggingProgramV1), "case::tag-program", 0, nullptr, 1, &tag_leaf,
       1, &tag_op, &tag_arg, 0, nullptr, nullptr, 0, 0, 0,
       POPS_TAGGING_NON_FINITE_REJECT_V1},
      {0, 0, 0}, {0, 0, 0}, {1, 2, 0}, {1.0, 1.0, 0.0}, 0,
      {sizeof(PopsByteViewV1), tags.data(), tags.size()},
      {sizeof(PopsByteViewV1), coarsen.data(), coarsen.size()},
      {sizeof(PopsByteViewV1), refine_equalities.data(), refine_equalities.size()},
      {sizeof(PopsByteViewV1), coarsen_equalities.data(), coarsen_equalities.size()},
      abi::logical_time(), abi::host_execution_context()};
  PopsComponentStatusV1 status{};
  ASSERT_EQ(pops::component::tag_batch(tagger, nullptr, tag_request, status), 0);
  EXPECT_EQ(tags, (std::array<std::uint8_t, 6>{0, 1, 1, 0, 1, 0}));

  std::array<std::int64_t, 4> boxes{};
  std::size_t box_count = 0;
  const std::array<std::int64_t, 1> extents{6};
  const PopsClusteringApiV1 clustering{
      header(sizeof(PopsClusteringApiV1), POPS_NATIVE_INTERFACE_CLUSTERING_V1),
      +[](void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* result_status) {
        std::size_t first = request->tags.size;
        std::size_t last = 0;
        for (std::size_t point = 0; point < request->tags.size; ++point) {
          if (request->tags.data[point] == 0) continue;
          if (first == request->tags.size) first = point;
          last = point;
        }
        *request->box_count = first == request->tags.size ? 0 : 1;
        if (*request->box_count != 0) {
          request->boxes[0] = static_cast<std::int64_t>(first);
          request->boxes[1] = static_cast<std::int64_t>(last);
        }
        *result_status = ok();
        return 0;
      }};
  const PopsClusteringRequestV1 cluster_request{
      sizeof(PopsClusteringRequestV1),
      {sizeof(PopsConstByteViewV1), tags.data(), tags.size()}, extents.data(), 1,
      boxes.data(), 2, &box_count, abi::host_execution_context()};
  ASSERT_EQ(pops::component::cluster_tags(
                clustering, nullptr, cluster_request, status), 0);
  ASSERT_EQ(box_count, 1u);
  EXPECT_EQ(boxes[0], 1);
  EXPECT_EQ(boxes[1], 4);
}

}  // namespace
