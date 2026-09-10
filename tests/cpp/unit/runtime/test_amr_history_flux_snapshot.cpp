#include <gtest/gtest.h>

#include <pops/runtime/program/amr_history_flux_snapshot_codec.hpp>
#include <pops/runtime/program/amr_history_flux_snapshot_execution.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

namespace hf = pops::runtime::program::history_flux;
using pops::Box;
using pops::Index;
using pops::Real;
using pops::amr::Rational;

template <int Dim>
double value(int axis, const Index<Dim>& face, int component) {
  double result = 100 * component + 10 * axis + 2 * face[axis];
  for (int d = 0; d < Dim; ++d)
    if (d != axis)
      result += 3 * face[d];
  return result;
}

template <int Dim>
void fill(hf::Snapshot<Dim>& snapshot) {
  snapshot.owned.clear();
  for (std::size_t patch = 0; patch < snapshot.patches.size(); ++patch) {
    if (snapshot.owners[patch] != snapshot.local_rank)
      continue;
    hf::OwnedPatch<Dim> owned;
    owned.global_patch = patch;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto faces = pops::nd::face_box(snapshot.patches[patch], axis);
      const auto count = static_cast<std::size_t>(faces.numPts());
      owned.density[axis].resize(count * snapshot.components);
      for (std::size_t index = 0; index < count; ++index) {
        auto remainder = index;
        Index<Dim> face;
        for (int d = 0; d < Dim; ++d) {
          face[d] = faces.lo[d] + static_cast<int>(remainder % faces.length(d));
          remainder /= faces.length(d);
        }
        for (int component = 0; component < snapshot.components; ++component)
          owned.density[axis][index + component * count] =
              static_cast<Real>(value(axis, face, component));
      }
    }
    snapshot.owned.push_back(std::move(owned));
  }
}

template <int Dim>
std::shared_ptr<hf::Snapshot<Dim>> raw() {
  auto result = std::make_shared<hf::Snapshot<Dim>>();
  result->identity = "evaluated-source";
  result->source_identity = "operator/point/clock/dt/topology";
  result->components = 2;
  for (int d = 0; d < Dim; ++d) {
    result->domain.lo[d] = -4 + d;
    result->domain.hi[d] = result->domain.lo[d] + 2;
    result->cell_size[d] = 0.25 * (d + 1);
  }
  result->patches.push_back(result->domain);
  result->owners.push_back(0);
  fill(*result);
  return result;
}

template <int Dim>
std::shared_ptr<hf::Snapshot<Dim>> project(std::shared_ptr<const hf::Snapshot<Dim>> parent,
                                           const std::array<int, Dim>& ratio, int shift) {
  auto result = std::make_shared<hf::Snapshot<Dim>>();
  result->identity = parent->identity + "/" + std::to_string(shift);
  result->level = parent->level + 1;
  result->components = parent->components;
  result->rank_count = parent->rank_count;
  result->local_rank = parent->local_rank;
  result->parent = parent;
  result->ratio = ratio;
  for (int d = 0; d < Dim; ++d) {
    result->domain.lo[d] = shift - d;
    result->domain.hi[d] =
        result->domain.lo[d] + static_cast<int>(parent->domain.length(d)) * ratio[d] - 1;
    result->cell_size[d] = parent->cell_size[d] / ratio[d];
  }
  result->patches.push_back(result->domain);
  result->owners.push_back(0);
  return result;
}

template <int Dim>
double evaluate(std::shared_ptr<const hf::Snapshot<Dim>> snapshot, const hf::FaceQuery<Dim>& query,
                int component = 0) {
  double result = 0;
  Rational sum;
  for (const auto& term : hf::source_queries<Dim>(snapshot, query)) {
    const auto payload = hf::local_payload(*term.source, term.query);
    if (!payload)
      throw std::runtime_error("test expected a locally owned source");
    result += term.weight.value() * (*payload)[component];
    sum = sum + term.weight;
  }
  EXPECT_EQ(sum, Rational(1, 1));
  return result;
}

template <int Dim>
void check_shifted_projection() {
  auto source = raw<Dim>();
  std::array<int, Dim> ratio{};
  for (int d = 0; d < Dim; ++d)
    ratio[d] = d + 2;
  auto child = project<Dim>(source, ratio, -17);
  hf::validate_snapshot(*child, 2);
  for (int axis = 0; axis < Dim; ++axis) {
    hf::FaceQuery<Dim> query{axis, child->domain.lo};
    auto parent_face = source->domain.lo;
    for (int d = 0; d < Dim; ++d) {
      query.face[d] += ratio[d] + 1;
      ++parent_face[d];
    }
    const auto terms = hf::source_queries<Dim>(child, query);
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_EQ(terms[0].query.face, parent_face);
    EXPECT_EQ(terms[0].weight, Rational(ratio[axis] - 1, ratio[axis]));
    ++parent_face[axis];
    EXPECT_EQ(terms[1].query.face, parent_face);
    EXPECT_EQ(terms[1].weight, Rational(1, ratio[axis]));
    --parent_face[axis];
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(evaluate<Dim>(child, query, component),
                  value(axis, parent_face, component) + 2.0 / ratio[axis], 1e-12);
    query.face = child->domain.lo;
    query.face[axis] = child->domain.hi[axis] + 1;
    EXPECT_EQ(hf::source_queries<Dim>(child, query).size(), 1u);
  }
  // Constant samples survive every synthetic interpolation exactly.
  for (auto& patch : source->owned)
    for (auto& axis : patch.density)
      for (auto& sample : axis)
        sample = Real(7);
  for (int axis = 0; axis < Dim; ++axis) {
    auto face = child->domain.lo;
    ++face[axis];
    EXPECT_DOUBLE_EQ(evaluate<Dim>(child, {axis, face}), 7.0);
  }
}

TEST(AmrHistoryFluxSnapshot, ZeroTermRegistryRetainsItsWireShape) {
  struct Term {
    std::shared_ptr<const hf::Basis<2>> basis;
    std::map<int, Rational> coefficient;
  };
  using Expression = std::map<std::uint64_t, Term>;
  std::map<std::string, std::vector<Expression>> histories;
  const auto no_shared_samples = [](auto&, const auto&) {
    throw std::logic_error("zero-term history must not serialize a physical sample");
  };
  EXPECT_TRUE(hf::serialize_expressions<2>(histories, 2, no_shared_samples).empty());
  histories.emplace("0:tracer.prior", std::vector<Expression>(2));
  const auto bytes = hf::serialize_expressions<2>(histories, 2, no_shared_samples);
  ASSERT_FALSE(bytes.empty());
  pops::runtime::program::checkpoint_detail::Reader in(bytes);
  EXPECT_EQ(in.u64(), UINT64_C(0x504f5053464c5833));
  EXPECT_EQ(in.size(), 1U);
  EXPECT_EQ(in.string(), "0:tracer.prior");
  EXPECT_EQ(in.size(), 2U);
  EXPECT_EQ(in.size(), 0U);
  EXPECT_EQ(in.size(), 0U);
  EXPECT_NO_THROW(in.finish());
}

TEST(AmrHistoryFluxSnapshot, InterfaceMappingWidensIndependentOriginOffsets) {
  using Layout = pops::amr::hierarchy::LevelLayout<2>;
  using Patches = pops::mesh::BoxArray<2>;
  using Distribution = pops::mesh::Distribution<2>;
  using Ratio = pops::amr::RefinementRatio<2>;
  struct Interface {
    int axis;
    Index<2> coarse_face;
  };
  const int lowest = std::numeric_limits<int>::min();
  const pops::mesh::RankSpace<2> ranks(Index<2>{0, 0}, pops::Extent<2>{1, 1});
  const pops::mesh::BoxArrayValidationBudget budget{1, 0};
  const auto layout = [&](int level, const Box<2>& domain, const Ratio& ratio) {
    const Patches patches(std::vector<Box<2>>{domain});
    return Layout(level, domain, patches, Distribution::replicated(patches, ranks), ratio, budget);
  };
  // No field allocation: only two patch descriptors and two requested tangential faces.
  for (const bool wide_subtraction : {false, true}) {
    SCOPED_TRACE(wide_subtraction);
    const Box<2> parent_domain{Index<2>{wide_subtraction ? lowest : lowest / 2, 0}, Index<2>{0, 1}};
    const Box<2> child_domain{Index<2>{wide_subtraction ? lowest + 1 : lowest, 0}, Index<2>{1, 3}};
    const auto parent = layout(0, parent_domain, Ratio{});
    const auto child = layout(1, child_domain, Ratio{wide_subtraction ? 1 : 2, 2});
    const auto geometry = pops::Geometry<2>::from_bounds(child_domain, pops::RealVector<2>{0, 0},
                                                         pops::RealVector<2>{1, 1});
    const std::vector<Interface> incoming{{0, Index<2>{0, 1}}};
    const auto faces = hf::execution::interface_faces<2, hf::BasisFace<2>>(
        geometry, child, &parent, std::vector<Interface>{}, incoming);
    ASSERT_EQ(faces.size(), 2U);
    for (std::size_t ordinal = 0; ordinal < faces.size(); ++ordinal) {
      EXPECT_EQ(faces[ordinal].role, pops::amr::reflux::FaceLedgerRole::Fine);
      EXPECT_EQ(faces[ordinal].face, (Index<2>{wide_subtraction ? 1 : 0, 2 + int(ordinal)}));
      EXPECT_EQ(faces[ordinal].coarse_face, (Index<2>{0, 1}));
      EXPECT_DOUBLE_EQ(faces[ordinal].face_measure, 0.25);
    }
    const std::vector<Interface> unrepresentable{{0, Index<2>{std::numeric_limits<int>::max(), 1}}};
    EXPECT_THROW((hf::execution::interface_faces<2, hf::BasisFace<2>>(
                     geometry, child, &parent, std::vector<Interface>{}, unrepresentable)),
                 std::overflow_error);
  }
}

TEST(AmrHistoryFluxSnapshot, ShiftedAsymmetricProjection1D) {
  check_shifted_projection<1>();
}
TEST(AmrHistoryFluxSnapshot, ShiftedAsymmetricProjection2D) {
  check_shifted_projection<2>();
}
TEST(AmrHistoryFluxSnapshot, ShiftedAsymmetricProjection3D) {
  check_shifted_projection<3>();
}

template <int Dim>
void check_interface_integral() {
  const auto source = raw<Dim>();
  std::array<int, Dim> ratio{};
  for (int d = 0; d < Dim; ++d)
    ratio[d] = d + 2;
  const auto child = project<Dim>(source, ratio, 13);
  for (int axis = 0; axis < Dim; ++axis) {
    auto coarse_face = source->domain.lo;
    ++coarse_face[axis];
    double coarse_measure = 1;
    double fine_measure = 1;
    int count = 1;
    for (int d = 0; d < Dim; ++d)
      if (d != axis) {
        count *= ratio[d];
        coarse_measure *= source->cell_size[d];
        fine_measure *= child->cell_size[d];
      }
    double integral = 0;
    for (int n = 0; n < count; ++n) {
      auto face = child->domain.lo;
      face[axis] += ratio[axis];
      int remainder = n;
      for (int d = 0; d < Dim; ++d)
        if (d != axis) {
          face[d] += remainder % ratio[d];
          remainder /= ratio[d];
        }
      integral += evaluate<Dim>(child, {axis, face}) * fine_measure;
    }
    EXPECT_NEAR(integral, value(axis, coarse_face, 0) * coarse_measure, 1e-12);
  }
}

TEST(AmrHistoryFluxSnapshot, FineFaceIntegralsCommute1D) {
  check_interface_integral<1>();
}
TEST(AmrHistoryFluxSnapshot, FineFaceIntegralsCommute2D) {
  check_interface_integral<2>();
}
TEST(AmrHistoryFluxSnapshot, FineFaceIntegralsCommute3D) {
  check_interface_integral<3>();
}

TEST(AmrHistoryFluxSnapshot, RecursiveProjectionCoalescesToDirectExactSourceTerms) {
  const auto source = raw<3>();
  const auto middle = project<3>(source, {2, 3, 1}, -21);
  const auto fine = project<3>(middle, {3, 1, 2}, 37);
  const auto direct = project<3>(source, {6, 3, 2}, 37);
  hf::validate_snapshot(*fine, 3);
  for (int axis = 0; axis < 3; ++axis)
    for (int displacement = 0; displacement <= 12; ++displacement) {
      auto face = fine->domain.lo;
      face[axis] += displacement % (fine->domain.length(axis) + 1);
      for (int d = 0; d < 3; ++d)
        if (d != axis)
          face[d] += displacement % fine->domain.length(d);
      const auto recursive_terms = hf::source_queries<3>(fine, {axis, face});
      const auto direct_terms = hf::source_queries<3>(direct, {axis, face});
      ASSERT_EQ(recursive_terms.size(), direct_terms.size());
      EXPECT_LE(recursive_terms.size(), 2u);
      for (std::size_t i = 0; i < direct_terms.size(); ++i) {
        EXPECT_EQ(recursive_terms[i].source.get(), source.get());
        EXPECT_EQ(recursive_terms[i].query, direct_terms[i].query);
        EXPECT_EQ(recursive_terms[i].weight, direct_terms[i].weight);
      }
    }
  EXPECT_THROW(hf::validate_snapshot(*fine, 2), std::invalid_argument);
  EXPECT_THROW(hf::source_queries<3>(fine, {0, fine->domain.lo}, 2), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, SharedFaceUsesOnlyLowestGlobalPatchOwner) {
  auto rank0 = raw<1>();
  rank0->rank_count = 2;
  rank0->patches = {Box<1>{Index<1>{-4}, Index<1>{-4}}, Box<1>{Index<1>{-3}, Index<1>{-2}}};
  rank0->owners = {0, 1};
  fill(*rank0);
  auto rank1 = std::make_shared<hf::Snapshot<1>>(*rank0);
  rank1->local_rank = 1;
  fill(*rank1);
  hf::validate_snapshot(*rank0, 1);
  hf::validate_snapshot(*rank1, 1);
  const hf::FaceQuery<1> shared{0, Index<1>{-3}};
  const auto owner_payload = hf::local_payload(*rank0, shared);
  ASSERT_TRUE(owner_payload);
  EXPECT_EQ((*owner_payload)[0], Real(-6));
  EXPECT_EQ((*owner_payload)[1], Real(94));
  EXPECT_FALSE(hf::local_payload(*rank1, shared));
  EXPECT_TRUE(hf::local_payload(*rank1, {0, Index<1>{-2}}));
  // Storage redistribution does not change the externally supplied physical source identity.
  EXPECT_EQ(rank0->identity, rank1->identity);
}

TEST(AmrHistoryFluxSnapshot, MissingSupportNeverBecomesZero) {
  auto source = raw<1>();
  source->patches[0].hi[0] = -4;
  fill(*source);
  const auto child = project<1>(source, {2}, 5);
  hf::validate_snapshot(*child, 2);
  EXPECT_THROW(hf::source_queries<1>(child, {0, Index<1>{8}}), std::out_of_range);
  EXPECT_THROW(hf::local_payload(*source, {0, Index<1>{-2}}), std::out_of_range);
  EXPECT_THROW(hf::source_queries<1>(child, {0, Index<1>{4}}), std::out_of_range);
  EXPECT_THROW(hf::source_queries<1>(child, {1, Index<1>{5}}), std::invalid_argument);
  EXPECT_THROW(hf::local_payload(*child, {0, Index<1>{5}}), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, MissingPartialDuplicateForeignAndNonfinitePayloadsAreRejected) {
  const auto source = raw<2>();
  auto broken = *source;
  broken.owned.clear();
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  EXPECT_THROW(hf::local_payload(broken, {0, broken.domain.lo}), std::invalid_argument);
  broken = *source;
  broken.owned[0].density[1].pop_back();
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  EXPECT_THROW(hf::local_payload(broken, {1, broken.domain.lo}), std::invalid_argument);
  broken = *source;
  broken.owned.push_back(broken.owned[0]);
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  EXPECT_THROW(hf::local_payload(broken, {0, broken.domain.lo}), std::invalid_argument);
  broken = *source;
  broken.rank_count = 2;
  broken.owners[0] = 1;
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  broken = *source;
  broken.owned[0].density[0][0] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  EXPECT_THROW(hf::local_payload(broken, {0, broken.domain.lo}), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, InvalidGeometryMetricAndLineageAreRejected) {
  const auto source = raw<2>();
  const auto child = project<2>(source, {2, 3}, -14);
  auto broken = *child;
  broken.ratio[0] = 0;
  EXPECT_THROW(hf::validate_snapshot(broken, 2), std::invalid_argument);
  broken = *child;
  ++broken.domain.hi[0];
  EXPECT_THROW(hf::validate_snapshot(broken, 2), std::invalid_argument);
  broken = *child;
  broken.cell_size[1] *= 2;
  EXPECT_THROW(hf::validate_snapshot(broken, 2), std::invalid_argument);
  broken = *child;
  broken.level = source->level;
  EXPECT_THROW(hf::validate_snapshot(broken, 2), std::invalid_argument);
  broken = *child;
  broken.identity = source->identity;
  EXPECT_THROW(hf::validate_snapshot(broken, 2), std::invalid_argument);
  broken = *source;
  broken.patches.push_back(broken.patches[0]);
  broken.owners.push_back(0);
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  broken = *source;
  broken.owners[0] = broken.rank_count;
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  broken = *source;
  broken.owned[0].global_patch = 100;
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
  broken = *source;
  broken.source_identity.clear();
  EXPECT_THROW(hf::validate_snapshot(broken, 1), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, ResidentAccountingIncludesSourceIdentityAndBoundedAncestry) {
  const auto source = raw<2>();
  const auto baseline = hf::resident_bytes(*source);
  const auto old_capacity = source->source_identity.capacity();
  source->source_identity.reserve(4096);
  EXPECT_EQ(hf::resident_bytes(*source),
            baseline + source->source_identity.capacity() - old_capacity);
  const auto child = project<2>(source, {2, 3}, 17);
  EXPECT_GT(hf::resident_bytes(*child), hf::resident_bytes(*source));
  EXPECT_THROW(hf::resident_bytes(*child, 1), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, OversizedCoordinatesAndFaceArraysAreRejectedBeforeAllocation) {
  auto source = raw<3>();
  source->domain.hi[0] = std::numeric_limits<int>::max();
  EXPECT_THROW(hf::validate_snapshot(*source, 1), std::overflow_error);
  for (int d = 0; d < 3; ++d) {
    source->domain.lo[d] = std::numeric_limits<int>::min();
    source->domain.hi[d] = std::numeric_limits<int>::max() - 1;
  }
  source->patches = {source->domain};
  EXPECT_THROW(hf::validate_snapshot(*source, 1), std::overflow_error);
  EXPECT_THROW(hf::validate_snapshot(*raw<1>(), 0), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshot, CertifiedGeometryCannotAuthenticateModifiedSnapshotCopy) {
  auto source = raw<1>();
  source->patches = {Box<1>{Index<1>{-4}, Index<1>{-4}}, Box<1>{Index<1>{-3}, Index<1>{-3}},
                     Box<1>{Index<1>{-2}, Index<1>{-2}}};
  source->owners = {0, 0, 0};
  fill(*source);
  const auto uncertified_bytes = hf::resident_bytes(*source);
  hf::certify_snapshot(*source, 1);
  EXPECT_GT(hf::resident_bytes(*source), uncertified_bytes);
  auto changed = *source;
  changed.patches[1] = changed.patches[0];
  fill(changed);  // Complete, finite, correctly owned payload; only geometry is invalid.
  EXPECT_THROW(hf::validate_snapshot(changed, 1), std::invalid_argument);
  EXPECT_THROW(hf::certify_snapshot(changed, 1), std::invalid_argument);
  EXPECT_NO_THROW(hf::validate_snapshot(*source, 1));
  changed = *source;
  changed.owned[0].density[0][0] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(hf::validate_snapshot(changed, 1), std::invalid_argument);
}

struct ShardedArchive {
  std::string token;
  std::array<std::shared_ptr<hf::Snapshot<1>>, 2> sources;
  std::vector<std::vector<std::uint8_t>> bytes;
};

ShardedArchive sharded_archive() {
  auto complete = raw<1>();
  complete->patches = {Box<1>{Index<1>{-4}, Index<1>{-4}}, Box<1>{Index<1>{-3}, Index<1>{-3}},
                       Box<1>{Index<1>{-2}, Index<1>{-2}}};
  complete->owners = {0, 0, 0};
  fill(*complete);
  std::vector<std::string> digests;
  for (const auto& patch : complete->owned)
    digests.push_back(hf::patch_digest(patch));
  ShardedArchive result;
  result.token = hf::content_identity(*complete, digests);
  for (int rank = 0; rank < 2; ++rank) {
    auto source = std::make_shared<hf::Snapshot<1>>(*complete);
    source->identity = result.token;
    source->rank_count = 2;
    source->local_rank = rank;
    source->owners = {0, 1, 0};
    fill(*source);
    result.sources[rank] = source;
    result.bytes.push_back(hf::encode_shard<1>({{result.token, source}}, 1));
  }
  return result;
}

TEST(AmrHistoryFluxSnapshotCodec, RankChangePreservesContentIdentityAndUniqueFaceOwnership) {
  const auto archive = sharded_archive();
  std::array<hf::SnapshotMap<1>, 3> restored;
  for (int rank = 0; rank < 3; ++rank) {
    restored[rank] = hf::decode_shards<1>(archive.bytes, 2, 3, rank, 1, 1u << 20);
    ASSERT_EQ(restored[rank].size(), 1u);
    const auto& snapshot = *restored[rank].at(archive.token);
    EXPECT_EQ(snapshot.identity, archive.token);
    EXPECT_EQ(snapshot.source_identity, archive.sources[0]->source_identity);
    EXPECT_EQ(snapshot.owners, (std::vector<int>{0, 1, 2}));
    ASSERT_EQ(snapshot.owned.size(), 1u);
    EXPECT_EQ(snapshot.owned[0].global_patch, static_cast<std::size_t>(rank));
  }
  for (int coordinate = -4; coordinate <= -1; ++coordinate) {
    int count = 0;
    for (int rank = 0; rank < 3; ++rank) {
      const auto payload =
          hf::local_payload(*restored[rank].at(archive.token), {0, Index<1>{coordinate}});
      if (payload) {
        ++count;
        EXPECT_EQ((*payload)[0], Real(2 * coordinate));
        EXPECT_EQ((*payload)[1], Real(100 + 2 * coordinate));
      }
    }
    EXPECT_EQ(count, 1);
  }
  const auto unchanged = hf::decode_shards<1>(archive.bytes, 2, 2, 1, 1, 1u << 20);
  EXPECT_EQ(unchanged.at(archive.token)->owners, (std::vector<int>{0, 1, 0}));
  const auto collapsed = hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20);
  EXPECT_EQ(collapsed.at(archive.token)->owned.size(), 3u);
  std::vector<std::string> digests;
  for (const auto& patch : collapsed.at(archive.token)->owned)
    digests.push_back(hf::patch_digest(patch));
  EXPECT_EQ(hf::content_identity(*collapsed.at(archive.token), digests), archive.token);
  const std::vector<std::vector<std::uint8_t>> single_shard{hf::encode_shard<1>(collapsed, 1)};
  for (int rank = 0; rank < 2; ++rank) {
    const auto expanded = hf::decode_shards<1>(single_shard, 1, 2, rank, 1, 1u << 20);
    const auto& snapshot = *expanded.at(archive.token);
    EXPECT_EQ(snapshot.owners, (std::vector<int>{0, 1, 0}));
    EXPECT_EQ(snapshot.owned.size(), rank == 0 ? 2u : 1u);
    for (int coordinate = -4; coordinate <= -1; ++coordinate) {
      const auto payload = hf::local_payload(snapshot, {0, Index<1>{coordinate}});
      const int expected_owner = coordinate == -2 ? 1 : 0;
      EXPECT_EQ(payload.has_value(), rank == expected_owner);
      if (payload)
        EXPECT_EQ((*payload)[1], Real(100 + 2 * coordinate));
    }
  }
}

TEST(AmrHistoryFluxSnapshotCodec, CorruptedContentAndPhysicalMetadataFailDigestValidation) {
  auto archive = sharded_archive();
  auto changed = *archive.sources[0];
  changed.owned[0].density[0][0] += Real(1);
  archive.bytes[0] =
      hf::encode_shard<1>({{archive.token, std::make_shared<const hf::Snapshot<1>>(changed)}}, 1);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  archive = sharded_archive();
  for (int rank = 0; rank < 2; ++rank) {
    changed = *archive.sources[rank];
    changed.source_identity += "/foreign-clock";
    archive.bytes[rank] =
        hf::encode_shard<1>({{archive.token, std::make_shared<const hf::Snapshot<1>>(changed)}}, 1);
  }
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshotCodec, MissingForeignAndUnboundedShardEnvelopesAreRejected) {
  auto archive = sharded_archive();
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 1, 1, 0, 1, 1u << 20), std::invalid_argument);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 1, 1, 1u << 20), std::invalid_argument);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1), std::invalid_argument);
  std::swap(archive.bytes[0], archive.bytes[1]);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  archive = sharded_archive();
  archive.bytes[1].clear();
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  archive = sharded_archive();
  archive.bytes[1].pop_back();
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::runtime_error);
  archive = sharded_archive();
  archive.bytes[1].push_back(0);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::runtime_error);
}

// Produce malformed wire input without going through the encoder's publication validator.
std::vector<std::uint8_t> unchecked_shard(
    const hf::Snapshot<1>& source, std::optional<double> first_scalar_override = std::nullopt) {
  pops::runtime::program::checkpoint_detail::Writer out;
  out.u64(UINT64_C(0x504f505348465831));
  out.size(1);
  out.string(source.identity);
  hf::write_physical_metadata(out, source);
  out.i32(source.rank_count);
  out.i32(source.local_rank);
  out.size(source.owners.size());
  for (int owner : source.owners)
    out.i32(owner);
  out.size(source.owned.size());
  bool first = true;
  for (const auto& patch : source.owned) {
    out.u64(patch.global_patch);
    for (const auto& axis : patch.density) {
      out.size(axis.size());
      for (Real sample : axis) {
        double wire = static_cast<double>(sample);
        if (first && first_scalar_override)
          wire = *first_scalar_override;
        out.real(wire);
        first = false;
      }
    }
  }
  return std::move(out).take();
}

TEST(AmrHistoryFluxSnapshotCodec, PartialDuplicateAndForeignOwnedSamplesAreRejected) {
  auto archive = sharded_archive();
  auto broken = *archive.sources[0];
  broken.owned.pop_back();
  archive.bytes[0] = unchecked_shard(broken);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  broken = *archive.sources[0];
  broken.owned[1] = broken.owned[0];
  archive.bytes[0] = unchecked_shard(broken);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  broken = *archive.sources[0];
  broken.owners[0] = 1;
  archive.bytes[0] = unchecked_shard(broken);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  broken = *archive.sources[0];
  broken.owned[0].density[0].pop_back();
  archive.bytes[0] = unchecked_shard(broken);
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshotCodec, WireScalarBitChangesCannotDisappearThroughNativeNarrowing) {
  auto archive = sharded_archive();
  ASSERT_EQ(unchecked_shard(*archive.sources[0]), archive.bytes[0]);
  const double original = static_cast<double>(archive.sources[0]->owned[0].density[0][0]);
  archive.bytes[0] = unchecked_shard(
      *archive.sources[0], std::nextafter(original, std::numeric_limits<double>::infinity()));
  EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
}

TEST(AmrHistoryFluxSnapshotCodec, OversizedAndNonfiniteWireScalarsAreRejected) {
  auto archive = sharded_archive();
  for (double encoded :
       {std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    archive.bytes[0] = unchecked_shard(*archive.sources[0], encoded);
    EXPECT_THROW(hf::decode_shards<1>(archive.bytes, 2, 1, 0, 1, 1u << 20), std::invalid_argument);
  }
}

TEST(AmrHistoryFluxSnapshotCodec, SignedZeroSurvivesExactPayloadRestoration) {
  auto source = raw<1>();
  source->owned[0].density[0][0] = -Real(0);
  source->identity = hf::content_identity(*source, {hf::patch_digest(source->owned[0])});
  const std::vector<std::vector<std::uint8_t>> bytes{
      hf::encode_shard<1>({{source->identity, source}}, 1)};
  const auto restored = hf::decode_shards<1>(bytes, 1, 1, 0, 1, 1u << 20);
  const auto payload = hf::local_payload(*restored.at(source->identity), {0, source->domain.lo});
  ASSERT_TRUE(payload);
  EXPECT_EQ((*payload)[0], Real(0));
  EXPECT_TRUE(std::signbit((*payload)[0]));
}

TEST(AmrHistoryFluxSnapshotCodec, ProjectionDescriptorAuthenticatesTargetAndParentLineage) {
  const auto archive = sharded_archive();
  const auto restored = hf::decode_shards<1>(archive.bytes, 2, 1, 0, 2, 1u << 20);
  auto projection = project<1>(restored.at(archive.token), {2}, 5);
  projection->identity = hf::projection_identity(*projection);
  const auto round_trip = [&](std::shared_ptr<const hf::Snapshot<1>> snapshot) {
    pops::runtime::program::checkpoint_detail::Writer out;
    hf::write_descriptor<1>(out, snapshot, 2);
    const auto bytes = std::move(out).take();
    pops::runtime::program::checkpoint_detail::Reader in(bytes);
    const auto decoded = hf::read_descriptor<1>(in, restored, 2);
    in.finish();
    return decoded;
  };
  const auto decoded = round_trip(projection);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->identity, projection->identity);
  EXPECT_EQ(decoded->parent.get(), restored.at(archive.token).get());
  EXPECT_DOUBLE_EQ(evaluate<1>(decoded, {0, Index<1>{6}}), -7.0);
  auto changed = std::make_shared<hf::Snapshot<1>>(*projection);
  changed->identity += "/foreign-projection";
  EXPECT_THROW(round_trip(changed), std::invalid_argument);
  changed = std::make_shared<hf::Snapshot<1>>(*projection);
  ++changed->domain.lo[0];
  ++changed->domain.hi[0];
  changed->patches = {changed->domain};
  hf::validate_snapshot(*changed, 2);  // Structurally valid, but a different coordinate mapping.
  EXPECT_THROW(round_trip(changed), std::invalid_argument);
  changed = project<1>(restored.at(archive.token), {3}, 5);
  changed->identity = projection->identity;
  hf::validate_snapshot(*changed, 2);  // Structurally valid, but a different refinement lineage.
  EXPECT_THROW(round_trip(changed), std::invalid_argument);
  auto foreign_parent = std::make_shared<hf::Snapshot<1>>(*projection->parent);
  foreign_parent->source_identity += "/foreign-clock";
  changed = std::make_shared<hf::Snapshot<1>>(*projection);
  changed->parent = foreign_parent;
  changed->identity = hf::projection_identity(*changed);
  EXPECT_THROW(round_trip(changed), std::invalid_argument);
}

}  // namespace
