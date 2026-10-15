#include <gtest/gtest.h>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/continuation_field_migration.hpp>

using namespace pops;
using namespace pops::runtime::program;

namespace {
constexpr int D = kNativeDimension;
using V = ContinuationValue<D>;
using R = ContinuationValueRef<D>;
using Codec = ContinuationCodec<D>;

MultiFab<D> migration_field(int ranks, int rank, int halo = 1, int components = 1) {
  Extent<D> cells{}, block{}, process_extent{}, ghosts{};
  Index<D> origin{}, coordinate{};
  for (int d = 0; d < D; ++d) {
    cells[d] = d == 0 ? 12 : 2;
    block[d] = 2;
    process_extent[d] = d == 0 ? ranks : 1;
    ghosts[d] = halo;
  }
  coordinate[0] = rank;
  auto boxes = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), block);
  mesh::RankSpace<D> space(origin, process_extent);
  std::vector<Index<D>> owners;
  for (std::size_t p = 0; p < boxes.size(); ++p)
    owners.push_back(space.coordinate(p % static_cast<std::size_t>(ranks)));
  auto distribution = mesh::Distribution<D>::partitioned(boxes, space, std::move(owners));
  MultiFab<D> field(boxes, distribution, coordinate, components, ghosts);
  for (std::size_t p = 0; p < field.local_size(); ++p) {
    auto host = field.fab(p).create_host_mirror();
    for (std::size_t i = 0; i < host.size(); ++i) {
      host(i) = static_cast<Real>(1000 * field.global_index(p) + i);
      if (i == 0)
        host(i) = -Real(0);
    }
    field.fab(p).copy_from_host(host);
  }
  return field;
}

ContinuationType migration_type(int components = 1) {
  ContinuationType field;
  field.kind = ContinuationKind::Field;
  field.field_owner = 0;
  field.field_components = components;
  field.scientific_contract = "migration/q";
  ContinuationType domain;
  domain.kind = ContinuationKind::Domain;
  domain.family = 19;
  domain.capacity = 4;
  auto index = domain;
  index.kind = ContinuationKind::Index;
  auto array = domain;
  array.kind = ContinuationKind::Array;
  ContinuationType integer;
  integer.kind = ContinuationKind::Int64;
  array.children = {integer};
  ContinuationType result;
  result.kind = ContinuationKind::Product;
  result.names = {"state", "domain", "index", "history", "control"};
  result.children = {field, domain, index, array, integer};
  return result;
}

R migration_root(const MultiFab<D>& field, FiniteDomainRef domain) {
  return V::product({V::field(field, 0, "migration/q"), V::domain(domain), V::index(domain, 1),
                     V::array(domain, {V::integer(5), V::integer(9)}),
                     V::integer(INT64_C(9007199254740997))});
}

std::vector<std::vector<std::uint8_t>> migrate_images(
    const std::vector<std::vector<std::uint8_t>>& images, int target_ranks,
    std::uint64_t workspace = 1ULL << 26, bool corrupt_owner = false, int components = 1,
    ContinuationFieldSupport::Ref expected_support = {}) {
  std::vector<std::vector<std::uint8_t>> result;
  for (int rank = 0; rank < target_ranks; ++rank) {
    auto target = migration_field(target_ranks, rank);
    Codec::Prototypes target_fields =
        [&](int owner, const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
      if (bool(support) != bool(expected_support) || (support && *support != *expected_support))
        throw std::invalid_argument("foreign producer support");
      if (owner != 0)
        throw std::invalid_argument("foreign owner");
      return target;
    };
    ContinuationMigrationBudget budget(workspace);
    Codec::DecodeContext aliases(workspace, [&](std::uint64_t bytes) { budget.charge(bytes); });
    std::vector<int> owners;
    for (std::size_t p = 0; p < target.layout().size(); ++p)
      owners.push_back(static_cast<int>(p % images.size()));
    if (corrupt_owner)
      owners.at(0) = 1;
    std::vector<R> values;
    for (std::size_t source = 0; source < images.size(); ++source) {
      ContinuationFieldArchive<D> archive(
          target_fields, owners, false, static_cast<int>(images.size()), static_cast<int>(source),
          budget, [&](const ContinuationFieldSupport::Ref& support) {
            if (!expected_support || *support != *expected_support)
              throw std::invalid_argument("producer topology is not in authenticated table");
            return ContinuationFieldArchive<D>::SourceTopology{owners, false};
          });
      Codec codec(
          migration_type(components), "migration/native", 1ULL << 20,
          [&](int owner, const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
            return archive.prototype(owner, support);
          },
          nullptr, nullptr, nullptr, &aliases,
          [&](const ContinuationType& type, std::span<const std::uint8_t> bytes,
              const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
            return archive.resolve(type, bytes, support);
          });
      auto decoded = codec.decode(images[source]);
      EXPECT_EQ(decoded.publication, 7U);
      EXPECT_EQ(codec.encode(decoded.root, decoded.publication), images[source]);
      values.push_back(std::move(decoded.root));
    }
    auto moved = rematerialize_continuation_fields<D>(values, target_fields, budget);
    EXPECT_EQ(moved->members()[1]->domain_ref(), moved->members()[2]->domain_ref());
    EXPECT_EQ(moved->members()[1]->domain_ref(), moved->members()[3]->domain_ref());
    EXPECT_EQ(moved->members()[4]->integer(), INT64_C(9007199254740997));
    Codec target_codec(migration_type(components), "migration/native", 1ULL << 20, target_fields);
    auto expected_field =
        migration_field(target_ranks, rank, moved->members()[0]->field().ghosts()[0], components);
    auto expected = migration_root(expected_field, moved->members()[1]->domain_ref());
    if (expected_support) {
      auto members = expected->members();
      members[0] = V::with_field_metadata(members[0], {{}, expected_support});
      expected = V::product(std::move(members));
      EXPECT_EQ(*moved->members()[0]->field_support(), *expected_support);
    }
    EXPECT_EQ(target_codec.encode(moved, 7), target_codec.encode(expected, 7));
    result.push_back(target_codec.encode(moved, 7));
  }
  return result;
}
}  // namespace

TEST(ContinuationFieldMigration, NativeCodecPreservesGrownBitsAndAliasesAcrossOneTwoThreeRanks) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {421, 991});
  const auto domain = registry.issue(2, 4, 19);
  auto source = migration_field(1, 0);
  Codec codec(migration_type(), "migration/native", 1ULL << 20,
              [&](int) -> const MultiFab<D>& { return source; });
  const auto initial = codec.encode(migration_root(source, domain), 7);
  const auto two = migrate_images({initial}, 2);
  const auto three = migrate_images(two, 3);
  const auto one = migrate_images(three, 1);
  ASSERT_EQ(one.size(), 1U);
  EXPECT_EQ(one.front(), initial);
}

TEST(ContinuationFieldMigration, RejectsInsufficientWorkspaceAndForeignOwnerWithoutChangingSource) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {421, 992});
  const auto domain = registry.issue(2, 4, 19);
  std::vector<std::vector<std::uint8_t>> images;
  for (int rank = 0; rank < 2; ++rank) {
    auto source = migration_field(2, rank);
    Codec codec(migration_type(), "migration/native", 1ULL << 20,
                [&](int) -> const MultiFab<D>& { return source; });
    images.push_back(codec.encode(migration_root(source, domain), 7));
  }
  const auto before = images;
  EXPECT_THROW(migrate_images(images, 3, 0), std::length_error);
  EXPECT_THROW(migrate_images(images, 3, 1ULL << 26, true), std::invalid_argument);
  EXPECT_EQ(images, before);
  EXPECT_NO_THROW(migrate_images(images, 3));
}

TEST(ContinuationFieldMigration, OutputBudgetDoesNotReplaceInstalledSchemaWorkCapacity) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {421, 993});
  auto field = migration_field(1, 0);
  const auto value = migration_root(field, registry.issue(2, 4, 19));
  Codec codec(
      migration_type(), "migration/native", 1ULL << 20,
      [&](int) -> const MultiFab<D>& { return field; }, nullptr, nullptr, nullptr, nullptr, {}, 16);
  EXPECT_NO_THROW(codec.validate(value));
  EXPECT_THROW(codec.encode(value, 7), std::length_error);
  EXPECT_THROW(codec.collective_identity(value, 7), std::length_error);
}

TEST(ContinuationFieldMigration, AuxiliaryStorageKeepsItsOwnComponentsAndHaloThroughMigration) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {421, 994});
  auto owner = migration_field(1, 0);
  auto auxiliary = migration_field(1, 0, 0, 2);
  Codec codec(migration_type(2), "migration/native", 1ULL << 20,
              [&](int) -> const MultiFab<D>& { return owner; });
  const auto initial = codec.encode(migration_root(auxiliary, registry.issue(2, 4, 19)), 7);
  const auto two = migrate_images({initial}, 2, 1ULL << 26, false, 2);
  const auto three = migrate_images(two, 3, 1ULL << 26, false, 2);
  const auto one = migrate_images(three, 1, 1ULL << 26, false, 2);
  EXPECT_EQ(one.front(), initial);
  const auto restored = codec.decode(initial);
  EXPECT_EQ(restored.root->members()[0]->field().ncomp(), 2);
  EXPECT_EQ(restored.root->members()[0]->field().ghosts()[0], 0);
}

TEST(ContinuationFieldMigration, RejectsForgedHaloBeforeAllocatingNativeStorage) {
  auto owner = migration_field(1, 0);
  auto auxiliary = migration_field(1, 0, 0, 2);
  const auto type = migration_type(2).children.front();
  Codec codec(type, "migration/auxiliary", 1ULL << 20,
              [&](int) -> const MultiFab<D>& { return owner; });
  const auto image = codec.encode(V::field(auxiliary, 0, "migration/q"), 3);
  continuation_detail::Reader input(image);
  EXPECT_EQ(input.text(), "POPSCONT1");
  (void)input.text();
  for (int i = 0; i < 5; ++i)
    (void)input.word();
  const auto first_ghost = input.at;
  auto forged = image;
  for (int i = 0; i < 8; ++i)
    forged[first_ghost + i] = 255;
  EXPECT_THROW(codec.decode(forged), std::invalid_argument);
  EXPECT_NO_THROW(codec.decode(image));
  auto wide = image;
  const auto width = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  for (int i = 0; i < 8; ++i)
    wide[first_ghost + i] = (width >> (8 * i)) & 255;
  EXPECT_THROW(codec.decode(wide), std::invalid_argument);
}

TEST(ContinuationFieldMigration, RetainedFaceExpressionsRepartitionWithOwningCellsAndSharedBases) {
  using F = FieldExchangeExpressions;
  const auto initial_field = migration_field(1, 0);
  F::Basis initial_basis{"native/retained/evaluation/17", 0, {}, {}};
  for (std::size_t p = 0; p < initial_field.layout().size(); ++p) {
    F::Cell cell;
    std::string incidence = "cell";
    for (int d = 0; d < D; ++d) {
      cell.push_back(initial_field.layout()[p].lo[d]);
      incidence += ":" + std::to_string(cell.back());
    }
    ExchangeRecord record;
    record.operation_identity = "authored/face";
    record.occurrence_identity = "authored/history";
    record.evaluation_context = "native/evaluation/17";
    record.quadrature_identity = incidence + "/axis:0/side:1";
    record.trace_axis = 0;
    record.trace_side = 1;
    record.trace_component = 0;
    record.numerical_flux = static_cast<double>(p) + .25;
    record.temporal_weight = .125;
    initial_basis.records.push_back(record);
    initial_basis.cells.push_back(std::move(cell));
  }
  std::vector<std::size_t> canonical;
  for (std::size_t i = 0; i < initial_basis.records.size(); ++i)
    canonical.push_back(i);
  std::sort(canonical.begin(), canonical.end(), [&](auto a, auto b) {
    return initial_basis.records[a].key() < initial_basis.records[b].key();
  });
  F::Basis ordered{initial_basis.identity, initial_basis.program_block, {}, {}};
  for (auto i : canonical) {
    ordered.records.push_back(initial_basis.records[i]);
    ordered.cells.push_back(initial_basis.cells[i]);
  }
  auto basis = std::make_shared<const F::Basis>(std::move(ordered));
  auto expression = std::make_shared<const F::Expression>(F::Expression{true, {{basis, 1.}}});
  auto weighted = std::make_shared<const F::Expression>(F::Expression{true, {{basis, -.5}}});
  std::vector<R> sources{V::product({V::field(initial_field, 0, "migration/q", expression),
                                     V::field(initial_field, 0, "migration/q", weighted)})};
  const auto one_field = migration_type().children.front();
  ContinuationType schema;
  schema.kind = ContinuationKind::Product;
  schema.names = {"state", "history"};
  schema.children = {one_field, one_field};
  for (int ranks : {2, 3, 1}) {
    std::vector<R> targets;
    for (int rank = 0; rank < ranks; ++rank) {
      auto field = migration_field(ranks, rank);
      Codec::Prototypes prototype = [&](int owner) -> const MultiFab<D>& {
        if (owner != 0)
          throw std::invalid_argument("foreign field owner");
        return field;
      };
      ContinuationMigrationBudget budget(1ULL << 26);
      auto migrated = rematerialize_continuation_fields<D>(sources, prototype, budget);
      const auto& a = migrated->members()[0]->field_exchange_expression();
      const auto& b = migrated->members()[1]->field_exchange_expression();
      ASSERT_TRUE(a && b);
      EXPECT_EQ(a->terms.at(0).basis, b->terms.at(0).basis);
      EXPECT_EQ(b->terms.at(0).coefficient, -.5);
      const auto& local = *a->terms.at(0).basis;
      EXPECT_EQ(local.identity, basis->identity);
      ASSERT_EQ(local.cells.size(), field.local_size());
      for (std::size_t i = 0; i < local.cells.size(); ++i) {
        Index<D> cell;
        for (int d = 0; d < D; ++d)
          cell[d] = static_cast<int>(local.cells[i][d]);
        bool owned = false;
        for (std::size_t p = 0; p < field.local_size(); ++p) {
          const auto gid = field.global_index(p);
          if (field.layout()[gid].contains(cell)) {
            owned = true;
            EXPECT_EQ(local.records[i].numerical_flux, static_cast<double>(gid) + .25);
            EXPECT_EQ(local.records[i].temporal_weight, .125);
          }
        }
        EXPECT_TRUE(owned);
      }
      Codec codec(schema, "migration/retained-expressions", 1ULL << 20, prototype);
      auto restored = codec.decode(codec.encode(migrated, 17));
      EXPECT_EQ(restored.root->members()[0]->field_exchange_expression()->terms.at(0).basis,
                restored.root->members()[1]->field_exchange_expression()->terms.at(0).basis);
      targets.push_back(std::move(restored.root));
    }
    sources = std::move(targets);
  }
  ContinuationMigrationBudget insufficient(1);
  EXPECT_THROW(rematerialize_continuation_fields<D>(
                   sources, [&](int) -> const MultiFab<D>& { return initial_field; }, insufficient),
               std::length_error);
  ASSERT_EQ(sources.size(), 1U);
  EXPECT_EQ(
      sources[0]->members()[0]->field_exchange_expression()->terms.at(0).basis->records.size(),
      basis->records.size());
}

TEST(ContinuationFieldMigration, ProducerSupportSurvivesOneTwoThreeOneWithoutTimeRelabelling) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {721, 192});
  const auto domain = registry.issue(2, 4, 19);
  auto field = migration_field(1, 0);
  auto support = std::make_shared<const ContinuationFieldSupport>(
      ContinuationFieldSupport{"same-program", "history/time=0.125/point=stage-2", 0, 2, 7, 19});
  auto root = migration_root(field, domain);
  auto members = root->members();
  members[0] = V::with_field_metadata(members[0], {{}, support});
  root = V::product(std::move(members));
  Codec codec(migration_type(), "migration/native", 1ULL << 20,
              [&](int owner, const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
                if (owner != 0 || !origin || *origin != *support)
                  throw std::invalid_argument("foreign producer support");
                return field;
              });
  const auto initial = codec.encode(root, 7);
  auto two = migrate_images({initial}, 2, 1ULL << 26, false, 1, support);
  auto three = migrate_images(two, 3, 1ULL << 26, false, 1, support);
  const auto one = migrate_images(three, 1, 1ULL << 26, false, 1, support);
  EXPECT_EQ(one.at(0), initial);
  EXPECT_THROW(migrate_images({initial}, 2), std::invalid_argument);
  EXPECT_THROW(migrate_images({initial}, 2, 1, false, 1, support), std::length_error);
  auto foreign = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      support->program_identity, support->origin_point, support->program_owner, 1,
      support->topology_epoch, support->materialization_generation, support->field_provider});
  EXPECT_THROW(migrate_images({initial}, 2, 1ULL << 26, false, 1, foreign), std::invalid_argument);
  EXPECT_EQ(codec.encode(root, 7), initial);
}

TEST(ContinuationFieldMigration, SameOwnerDistinctProducerDescriptorsNeverAliasArchivePrototypes) {
  auto field = migration_field(1, 0);
  auto a = std::make_shared<const ContinuationFieldSupport>(
      ContinuationFieldSupport{"program", "point/time=0.125", 0, 0, 7, 19});
  auto b = std::make_shared<const ContinuationFieldSupport>(
      ContinuationFieldSupport{"program", "point/time=0.25", 0, 1, 7, 19});
  auto type = migration_type().children[0];
  ContinuationType schema;
  schema.kind = ContinuationKind::Product;
  schema.names = {"old", "other-level"};
  schema.children = {type, type};
  auto fields = [&](int owner, const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
    if (owner != 0 || !support || (*support != *a && *support != *b))
      throw std::invalid_argument("unauthenticated producer descriptor");
    return field;
  };
  Codec source(schema, "two-producers", 1ULL << 20, fields);
  auto root = V::product({V::field(field, 0, type.scientific_contract, {}, a),
                          V::field(field, 0, type.scientific_contract, {}, b)});
  const auto bytes = source.encode(root, 9);
  ContinuationMigrationBudget budget(1ULL << 24);
  std::vector<int> owners(field.layout().size(), 0);
  ContinuationFieldArchive<D> archive(
      fields, owners, false, 1, 0, budget, [&](const ContinuationFieldSupport::Ref& support) {
        if (*support != *a && *support != *b)
          throw std::invalid_argument("producer absent from source forest");
        return ContinuationFieldArchive<D>::SourceTopology{owners, false};
      });
  Codec target(
      schema, "two-producers", 1ULL << 20,
      [&](int owner, const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
        return archive.prototype(owner, support);
      },
      nullptr, nullptr, nullptr, nullptr,
      [&](const ContinuationType& item, std::span<const std::uint8_t> payload,
          const ContinuationFieldSupport::Ref& support) -> const MultiFab<D>& {
        return archive.resolve(item, payload, support);
      });
  const auto decoded = target.decode(bytes);
  EXPECT_NE(&archive.prototype(0, a), &archive.prototype(0, b));
  EXPECT_EQ(*decoded.root->members()[0]->field_support(), *a);
  EXPECT_EQ(*decoded.root->members()[1]->field_support(), *b);
  EXPECT_EQ(target.encode(decoded.root, 9), bytes);
  auto forged = bytes;
  const std::string original = "point/time=0.25";
  const auto found = std::search(forged.begin(), forged.end(), original.begin(), original.end());
  ASSERT_NE(found, forged.end());
  *(found + original.size() - 1) = '6';
  EXPECT_THROW(target.decode(forged), std::invalid_argument);
  EXPECT_EQ(target.encode(decoded.root, 9), bytes);
}

TEST(ContinuationFieldMigration, IndependentProviderAndClosedRepresentationSurviveOneTwoThreeOne) {
  auto type = migration_type().children[0];
  type.field_owner = -1;
  type.field_provider = "problem:A/unknown:potential/components:1";
  auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      "program", "history/time=0.125/invocation=4", -1, 2, 7, 19, type.field_provider});
  auto obligation = std::make_shared<const FieldRepresentationObligation>(
      FieldRepresentationObligation{"transform:T/closed:a=1", "evaluation:4"});
  auto initial_field = migration_field(1, 0);
  auto prototypes = [&](const ContinuationType& selected,
                        const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
    if (selected != type || !origin || *origin != *support)
      throw std::invalid_argument("foreign independent field authority");
    return initial_field;
  };
  Codec initial_codec(type, "migration/provider", 1ULL << 20, prototypes);
  const auto value = V::field(initial_field, -1, type.scientific_contract, {}, support,
                              type.field_provider, obligation);
  const auto initial = initial_codec.encode(value, 7);
  auto migrate = [&](const std::vector<std::vector<std::uint8_t>>& images, int ranks) {
    std::vector<std::vector<std::uint8_t>> result;
    for (int rank = 0; rank < ranks; ++rank) {
      auto target = migration_field(ranks, rank);
      Codec::Prototypes targets =
          [&](const ContinuationType& selected,
              const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
        if (selected != type || !origin || *origin != *support)
          throw std::invalid_argument("foreign independent field target");
        return target;
      };
      std::vector<int> owners;
      for (std::size_t patch = 0; patch < target.layout().size(); ++patch)
        owners.push_back(static_cast<int>(patch % images.size()));
      ContinuationMigrationBudget budget(1ULL << 26);
      std::vector<R> values;
      for (std::size_t source = 0; source < images.size(); ++source) {
        ContinuationFieldArchive<D> archive(
            targets, owners, false, static_cast<int>(images.size()), static_cast<int>(source),
            budget, [&](const ContinuationFieldSupport::Ref& origin) {
              if (!origin || *origin != *support)
                throw std::invalid_argument("foreign source topology");
              return ContinuationFieldArchive<D>::SourceTopology{owners, false};
            });
        Codec codec(
            type, "migration/provider", 1ULL << 20,
            [&](const ContinuationType& selected, const ContinuationFieldSupport::Ref& origin)
                -> const MultiFab<D>& { return archive.prototype(selected, origin); },
            nullptr, nullptr, nullptr, nullptr,
            [&](const ContinuationType& selected, std::span<const std::uint8_t> bytes,
                const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
              return archive.resolve(selected, bytes, origin);
            });
        auto decoded = codec.decode(images[source]);
        EXPECT_EQ(codec.encode(decoded.root, 7), images[source]);
        values.push_back(decoded.root);
      }
      auto moved = rematerialize_continuation_fields<D>(values, targets, budget);
      EXPECT_EQ(moved->field_provider(), type.field_provider);
      EXPECT_TRUE(moved->field_representation());
      EXPECT_EQ(*moved->field_representation(), *obligation);
      EXPECT_EQ(*moved->field_support(), *support);
      auto expected = V::field(target, -1, type.scientific_contract, {}, support,
                               type.field_provider, obligation);
      Codec output(type, "migration/provider", 1ULL << 20, targets);
      EXPECT_EQ(output.encode(moved, 7), output.encode(expected, 7));
      result.push_back(output.encode(moved, 7));
    }
    return result;
  };
  const auto two = migrate({initial}, 2), three = migrate(two, 3), one = migrate(three, 1);
  EXPECT_EQ(one.at(0), initial);
  EXPECT_EQ(initial_codec.encode(value, 7), initial);
}

TEST(ContinuationFieldMigration, IndependentFieldRejectsCrossRankProviderAndClosedCaptureChanges) {
  auto type = migration_type().children[0];
  type.field_owner = -1;
  type.field_provider = "problem:A/unknown:potential";
  const auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      "program", "historical-evaluation:7", -1, 1, 2, 3, type.field_provider});
  const auto representation = std::make_shared<const FieldRepresentationObligation>(
      FieldRepresentationObligation{"transform:T/capture:a=1", "producer:7"});
  auto first = migration_field(2, 0), second = migration_field(2, 1),
       target = migration_field(3, 0);
  const auto a = V::field(first, -1, type.scientific_contract, {}, support, type.field_provider,
                          representation);
  const auto b = V::field(second, -1, type.scientific_contract, {}, support, type.field_provider,
                          representation);
  int prototype_calls = 0;
  Codec::Prototypes targets =
      [&](const ContinuationType& selected,
          const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
    ++prototype_calls;
    if (selected != type || !origin || *origin != *support)
      throw std::invalid_argument("foreign target field authority");
    return target;
  };
  const auto changed = V::with_field_representation(
      b, std::make_shared<const FieldRepresentationObligation>(
             FieldRepresentationObligation{"transform:T/capture:a=2", "producer:7"}));
  std::vector<R> mismatch{a, changed};
  ContinuationMigrationBudget first_budget(1ULL << 26);
  EXPECT_THROW(rematerialize_continuation_fields<D>(mismatch, targets, first_budget),
               std::invalid_argument);
  EXPECT_EQ(prototype_calls, 0);
  EXPECT_EQ(a->field_representation(), representation);
  EXPECT_EQ(b->field_representation(), representation);
  for (const std::string provider : {"problem:B/unknown:potential", "problem:A/unknown:other"}) {
    auto foreign_support =
        std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
            support->program_identity, support->origin_point, -1, support->producer_level,
            support->topology_epoch, support->materialization_generation, provider});
    mismatch[1] = V::field(second, -1, type.scientific_contract, {}, foreign_support, provider,
                           representation);
    ContinuationMigrationBudget budget(1ULL << 26);
    EXPECT_THROW(rematerialize_continuation_fields<D>(mismatch, targets, budget),
                 std::invalid_argument);
    EXPECT_EQ(prototype_calls, 0);
  }
  std::vector<R> valid{a, b};
  ContinuationMigrationBudget valid_budget(1ULL << 26);
  EXPECT_NO_THROW(rematerialize_continuation_fields<D>(valid, targets, valid_budget));
  EXPECT_EQ(prototype_calls, 1);
}
