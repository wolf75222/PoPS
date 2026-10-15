#pragma once

#include <pops/runtime/program/continuation_forest_migration.hpp>
#include <pops/runtime/program/continuation_distributed_migration.hpp>
#include <pops/runtime/program/continuation_closure.hpp>

namespace pops::runtime::program {

/// Reconstruct the complete accepted forest against actual target prototypes and
/// lane. Parsing and RSA1 validation precede native issuer construction. This
/// method owns a prepare transaction only; the caller still authenticates and
/// atomically publishes the resulting checkpoint through ordinary restart.
template <int Dim>
std::vector<AmrProgramContinuationLevel> rematerialize_continuation_forest(
    std::span<const std::vector<AmrProgramContinuationLevel>> ranked,
    const std::map<int, ContinuationStore<Dim>>& installed,
    const std::function<typename ContinuationCodec<Dim>::Prototypes(int)>& targets,
    std::span<const std::vector<int>> source_owners, std::span<const std::string> source_modes,
    std::uint64_t target_epoch, std::uint64_t target_generation, double accepted_time,
    std::int64_t accepted_step, std::uint64_t workspace_byte_capacity, const ExecutionLane& lane) {
  using Codec = ContinuationCodec<Dim>;
  using Ref = ContinuationValueRef<Dim>;
  using Archive = ContinuationFieldArchive<Dim>;
  using Image = typename Codec::ArchiveImage;
  using Plan = ContinuationDistributedMigration<Dim>;
  struct Source {
    std::unique_ptr<Archive> fields;
    std::unique_ptr<Codec> codec;
    Image image;
  };
  ContinuationMigrationBudget budget(workspace_byte_capacity);
  typename Codec::DecodeContext authorities(workspace_byte_capacity,
                                            [&](std::uint64_t bytes) { budget.charge(bytes); });
  Plan plan(budget);
  std::vector<Source> sources;
  std::vector<typename Plan::Input> inputs;
  std::vector<std::vector<AmrProgramContinuationLevel>> common;
  std::vector<AmrProgramContinuationLevel> result;
  const auto count = installed.size();
  distributed::phase(lane, [&] {
    if (ranked.empty() || ranked.size() > std::size_t(INT_MAX) || !targets || count < 2 ||
        source_owners.size() + 1 != count || source_modes.size() != source_owners.size() ||
        count > SIZE_MAX / ranked.size())
      throw std::invalid_argument(
          "forest migration requires the complete source rank/member table");
    const auto total = count * ranked.size();
    budget.charge(total, sizeof(Source) + sizeof(typename Plan::Input));
    budget.charge(ranked.size(), sizeof(std::vector<AmrProgramContinuationLevel>));
    sources.reserve(total);
    inputs.reserve(total);
    common.reserve(ranked.size());
    budget.charge(count, sizeof(AmrProgramContinuationLevel));
    result.reserve(count);
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
      if (ranked[rank].size() != count)
        throw std::invalid_argument("forest migration archive omits a parent or child image");
      budget.charge(count, sizeof(AmrProgramContinuationLevel));
      common.emplace_back();
      common.back().reserve(count);
      for (std::size_t slot = 0; slot < count; ++slot) {
        const int level = static_cast<int>(slot) - 1;
        const auto& record = ranked[rank][slot];
        const auto& reference = ranked.front()[slot];
        const auto& configuration = installed.at(level).configuration();
        if (record.level != level || record.schema_identity != configuration.identity ||
            record.schema_identity != reference.schema_identity ||
            record.topology_epoch != reference.topology_epoch ||
            record.materialization_generation != reference.materialization_generation ||
            record.domain_high_water != reference.domain_high_water ||
            record.collective_identity != reference.collective_identity || record.image.empty() ||
            record.image.size() > configuration.byte_capacity)
          throw std::invalid_argument(
              "forest migration source schema/control/budget is inconsistent");
        const std::span<const int> owners =
            level < 0 ? std::span<const int>{} : std::span<const int>(source_owners[level]);
        const bool replicated = level < 0 || source_modes[level] == "replicated";
        if (level >= 0 && !replicated && source_modes[level] != "partitioned")
          throw std::invalid_argument("forest migration has unknown source Field distribution");
        budget.charge(record.image.size());
        budget.charge(sizeof(Archive) + sizeof(Codec));
        sources.emplace_back();
        auto& source = sources.back();
        source.fields = std::make_unique<Archive>(targets(level), owners, replicated,
                                                  static_cast<int>(ranked.size()),
                                                  static_cast<int>(rank), budget);
        auto* archive = source.fields.get();
        source.codec = std::make_unique<Codec>(
            configuration.type, configuration.identity, configuration.byte_capacity,
            [archive](int owner) -> const MultiFab<Dim>& { return archive->prototype(owner); },
            nullptr, nullptr, nullptr, &authorities,
            [archive](const ContinuationType& type, std::span<const std::uint8_t> bytes)
                -> const MultiFab<Dim>& { return archive->resolve(type, bytes); });
        source.image = source.codec->decode_archive(
            record.image, static_cast<int>(rank), static_cast<int>(ranked.size()),
            [&](std::uint64_t bytes) { budget.charge(bytes); });
        const auto& image = source.image;
        if (image.max_domain_version > record.domain_high_water)
          throw std::invalid_argument("forest migration regresses its finite allocator high-water");
        const auto at = [&](const ContinuationPath& path) -> const typename Codec::ArchiveNode& {
          auto index = image.root;
          for (auto ordinal : path)
            index = image.nodes.at(index).children.at(ordinal);
          return image.nodes.at(index);
        };
        const auto& time = at(configuration.time_path);
        const auto& step = at(configuration.step_path);
        if (!time.leaf || !step.leaf || time.kind != ContinuationKind::Float64 ||
            step.kind != ContinuationKind::Int64 || step.leaf->integer() != accepted_step ||
            std::bit_cast<std::uint64_t>(time.leaf->real()) !=
                std::bit_cast<std::uint64_t>(accepted_time))
          throw std::invalid_argument(
              "forest migration source frontier differs from accepted state");
        auto identity = source.codec->archive_collective_identity(
            image, budget.remaining(), [&](std::uint64_t bytes) { budget.charge(bytes); });
        budget.charge(configuration.identity.size());
        common.back().push_back({level,
                                 record.topology_epoch,
                                 record.materialization_generation,
                                 record.domain_high_water,
                                 configuration.identity,
                                 std::move(identity),
                                 {}});
        if (level >= 0 && common.back().back().collective_identity != record.collective_identity)
          throw std::invalid_argument("forest migration child does not match its sealed identity");
        if (rank > 0) {
          const auto& first = sources.at(slot).image;
          const auto primary =
              [](const Image& image) -> std::optional<std::array<std::uint64_t, 2>> {
            if (!image.primary_registry)
              return std::nullopt;
            return image.registries.at(*image.primary_registry).issuer;
          };
          if (first.publication != image.publication ||
              ((first.realm && first.realm->synchronized()) ||
               (image.realm && image.realm->synchronized())) &&
                  first.realm != image.realm ||
              primary(first) != primary(image))
            throw std::invalid_argument(
                "forest migration allocator/publication differs between ranks");
        }
        inputs.push_back({&image, source.codec.get(), configuration.byte_capacity,
                          configuration.distributed_storage_bytes,
                          configuration.distributed_storage_versions});
      }
      if (amr_continuation_forest_identity(common.back(),
                                           common.back().front().collective_identity) !=
          ranked[rank].front().collective_identity)
        throw std::invalid_argument(
            "forest migration parent does not seal its complete child forest");
    }
    plan.prepare(inputs);
  });
  plan.materialize(lane);
  for (std::size_t slot = 0; slot < count; ++slot) {
    const int level = static_cast<int>(slot) - 1;
    const auto& configuration = installed.at(level).configuration();
    const auto& source = sources.at(slot).image;
    Ref moved;
    distributed::phase(lane, [&] {
      budget.charge(ranked.size(), sizeof(Ref));
      std::vector<Ref> roots;
      roots.reserve(ranked.size());
      for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
        const auto& image = sources.at(rank * count + slot).image;
        roots.push_back(plan.value(image, image.root));
      }
      moved = rematerialize_continuation_fields<Dim>(roots, targets(level), budget);
    });
    auto primary = plan.primary(source);
    auto closure = continuation_collective_closure<Dim>(
        moved, primary, lane, configuration.byte_capacity, source.realm, plan.authorities());
    distributed::phase(lane, [&] {
      Codec codec(configuration.type, configuration.identity, configuration.byte_capacity,
                  targets(level), &lane, primary.get(), closure.get(), nullptr, {},
                  budget.remaining());
      auto identity = codec.collective_identity(moved, source.publication);
      budget.charge(identity.size());
      Codec output(configuration.type, configuration.identity, configuration.byte_capacity,
                   targets(level), &lane, primary.get(), closure.get(), nullptr, {},
                   budget.remaining());
      auto bytes = output.encode(moved, source.publication);
      budget.charge(bytes.size());
      budget.charge(configuration.identity.size());
      result.push_back({level, target_epoch, target_generation,
                        ranked.front()[slot].domain_high_water, configuration.identity,
                        std::move(identity), std::move(bytes)});
    });
  }
  distributed::phase(lane, [&] {
    result.front().collective_identity =
        amr_continuation_forest_identity(result, result.front().collective_identity);
  });
  return result;
}
}  // namespace pops::runtime::program
