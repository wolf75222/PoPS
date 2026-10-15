/// @file
/// @brief Authenticated native rematerialization of complete finite AMR
/// continuation forests.

#pragma once

#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/continuation_authority.hpp>
#include <pops/runtime/program/continuation_field_migration.hpp>

namespace pops::runtime::program {

/// This is a local prepare operation. The caller has authenticated every full
/// old-rank accepted image under RSA1 and the same installed program_hash, and
/// converges failures before publishing. Neither accepted stores nor finite
/// issuers are mutated here.
template <int Dim>
std::vector<AmrProgramContinuationLevel> rematerialize_continuation_forest_fields(
    std::span<const std::vector<AmrProgramContinuationLevel>> ranked,
    const std::map<int, ContinuationStore<Dim>>& installed,
    const std::function<typename ContinuationCodec<Dim>::Prototypes(int)>& targets,
    std::span<const std::vector<int>> source_owners, std::span<const std::string> source_modes,
    std::uint64_t target_epoch, std::uint64_t target_generation, double accepted_time,
    std::int64_t accepted_step, std::uint64_t workspace_byte_capacity) {
  using Store = ContinuationStore<Dim>;
  using Ref = ContinuationValueRef<Dim>;
  using Archive = ContinuationFieldArchive<Dim>;
  if (ranked.empty() || ranked.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !targets || ranked.front().size() < 2 || ranked.front().size() != installed.size() ||
      source_owners.size() + 1 != installed.size() || source_modes.size() != source_owners.size())
    throw std::invalid_argument(
        "continuation migration requires every source rank and forest member");
  ContinuationMigrationBudget budget(workspace_byte_capacity);
  typename ContinuationCodec<Dim>::DecodeContext authorities(
      workspace_byte_capacity, [&](std::uint64_t bytes) { budget.charge(bytes); });
  const auto count = installed.size();
  budget.charge(count, sizeof(AmrProgramContinuationLevel));
  budget.charge(ranked.size(), sizeof(std::vector<AmrProgramContinuationLevel>));
  std::vector<std::vector<AmrProgramContinuationLevel>> common;
  common.reserve(ranked.size());
  for (const auto& records : ranked) {
    if (records.size() != count)
      throw std::invalid_argument("continuation migration archive omits a forest member");
    budget.charge(count, sizeof(AmrProgramContinuationLevel));
    std::vector<AmrProgramContinuationLevel> row;
    row.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto& record = records[i];
      const auto& reference = ranked.front()[i];
      if (record.level != static_cast<int>(i) - 1 || record.level != reference.level ||
          record.schema_identity != reference.schema_identity ||
          record.topology_epoch != reference.topology_epoch ||
          record.materialization_generation != reference.materialization_generation ||
          record.domain_high_water != reference.domain_high_water ||
          record.collective_identity != reference.collective_identity)
        throw std::invalid_argument("continuation migration source control differs between ranks");
      const auto& configuration = installed.at(record.level).configuration();
      if (record.schema_identity != configuration.identity || record.image.empty() ||
          record.image.size() > configuration.byte_capacity)
        throw std::invalid_argument(
            "continuation migration source violates "
            "its installed schema/budget");
      budget.charge(record.schema_identity.size() + record.collective_identity.size());
      row.push_back({record.level,
                     record.topology_epoch,
                     record.materialization_generation,
                     record.domain_high_water,
                     record.schema_identity,
                     {},
                     {}});
    }
    common.push_back(std::move(row));
  }
  std::vector<AmrProgramContinuationLevel> result;
  result.reserve(count);
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto level = static_cast<int>(slot) - 1;
    const auto& store = installed.at(level);
    const auto& configuration = store.configuration();
    const auto target_fields = targets(level);
    budget.charge(ranked.size(),
                  sizeof(std::unique_ptr<Archive>) + sizeof(Archive) + sizeof(Store) + sizeof(Ref));
    std::vector<std::unique_ptr<Archive>> archives;
    std::vector<Store> source_stores;
    std::vector<Ref> roots;
    archives.reserve(ranked.size());
    source_stores.reserve(ranked.size());
    roots.reserve(ranked.size());
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
      const auto& record = ranked[rank][slot];
      const std::span<const int> owners =
          level < 0 ? std::span<const int>{} : std::span<const int>(source_owners[level]);
      const bool replicated = level < 0 || source_modes[level] == "replicated";
      if (level >= 0 && source_modes[level] != "partitioned" && !replicated)
        throw std::invalid_argument("continuation migration source distribution is unknown");
      // Source image bytes bound codec-owned decoded payloads. Their native
      // object overhead is accounted separately by the structural decoder; this
      // workspace adds archive prototypes and conversion temporaries to that
      // per-store bound.
      budget.charge(record.image.size());
      archives.push_back(std::make_unique<Archive>(target_fields, owners, replicated,
                                                   static_cast<int>(ranked.size()),
                                                   static_cast<int>(rank), budget));
      Archive& archive = *archives.back();
      auto fields = [&archive](int owner) -> const MultiFab<Dim>& {
        return archive.prototype(owner);
      };
      auto resolve = [&archive](const ContinuationType& type,
                                std::span<const std::uint8_t> bytes) -> const MultiFab<Dim>& {
        return archive.resolve(type, bytes);
      };
      // No source communicator is fabricated. Distributed values require the
      // later native GID plan and are rejected by the codec when no source
      // ExecutionLane is supplied.
      auto prepared = store.prepare_restart(record.image, fields, nullptr, &authorities, resolve);
      if (prepared.distributed_registry ||
          (prepared.closure &&
           (!prepared.closure->arrays.empty() || !prepared.closure->domains.empty())))
        throw std::invalid_argument(
            "continuation migration requires a distributed target GID plan");
      if (prepared.domain_high_water > record.domain_high_water)
        throw std::invalid_argument("continuation migration regresses the finite allocator");
      prepared.domain_high_water = record.domain_high_water;
      Store decoded = store;
      decoded.publish(std::move(prepared));
      decoded.require_frontier(accepted_time, accepted_step);
      // A canonical source re-encoding must have this exact size. Bound the
      // writer before allocating its comparison copy, then retain its charge
      // for the whole operation.
      budget.charge(record.image.size());
      auto source_codec = ContinuationCodec<Dim>(
          configuration.type, configuration.identity, configuration.byte_capacity, fields, nullptr,
          decoded.distributed_registry().get(), decoded.closure().get(), nullptr, {},
          record.image.size());
      if (source_codec.encode(decoded.read(), decoded.publication()) != record.image)
        throw std::invalid_argument("continuation migration source image is not canonical");
      auto identity_codec = ContinuationCodec<Dim>(
          configuration.type, configuration.identity, configuration.byte_capacity, fields, nullptr,
          decoded.distributed_registry().get(), decoded.closure().get(), nullptr, {},
          budget.remaining());
      const auto identity =
          identity_codec.collective_identity(decoded.read(), decoded.publication());
      budget.charge(identity.size());
      common[rank][slot].collective_identity = identity;
      if (level >= 0 && identity != record.collective_identity)
        throw std::invalid_argument(
            "continuation migration child common "
            "identity is not authenticated");
      if (!source_stores.empty() && decoded.publication() != source_stores.front().publication())
        throw std::invalid_argument(
            "continuation migration publication differs between source ranks");
      roots.push_back(decoded.read());
      source_stores.push_back(std::move(decoded));
    }
    Ref moved = rematerialize_continuation_fields<Dim>(roots, target_fields, budget);
    // Encode against the genuine target prototype; this is also the final
    // destination storage/type/capacity check. The source publication and exact
    // finite allocator remain.
    const auto& source = source_stores.front();
    budget.charge(configuration.identity.size());
    auto codec = ContinuationCodec<Dim>(
        configuration.type, configuration.identity, configuration.byte_capacity, target_fields,
        nullptr, nullptr, source.closure().get(), nullptr, {}, budget.remaining());
    const auto identity = codec.collective_identity(moved, source.publication());
    budget.charge(identity.size());
    auto output_codec = ContinuationCodec<Dim>(
        configuration.type, configuration.identity, configuration.byte_capacity, target_fields,
        nullptr, nullptr, source.closure().get(), nullptr, {}, budget.remaining());
    auto image = output_codec.encode(moved, source.publication());
    budget.charge(image.size());
    result.push_back({level, target_epoch, target_generation, source.domain_high_water(),
                      configuration.identity, identity, std::move(image)});
  }
  for (std::size_t rank = 0; rank < ranked.size(); ++rank)
    if (amr_continuation_forest_identity(common[rank], common[rank].front().collective_identity) !=
        ranked[rank].front().collective_identity)
      throw std::invalid_argument(
          "continuation migration parent does not seal its complete forest");
  // Common source identities were validated before deriving the target
  // collective authority.
  result.front().collective_identity =
      amr_continuation_forest_identity(result, result.front().collective_identity);
  return result;
}

}  // namespace pops::runtime::program
