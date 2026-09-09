#pragma once

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/amr_history_flux_snapshot.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

namespace pops::runtime::program::history_flux {

template <int Dim>
using SnapshotMap = std::map<std::string, std::shared_ptr<const Snapshot<Dim>>>;

// Storage ownership is deliberately absent: the same physical archive survives rank changes.
template <int Dim, class Writer>
void write_physical_metadata(Writer& out, const Snapshot<Dim>& snapshot) {
  out.string(snapshot.source_identity);
  out.i32(snapshot.level);
  out.i32(snapshot.components);
  for (int axis = 0; axis < Dim; ++axis) {
    out.i32(snapshot.domain.lo[axis]);
    out.i32(snapshot.domain.hi[axis]);
    out.real(snapshot.cell_size[axis]);
  }
  out.size(snapshot.patches.size());
  for (const auto& patch : snapshot.patches)
    for (int axis = 0; axis < Dim; ++axis) {
      out.i32(patch.lo[axis]);
      out.i32(patch.hi[axis]);
    }
}

template <int Dim>
Snapshot<Dim> read_physical_metadata(checkpoint_detail::Reader& in, int max_levels) {
  Snapshot<Dim> result;
  result.source_identity = in.string();
  result.level = in.i32();
  result.components = in.i32();
  if (result.level < 0 || result.level >= max_levels || result.components <= 0)
    throw std::invalid_argument("AMR history flux archive has foreign level/components");
  for (int axis = 0; axis < Dim; ++axis) {
    result.domain.lo[axis] = in.i32();
    result.domain.hi[axis] = in.i32();
    result.cell_size[axis] = in.real();
  }
  const auto count = in.size(2 * Dim * sizeof(std::uint64_t));
  if (count == 0 || count > detail::points(result.domain))
    throw std::invalid_argument("AMR history flux archive patch count exceeds its domain");
  result.patches.resize(count);
  for (auto& patch : result.patches)
    for (int axis = 0; axis < Dim; ++axis) {
      patch.lo[axis] = in.i32();
      patch.hi[axis] = in.i32();
    }
  return result;
}

template <int Dim>
std::string patch_digest(const OwnedPatch<Dim>& patch) {
  checkpoint_detail::Writer out;
  out.u64(patch.global_patch);
  for (const auto& axis : patch.density) {
    out.size(axis.size());
    for (const Real value : axis)
      out.real(static_cast<double>(value));
  }
  return identity::sha256_hex(std::move(out).take());
}

template <int Dim>
std::string content_identity(const Snapshot<Dim>& snapshot,
                             const std::vector<std::string>& patch_digests) {
  if (patch_digests.size() != snapshot.patches.size())
    throw std::invalid_argument("AMR history flux digest omits source patches");
  checkpoint_detail::Writer out;
  out.string("pops.amr.history-physical-face-snapshot.v1");
  write_physical_metadata(out, snapshot);
  for (const auto& digest : patch_digests) {
    if (digest.size() != 64 || !std::all_of(digest.begin(), digest.end(), [](char value) {
          return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        }))
      throw std::invalid_argument("AMR history flux patch digest is incomplete");
    out.string(digest);
  }
  return "pops.amr.history-physical-face-snapshot.v1:sha256:" +
         identity::sha256_hex(std::move(out).take());
}

template <int Dim, class Writer>
void write_descriptor(Writer& out, const std::shared_ptr<const Snapshot<Dim>>& snapshot,
                      int max_levels) {
  out.u64(snapshot ? (snapshot->parent ? 2u : 1u) : 0u);
  if (!snapshot)
    return;
  if (max_levels <= 0)
    throw std::invalid_argument("AMR history flux descriptor exceeds authored levels");
  out.string(snapshot->identity);
  write_physical_metadata(out, *snapshot);
  if (snapshot->parent) {
    for (int ratio : snapshot->ratio)
      out.i32(ratio);
    write_descriptor(out, snapshot->parent, max_levels - 1);
  }
}

template <int Dim>
std::string projection_identity(const Snapshot<Dim>& snapshot) {
  if (!snapshot.parent)
    throw std::invalid_argument("AMR history flux projection identity lacks its parent");
  checkpoint_detail::Writer out;
  out.string("pops.amr.history-face-projection.v1");
  out.string(snapshot.parent->identity);
  write_physical_metadata(out, snapshot);
  for (int ratio : snapshot.ratio)
    out.i32(ratio);
  return "pops.amr.history-face-projection.v1:sha256:" +
         identity::sha256_hex(std::move(out).take());
}

template <int Dim>
std::shared_ptr<const Snapshot<Dim>> read_descriptor(checkpoint_detail::Reader& in,
                                                     const SnapshotMap<Dim>& raw, int max_levels,
                                                     int remaining_levels = -1) {
  if (remaining_levels == -1)
    remaining_levels = max_levels;
  const auto kind = in.u64();
  if (kind == 0)
    return {};
  if ((kind != 1 && kind != 2) || remaining_levels <= 0)
    throw std::invalid_argument("AMR history flux descriptor has invalid bounded lineage");
  const auto token = in.string();
  auto result = read_physical_metadata<Dim>(in, max_levels);
  result.identity = token;
  if (kind == 1) {
    const auto found = raw.find(token);
    if (found == raw.end() || !found->second || found->second->parent)
      throw std::invalid_argument("AMR history flux checkpoint lacks its physical face archive");
    checkpoint_detail::Writer expected, actual;
    write_physical_metadata(expected, result);
    write_physical_metadata(actual, *found->second);
    if (std::move(expected).take() != std::move(actual).take())
      throw std::invalid_argument("AMR history flux archive differs from its source descriptor");
    return found->second;
  }
  for (auto& ratio : result.ratio)
    ratio = in.i32();
  result.parent = read_descriptor(in, raw, max_levels, remaining_levels - 1);
  if (!result.parent)
    throw std::invalid_argument("AMR history flux projection lacks its parent snapshot");
  result.rank_count = result.parent->rank_count;
  result.local_rank = result.parent->local_rank;
  // Projection nodes contain no owned samples. Only raw source ownership controls queries.
  result.owners.assign(result.patches.size(), 0);
  certify_snapshot(result, max_levels);
  if (projection_identity(result) != token)
    throw std::invalid_argument("AMR history flux projection differs from its lineage digest");
  return std::make_shared<const Snapshot<Dim>>(std::move(result));
}

template <int Dim>
std::vector<std::uint8_t> encode_shard(const SnapshotMap<Dim>& snapshots, int max_levels) {
  if (snapshots.empty())
    return {};
  checkpoint_detail::Writer out;
  out.u64(UINT64_C(0x504f505348465831));  // POPSHFX1, owned physical face archives.
  out.size(snapshots.size());
  for (const auto& [token, source] : snapshots) {
    if (!source || source->parent || source->identity != token)
      throw std::invalid_argument("AMR history flux export contains a non-physical source");
    validate_snapshot(*source, max_levels);
    out.string(token);
    write_physical_metadata(out, *source);
    out.i32(source->rank_count);
    out.i32(source->local_rank);
    out.size(source->owners.size());
    for (int owner : source->owners)
      out.i32(owner);
    out.size(source->owned.size());
    for (const auto& patch : source->owned) {
      out.u64(patch.global_patch);
      for (const auto& axis : patch.density) {
        out.size(axis.size());
        for (Real value : axis)
          out.real(static_cast<double>(value));
      }
    }
  }
  return std::move(out).take();
}

template <int Dim>
SnapshotMap<Dim> decode_shards(const std::vector<std::vector<std::uint8_t>>& shards,
                               int source_ranks, int target_ranks, int target_rank, int max_levels,
                               std::size_t shard_budget) {
  if (source_ranks <= 0 || static_cast<std::size_t>(source_ranks) != shards.size() ||
      target_ranks <= 0 || target_rank < 0 || target_rank >= target_ranks)
    throw std::invalid_argument("AMR history flux archive rank envelope is invalid");
  if (std::all_of(shards.begin(), shards.end(), [](const auto& row) { return row.empty(); }))
    return {};
  struct Complete {
    Snapshot<Dim> metadata;
    std::map<std::size_t, OwnedPatch<Dim>> patches;
  };
  std::map<std::string, Complete> complete;
  std::vector<std::string> expected_tokens;
  for (int rank = 0; rank < source_ranks; ++rank) {
    const auto& bytes = shards[static_cast<std::size_t>(rank)];
    if (bytes.empty() || bytes.size() > shard_budget)
      throw std::invalid_argument("AMR history flux archive shard exceeds its sealed envelope");
    checkpoint_detail::Reader in(bytes);
    if (in.u64() != UINT64_C(0x504f505348465831))
      throw std::invalid_argument("AMR history flux archive has an unsupported codec");
    const auto count = in.size(sizeof(std::uint64_t));
    std::vector<std::string> tokens;
    for (std::size_t index = 0; index < count; ++index) {
      auto token = in.string();
      if (token.empty() || (!tokens.empty() && tokens.back() >= token))
        throw std::invalid_argument("AMR history flux archive has noncanonical source identities");
      tokens.push_back(token);
      auto source = read_physical_metadata<Dim>(in, max_levels);
      source.identity = token;
      source.rank_count = in.i32();
      source.local_rank = in.i32();
      if (source.rank_count != source_ranks || source.local_rank != rank)
        throw std::invalid_argument("AMR history flux archive has a foreign source owner");
      const auto owners = in.size(sizeof(std::uint64_t));
      if (owners != source.patches.size())
        throw std::invalid_argument("AMR history flux archive owner map omits patches");
      source.owners.resize(owners);
      for (int& owner : source.owners)
        owner = in.i32();
      const auto local = in.size(sizeof(std::uint64_t));
      if (local > owners)
        throw std::invalid_argument("AMR history flux archive contains too many owned patches");
      for (std::size_t entry = 0; entry < local; ++entry) {
        OwnedPatch<Dim> patch;
        patch.global_patch = static_cast<std::size_t>(in.u64());
        if (patch.global_patch >= owners)
          throw std::invalid_argument("AMR history flux archive has a foreign patch ordinal");
        for (int axis = 0; axis < Dim; ++axis) {
          const auto values = in.size(sizeof(double));
          if (values !=
              detail::samples(source.patches[patch.global_patch], axis, source.components))
            throw std::invalid_argument("AMR history flux archive has a partial face payload");
          auto& density = patch.density[axis];
          density.resize(values);
          for (Real& value : density) {
            const double encoded = in.real();
            if (!std::isfinite(encoded) ||
                encoded > static_cast<double>(std::numeric_limits<Real>::max()) ||
                encoded < -static_cast<double>(std::numeric_limits<Real>::max()))
              throw std::invalid_argument("AMR history flux payload exceeds native precision");
            value = static_cast<Real>(encoded);
            if (std::bit_cast<std::uint64_t>(static_cast<double>(value)) !=
                std::bit_cast<std::uint64_t>(encoded))
              throw std::invalid_argument("AMR history flux payload changes native precision");
          }
        }
        source.owned.push_back(std::move(patch));
      }
      certify_snapshot(source, max_levels);
      auto [found, inserted] = complete.try_emplace(token);
      if (inserted) {
        found->second.metadata = source;
        found->second.metadata.owned.clear();
      } else {
        checkpoint_detail::Writer left, right;
        write_physical_metadata(left, found->second.metadata);
        write_physical_metadata(right, source);
        if (std::move(left).take() != std::move(right).take() ||
            found->second.metadata.owners != source.owners)
          throw std::invalid_argument("AMR history flux source metadata differs across ranks");
      }
      for (auto& patch : source.owned)
        if (!found->second.patches.emplace(patch.global_patch, std::move(patch)).second)
          throw std::invalid_argument("AMR history flux archive duplicates an owned patch");
    }
    in.finish();
    if (rank == 0)
      expected_tokens = std::move(tokens);
    else if (tokens != expected_tokens)
      throw std::invalid_argument("AMR history flux archive source set differs across ranks");
  }
  SnapshotMap<Dim> result;
  for (auto& [token, entry] : complete) {
    auto& source = entry.metadata;
    if (entry.patches.size() != source.patches.size())
      throw std::invalid_argument("AMR history flux archive omits physical source support");
    std::vector<std::string> digests;
    for (std::size_t patch = 0; patch < source.patches.size(); ++patch)
      digests.push_back(patch_digest(entry.patches.at(patch)));
    if (content_identity(source, digests) != token)
      throw std::invalid_argument(
          "AMR history flux physical payload differs from its source digest");
    source.rank_count = target_ranks;
    source.local_rank = target_rank;
    for (std::size_t patch = 0; patch < source.patches.size(); ++patch) {
      if (target_ranks != source_ranks)
        source.owners[patch] = static_cast<int>(patch % static_cast<std::size_t>(target_ranks));
      if (source.owners[patch] == target_rank)
        source.owned.push_back(std::move(entry.patches.at(patch)));
    }
    certify_snapshot(source, max_levels);
    result.emplace(token, std::make_shared<const Snapshot<Dim>>(std::move(source)));
  }
  return result;
}

}  // namespace pops::runtime::program::history_flux
