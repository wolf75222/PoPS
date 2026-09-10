#pragma once

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/amr_history_flux_snapshot.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

namespace pops::runtime::multiblock {
struct InterfaceFluxSample;
}

namespace pops::runtime::program::history_flux {

template <int Dim>
struct BasisFace {
  ::pops::amr::reflux::FaceLedgerRole role = ::pops::amr::reflux::FaceLedgerRole::Coarse;
  int axis = 0;
  Index<Dim> face{};
  Index<Dim> coarse_face{};
  double face_measure = 0.0;
  std::vector<Real> flux_density;
};

enum class BasisProvider : std::uint8_t {
  PreparedResidual = 0,
  PreparedDefaultFlux = 1,
  ExactFace = 2,
  NamedCell = 3,
  DiffusiveFace = 4,
};

template <int Dim>
struct Basis {
  std::uint64_t identity = 0;
  std::size_t runtime_block = 0;
  int level = 0;
  runtime::multiblock::BoundaryEvaluationPoint point{};
  int rhs_identity = -1;
  BasisProvider provider = BasisProvider::PreparedResidual;
  std::string temporal_family;
  ::pops::amr::ClockWindow window{};
  std::vector<BasisFace<Dim>> faces;
  std::vector<std::shared_ptr<const runtime::multiblock::InterfaceFluxSample>> shared_samples;
  // Complete raw source samples remain distributed. Projected nodes are internal conservative
  // history transport, never evaluated physical fluxes or public whole-face exchange evidence.
  std::shared_ptr<const Snapshot<Dim>> history_snapshot;
};

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

// These read-only graph operations neither own a history registry nor publish accepted state.
template <class Expression, class Bounds>
void require_expression_budget(const Expression& expression, const Bounds& rhs_bounds,
                               const Bounds& coefficient_bounds) {
  std::map<std::size_t, std::size_t> bases_by_block;
  for (const auto& [identity, term] : expression) {
    (void)identity;
    if (!term.basis || term.basis->runtime_block >= rhs_bounds.size() ||
        term.basis->runtime_block >= coefficient_bounds.size())
      throw std::logic_error("AMR Program flux expression has a foreign basis identity");
    const std::size_t block = term.basis->runtime_block;
    if (++bases_by_block[block] > rhs_bounds[block] ||
        term.coefficient.size() > coefficient_bounds[block])
      throw std::length_error(
          "AMR Program flux expression exceeds its authenticated artifact budget");
  }
}

template <int Dim, class Registry>
SnapshotMap<Dim> reachable_sources(const Registry& expressions, int max_levels) {
  SnapshotMap<Dim> raw;
  for (const auto& [key, slots] : expressions) {
    (void)key;
    for (const auto& expression : slots)
      for (const auto& [identity, term] : expression) {
        (void)identity;
        auto snapshot = term.basis ? term.basis->history_snapshot : nullptr;
        int remaining = max_levels;
        while (snapshot && snapshot->parent) {
          if (--remaining <= 0)
            throw std::logic_error("AMR history snapshot lineage exceeds authored levels");
          snapshot = snapshot->parent;
        }
        if (snapshot) {
          const auto source_identity = snapshot->identity;
          raw.emplace(source_identity, std::move(snapshot));
        }
      }
  }
  return raw;
}

template <class Face>
std::vector<Face> certified_legacy_faces(const std::vector<Face>& retained,
                                         const std::vector<Face>& geometry) {
  std::vector<Face> result;
  for (const auto& face : geometry) {
    const auto old = std::find_if(retained.begin(), retained.end(), [&](const auto& previous) {
      return face.role == previous.role && face.axis == previous.axis &&
             face.face == previous.face && face.coarse_face == previous.coarse_face &&
             face.face_measure == previous.face_measure;
    });
    if (old == retained.end())
      throw std::invalid_argument(
          "AMR moved legacy history interface lacks authenticated source samples");
    result.push_back(*old);
  }
  return result;
}

// Bind raw payload identity to the immutable evaluation point supplied by its producer.
template <class Point, class Provider>
std::string source_point_identity(std::size_t block, std::uint64_t basis_identity,
                                  const Point& point, int rhs_identity, Provider provider,
                                  std::uint64_t topology_epoch, std::uint64_t generation) {
  checkpoint_detail::Writer source;
  source.string("pops.amr.history-face-source-point.v1");
  source.u64(block);
  source.u64(basis_identity);
  source.i32(rhs_identity);
  source.u64(static_cast<std::uint64_t>(provider));
  source.u64(topology_epoch);
  source.u64(generation);
  source.string(point.clock);
  source.i64(point.tick);
  source.i32(point.level);
  source.i32(point.substep);
  source.i32(point.stage);
  source.i64(point.stage_fraction.numerator);
  source.i64(point.stage_fraction.denominator);
  source.real(point.dt);
  source.real(point.physical_time);
  source.string(point.graph_identity);
  source.string(point.rate_identity);
  source.string(point.application_identity);
  return "pops.amr.history-face-source-point.v1:sha256:" +
         identity::sha256_hex(std::move(source).take());
}

// Encode immutable history expressions. The caller supplies only its shared-source wire writer;
// this routine has no registry ownership, active clock, or publication side effects.
template <int Dim, class Registry, class SharedWriter>
std::vector<std::uint8_t> serialize_expressions(const Registry& histories, int max_levels,
                                                SharedWriter&& write_shared) {
  // A retained state sample can have no evaluated flux contribution. Preserve
  // its explicit ring/slot registry: empty bytes mean absent provenance to the
  // restore validator, not an authenticated collection of zero-term samples.
  // Keep the no-history wire representation compatible with existing readers.
  if (histories.empty())
    return {};
  const bool snapshots = std::any_of(histories.begin(), histories.end(), [](const auto& ring) {
    return std::any_of(ring.second.begin(), ring.second.end(), [](const auto& expression) {
      return std::any_of(expression.begin(), expression.end(), [](const auto& term) {
        return term.second.basis && term.second.basis->history_snapshot;
      });
    });
  });
  checkpoint_detail::Writer out;
  out.u64(snapshots ? UINT64_C(0x504f5053464c5834) : UINT64_C(0x504f5053464c5833));
  out.size(histories.size());
  for (const auto& [key, slots] : histories) {
    out.string(key);
    out.size(slots.size());
    for (const auto& expression : slots) {
      out.size(expression.size());
      for (const auto& [identity, term] : expression) {
        if (!term.basis || term.basis->identity != identity)
          throw std::logic_error("AMR Program history flux payload has an unauthenticated basis");
        out.u64(identity);
        out.size(term.coefficient.size());
        for (const auto& [power, coefficient] : term.coefficient) {
          out.i32(power);
          out.i64(coefficient.numerator);
          out.i64(coefficient.denominator);
        }
        const auto& basis = *term.basis;
        out.u64(basis.runtime_block);
        out.i32(basis.level);
        out.i32(basis.rhs_identity);
        out.u64(static_cast<std::uint64_t>(basis.provider));
        out.string(basis.temporal_family);
        const auto write_point = [&](const auto& point) {
          out.string(point.clock);
          out.i64(point.tick);
          out.i32(point.level);
          out.i32(point.substep);
          out.i32(point.stage);
          out.i64(point.stage_fraction.numerator);
          out.i64(point.stage_fraction.denominator);
          out.real(point.dt);
          out.real(point.physical_time);
          out.string(point.graph_identity);
          out.string(point.rate_identity);
          out.string(point.application_identity);
        };
        write_point(basis.point);
        checkpoint_detail::write_clock(out, basis.window.begin);
        checkpoint_detail::write_clock(out, basis.window.end);
        out.size(basis.faces.size());
        for (const auto& face : basis.faces) {
          out.u64(static_cast<std::uint64_t>(face.role));
          out.i32(face.axis);
          for (int axis = 0; axis < Dim; ++axis) {
            out.i32(face.face[axis]);
            out.i32(face.coarse_face[axis]);
          }
          out.real(face.face_measure);
          out.size(face.flux_density.size());
          for (const Real value : face.flux_density)
            out.real(static_cast<double>(value));
        }
        out.size(basis.shared_samples.size());
        for (const auto& sample : basis.shared_samples) {
          if (!sample)
            throw std::logic_error("history shared flux sample is absent");
          write_shared(out, *sample);
        }
        if (snapshots)
          history_flux::write_descriptor(out, basis.history_snapshot, max_levels);
      }
    }
  }
  return std::move(out).take();
}

}  // namespace pops::runtime::program::history_flux
