#pragma once

#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/amr_history_flux_snapshot_codec.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program::history_flux::execution {

namespace execution_detail {

template <class Prepare>
void prepare_collectively(const ExecutionLane& lane, std::string_view operation,
                          Prepare&& prepare) {
  std::exception_ptr error;
  try {
    std::forward<Prepare>(prepare)();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, operation);
}

template <int Dim>
double face_measure(const Geometry<Dim>& geometry, int normal_axis) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    if (axis != normal_axis)
      result *= static_cast<double>(geometry.spacing(axis));
  return result;
}

template <int Dim>
POPS_HD Real cell_centered_face_value(const FieldView<const Real, Dim>& flux,
                                      const Index<Dim>& face, int axis, int component) {
  Index<Dim> lower = face;
  --lower[axis];
  return Real(0.5) * (flux(lower, component) + flux(face, component));
}

}  // namespace execution_detail

// Pack already evaluated samples into owned immutable storage. The caller supplies the exact
// physical source identity and selects integrated-face versus density values; named cell fields
// retain their existing arithmetic face average and are always densities.
template <int Dim, class Field, class Evaluation>
std::shared_ptr<const Snapshot<Dim>> capture_owned_snapshot(
    const std::string& source_identity, const Field& rhs, const Evaluation* evaluation,
    const std::array<Field, Dim>* exact, const std::array<Field*, Dim>* named,
    const Geometry<Dim>& geometry, const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
    const ExecutionLane& lane, std::size_t max_levels, bool integrated_face_values) {
  using MemorySpace = typename Field::memory_space;
  std::shared_ptr<Snapshot<Dim>> snapshot;
  std::vector<char> digests;
  std::string contract;
  execution_detail::prepare_collectively(lane, "AMR history face capture preparation", [&] {
    if (named == nullptr && exact == nullptr && evaluation == nullptr)
      throw std::invalid_argument("AMR history face capture lacks its evaluated source");
    snapshot = std::make_shared<Snapshot<Dim>>();
    snapshot->identity = "pending-physical-source-digest";
    snapshot->source_identity = source_identity;
    snapshot->level = layout.level();
    snapshot->components = rhs.ncomp();
    snapshot->domain = geometry.domain();
    snapshot->rank_count = lane.size();
    snapshot->local_rank = lane.rank();
    for (int axis = 0; axis < Dim; ++axis)
      snapshot->cell_size[axis] = static_cast<double>(geometry.spacing(axis));
    for (std::size_t global = 0; global < rhs.layout().size(); ++global) {
      snapshot->patches.push_back(rhs.layout()[global]);
      snapshot->owners.push_back(
          rhs.distribution().replicated()
              ? 0
              : static_cast<int>(rhs.rank_space().linear_rank(rhs.distribution().owner(global))));
      if (snapshot->owners.back() != lane.rank())
        continue;
      const auto local = rhs.local_index_of(global);
      if (local == Field::not_local)
        throw std::logic_error("AMR history flux source lost its owned patch");
      OwnedPatch<Dim> patch;
      patch.global_patch = global;
      for (int axis = 0; axis < Dim; ++axis) {
        const auto face_box = nd::face_box(rhs.layout()[global], axis);
        const auto cells = detail::points(face_box);
        const auto count = detail::checked_mul(cells, static_cast<std::size_t>(rhs.ncomp()));
        FieldView<const Real, Dim> values;
        if (named != nullptr) {
          if ((*named)[axis] == nullptr)
            throw std::invalid_argument("AMR named history source has an absent axis field");
          const auto& field = *(*named)[axis];
          const auto source_local = field.local_index_of(global);
          if (source_local == Field::not_local)
            throw std::logic_error("AMR named history source lacks its owned patch");
          values = field.fab(source_local).view();
        } else if (exact != nullptr) {
          const auto& field = (*exact)[axis];
          const auto source_local = field.local_index_of(global);
          if (source_local == Field::not_local)
            throw std::logic_error("AMR exact history source lacks its owned patch");
          values = field.fab(source_local).view();
        } else {
          values = evaluation->integrated_face_fluxes.at(local).view().axes[axis];
        }
        Kokkos::View<Real*, MemorySpace> packed("pops_owned_history_face_snapshot", count);
        const bool cell_provider = named != nullptr;
        const Real measure = static_cast<Real>(execution_detail::face_measure(geometry, axis));
        const bool integrated = named == nullptr && integrated_face_values;
        Kokkos::parallel_for(
            "pops_pack_owned_history_faces",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, count),
            [=] POPS_HD(std::size_t ordinal) {
              std::size_t remainder = ordinal % cells;
              Index<Dim> face = face_box.lo;
              for (int direction = 0; direction < Dim; ++direction) {
                const auto width = static_cast<std::size_t>(face_box.length(direction));
                face[direction] += static_cast<int>(remainder % width);
                remainder /= width;
              }
              const int component = static_cast<int>(ordinal / cells);
              const Real value = cell_provider ? execution_detail::cell_centered_face_value(
                                                     values, face, axis, component)
                                               : values(face, component);
              packed(ordinal) = integrated ? value / measure : value;
            });
        const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed);
        patch.density[axis].assign(host.data(), host.data() + count);
      }
      snapshot->owned.push_back(std::move(patch));
    }
    certify_prepared_layout(*snapshot, layout);
    validate_snapshot(*snapshot, max_levels);
    digests.assign(detail::checked_mul(snapshot->patches.size(), std::size_t{64}), char{0});
    for (const auto& patch : snapshot->owned) {
      const auto digest = patch_digest(patch);
      std::copy(digest.begin(), digest.end(), digests.begin() + patch.global_patch * 64);
    }
    checkpoint_detail::Writer metadata;
    write_physical_metadata(metadata, *snapshot);
    metadata.i32(snapshot->rank_count);
    for (int owner : snapshot->owners)
      metadata.i32(owner);
    const auto bytes = std::move(metadata).take();
    contract.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"history-face-source", contract}}, lane))
    throw std::invalid_argument("AMR history face source layout differs across execution ranks");
  // Only fixed-size per-patch digests communicate here. Physical face samples remain owned.
  all_reduce_or_inplace(digests.data(), digests.size(), lane);
  execution_detail::prepare_collectively(lane, "AMR history face source digest preparation", [&] {
    std::vector<std::string> patch_digests;
    for (std::size_t patch = 0; patch < snapshot->patches.size(); ++patch)
      patch_digests.emplace_back(digests.data() + patch * 64, 64);
    snapshot->identity = content_identity(*snapshot, patch_digests);
  });
  return snapshot;
}

// Face supplies axis, face, and flux_density members. Its other geometric/role fields are copied
// unchanged. Communication contains only requested raw source faces, in exact canonical order.
template <int Dim, class Face>
std::vector<Face> query_faces(const std::shared_ptr<const Snapshot<Dim>>& snapshot,
                              const std::vector<Face>& current_faces, const ExecutionLane& lane,
                              std::size_t max_levels) {
  using Key = std::pair<std::string, FaceQuery<Dim>>;
  std::map<Key, std::shared_ptr<const Snapshot<Dim>>> queries;
  std::vector<std::vector<SourceTerm<Dim>>> terms;
  std::vector<Real> samples;
  std::vector<Real> counts;
  std::string contract;
  execution_detail::prepare_collectively(lane, "AMR sparse history face query preparation", [&] {
    if (!snapshot)
      throw std::invalid_argument("AMR moved history interface lacks physical source samples");
    for (const auto& face : current_faces) {
      terms.push_back(source_queries(snapshot, {face.axis, face.face}, max_levels));
      for (const auto& term : terms.back())
        queries.emplace(Key{term.source->identity, term.query}, term.source);
    }
    samples.assign(
        detail::checked_mul(queries.size(), static_cast<std::size_t>(snapshot->components)),
        Real(0));
    counts.assign(queries.size(), Real(0));
    checkpoint_detail::Writer exact;
    exact.string(snapshot->identity);
    exact.size(queries.size());
    std::size_t ordinal = 0;
    for (const auto& [key, source] : queries) {
      exact.string(key.first);
      exact.i32(key.second.axis);
      for (int axis = 0; axis < Dim; ++axis)
        exact.i32(key.second.face[axis]);
      if (const auto payload = local_payload(*source, key.second)) {
        counts[ordinal] = Real(1);
        std::copy(payload->begin(), payload->end(),
                  samples.begin() + ordinal * static_cast<std::size_t>(snapshot->components));
      }
      ++ordinal;
    }
    const auto bytes = std::move(exact).take();
    contract.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"history-face-queries", contract}}, lane))
    throw std::invalid_argument("AMR history sparse query plan differs across ranks");
  all_reduce_sum_inplace(counts.data(), counts.size(), lane);
  all_reduce_sum_inplace(samples.data(), samples.size(), lane);
  std::vector<Face> result;
  execution_detail::prepare_collectively(lane, "AMR sparse history face payload preparation", [&] {
    if (!std::all_of(counts.begin(), counts.end(), [](Real count) { return count == Real(1); }))
      throw std::invalid_argument("AMR historical face query has missing or duplicate ownership");
    std::map<Key, std::size_t> ordinals;
    for (const auto& [key, source] : queries) {
      (void)source;
      ordinals.emplace(key, ordinals.size());
    }
    result = current_faces;
    for (std::size_t face = 0; face < result.size(); ++face) {
      result[face].flux_density.assign(static_cast<std::size_t>(snapshot->components), Real(0));
      for (const auto& term : terms[face]) {
        const auto ordinal = ordinals.at(Key{term.source->identity, term.query});
        for (int component = 0; component < snapshot->components; ++component)
          result[face].flux_density[static_cast<std::size_t>(component)] +=
              static_cast<Real>(term.weight.value()) *
              samples[ordinal * static_cast<std::size_t>(snapshot->components) + component];
      }
    }
  });
  return result;
}

// Enumerate supplied interfaces only; selection of coarse/fine topology belongs to the caller.
// Face follows the geometric record {role, axis, face, coarse_face, face_measure, flux_density}.
template <int Dim, class Face, class Interface>
std::vector<Face> interface_faces(const Geometry<Dim>& geometry,
                                  const ::pops::amr::hierarchy::LevelLayout<Dim>& current,
                                  const ::pops::amr::hierarchy::LevelLayout<Dim>* parent,
                                  const std::vector<Interface>& outgoing,
                                  const std::vector<Interface>& incoming) {
  using Role = ::pops::amr::reflux::FaceLedgerRole;
  std::vector<Face> faces;
  for (const auto& interface : outgoing)
    faces.push_back({Role::Coarse,
                     interface.axis,
                     interface.coarse_face,
                     interface.coarse_face,
                     execution_detail::face_measure(geometry, interface.axis),
                     {}});
  if (parent == nullptr) {
    if (!incoming.empty())
      throw std::invalid_argument("AMR incoming history interfaces require their parent layout");
    return faces;
  }
  const auto ratio = current.ratio_from_parent();
  for (const auto& interface : incoming) {
    std::size_t count = 1;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != interface.axis)
        count = detail::checked_mul(count, static_cast<std::size_t>(ratio[axis]));
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      auto remainder = ordinal;
      Index<Dim> face;
      for (int axis = 0; axis < Dim; ++axis) {
        // Independently shifted domains may have a wide offset even when the final index fits.
        const std::int64_t relative =
            static_cast<std::int64_t>(interface.coarse_face[axis]) - parent->domain().lo[axis];
        std::int64_t coordinate =
            static_cast<std::int64_t>(current.domain().lo[axis]) + relative * ratio[axis];
        if (axis != interface.axis) {
          coordinate +=
              static_cast<std::int64_t>(remainder % static_cast<std::size_t>(ratio[axis]));
          remainder /= static_cast<std::size_t>(ratio[axis]);
        }
        face[axis] = pops::detail::checked_box_index(
            coordinate, "AMR history interface face exceeds integer coordinates");
      }
      faces.push_back({Role::Fine,
                       interface.axis,
                       face,
                       interface.coarse_face,
                       execution_detail::face_measure(geometry, interface.axis),
                       {}});
    }
  }
  return faces;
}

// Construct only the lazy mathematical projection; no physics evaluation or state publication.
template <int Dim>
std::shared_ptr<const Snapshot<Dim>> project_snapshot(
    const std::shared_ptr<const Snapshot<Dim>>& parent,
    const ::pops::amr::hierarchy::LevelLayout<Dim>& layout, const Geometry<Dim>& geometry,
    std::size_t max_levels) {
  if (!parent)
    throw std::invalid_argument(
        "AMR parent history transfer lacks its authenticated complete face source");
  auto projected = std::make_shared<Snapshot<Dim>>();
  projected->identity = "pending-conservative-projection-digest";
  projected->source_identity = parent->source_identity;
  projected->parent = parent;
  projected->level = layout.level();
  projected->components = parent->components;
  projected->rank_count = parent->rank_count;
  projected->local_rank = parent->local_rank;
  projected->domain = layout.domain();
  projected->patches = layout.patches().boxes();
  projected->owners.assign(projected->patches.size(), 0);
  for (int axis = 0; axis < Dim; ++axis) {
    projected->ratio[axis] = layout.ratio_from_parent()[axis];
    projected->cell_size[axis] = static_cast<double>(geometry.spacing(axis));
  }
  certify_prepared_layout(*projected, layout);
  validate_snapshot(*projected, max_levels);
  projected->identity = projection_identity(*projected);
  return projected;
}

}  // namespace pops::runtime::program::history_flux::execution
