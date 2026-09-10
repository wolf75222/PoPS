#pragma once

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::runtime::program::history_flux {

template <int Dim>
struct FaceQuery {
  int axis = 0;
  Index<Dim> face{};
  friend bool operator==(const FaceQuery&, const FaceQuery&) = default;
  friend bool operator<(const FaceQuery& a, const FaceQuery& b) {
    if (a.axis != b.axis)
      return a.axis < b.axis;
    for (int d = Dim - 1; d >= 0; --d)
      if (a.face[d] != b.face[d])
        return a.face[d] < b.face[d];
    return false;
  }
};

template <int Dim>
struct OwnedPatch {
  std::size_t global_patch = 0;
  // Component-slowest, x-fastest samples on each axis's complete face box.
  std::array<std::vector<Real>, Dim> density;
};

template <int Dim>
struct Snapshot {
  static_assert(Dim >= 1 && Dim <= 3);
  // A globally canonical physical source/lineage identity, independent of current storage owners.
  std::string identity;
  // Raw producer point/operator/clock/dt/topology token. The archive identity additionally binds
  // physical metadata and full sample content; projections retain that lineage through parent.
  std::string source_identity;
  int level = 0;
  int components = 0;
  Box<Dim> domain;
  std::array<double, Dim> cell_size{};
  // The supplied global patch order is authoritative, including for shared-face ownership.
  std::vector<Box<Dim>> patches;
  std::vector<int> owners;
  int rank_count = 1;
  int local_rank = 0;
  std::vector<OwnedPatch<Dim>> owned;
  std::shared_ptr<const Snapshot<Dim>> parent;
  std::array<int, Dim> ratio{};

 private:
  struct GeometryCertificate {
    Box<Dim> domain;
    std::vector<Box<Dim>> patches;
  };
  // This witness is never serialized. Only the checked factories below can construct it.
  // Copying a snapshot may share the witness, but changing either public geometry field makes
  // the exact comparison fail and restores exhaustive validation. Const validation never writes.
  std::shared_ptr<const GeometryCertificate> geometry_certificate_;

  bool certified_geometry_matches_() const {
    return geometry_certificate_ && geometry_certificate_->domain == domain &&
           geometry_certificate_->patches == patches;
  }

  template <int D>
  friend void validate_snapshot(const Snapshot<D>&, std::size_t);
  template <int D>
  friend void certify_snapshot(Snapshot<D>&, std::size_t);
  template <int D>
  friend void certify_prepared_layout(Snapshot<D>&, const ::pops::amr::hierarchy::LevelLayout<D>&);
  template <int D>
  friend std::size_t resident_bytes(const Snapshot<D>&, std::size_t);
};

template <int Dim>
struct SourceTerm {
  std::shared_ptr<const Snapshot<Dim>> source;
  FaceQuery<Dim> query;
  ::pops::amr::Rational weight;
};

namespace detail {

inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("AMR history flux allocation sum overflow");
  return a + b;
}

inline std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("AMR history flux allocation product overflow");
  return a * b;
}

template <int Dim>
std::size_t points(const Box<Dim>& box) {
  if (box.empty())
    throw std::invalid_argument("AMR history flux requires nonempty boxes");
  std::size_t result = 1;
  for (int d = 0; d < Dim; ++d) {
    const auto extent = static_cast<std::uint64_t>(box.length(d));
    if (extent > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("AMR history flux extent exceeds size_t");
    result = checked_mul(result, static_cast<std::size_t>(extent));
  }
  return result;
}

template <int Dim>
std::size_t samples(const Box<Dim>& box, int axis, int components) {
  if (components <= 0)
    throw std::invalid_argument("AMR history flux requires positive components");
  const auto count =
      checked_mul(points(nd::face_box(box, axis)), static_cast<std::size_t>(components));
  (void)checked_mul(count, sizeof(Real));
  return count;
}

template <int Dim>
void metadata(const Snapshot<Dim>& s) {
  if (s.identity.empty() || s.level < 0 || s.components <= 0 || s.domain.empty() ||
      s.rank_count <= 0 || s.local_rank < 0 || s.local_rank >= s.rank_count || s.patches.empty() ||
      s.owners.size() != s.patches.size())
    throw std::invalid_argument("AMR history flux snapshot metadata is invalid");
  for (int d = 0; d < Dim; ++d) {
    if (!std::isfinite(s.cell_size[d]) || s.cell_size[d] <= 0)
      throw std::invalid_argument("AMR history flux cell metric is invalid");
    (void)nd::face_box(s.domain, d);
  }
  if (!s.parent) {
    if (s.source_identity.empty())
      throw std::invalid_argument("AMR history flux raw source identity is missing");
    return;
  }
  const auto& p = *s.parent;
  if (s.level <= 0 || p.level != s.level - 1 || s.components != p.components || !s.owned.empty() ||
      s.rank_count != p.rank_count || s.local_rank != p.local_rank)
    throw std::invalid_argument("AMR history flux projection lineage is invalid");
  bool refined = false;
  for (int d = 0; d < Dim; ++d) {
    if (s.ratio[d] < 1)
      throw std::invalid_argument("AMR history flux refinement ratio is invalid");
    refined = refined || s.ratio[d] > 1;
    if (p.domain.empty() || s.domain.length(d) != p.domain.length(d) * s.ratio[d])
      throw std::invalid_argument("AMR history flux projection domain extent mismatch");
    const double expected = p.cell_size[d] / s.ratio[d];
    if (!std::isfinite(expected) || expected <= 0 ||
        std::abs(s.cell_size[d] - expected) >
            32 * std::numeric_limits<double>::epsilon() * expected)
      throw std::invalid_argument("AMR history flux projection metric mismatch");
  }
  if (!refined)
    throw std::invalid_argument("AMR history flux projection must refine at least one axis");
}

template <int Dim>
std::size_t supporting_patch(const Snapshot<Dim>& s, const FaceQuery<Dim>& query) {
  if (query.axis < 0 || query.axis >= Dim)
    throw std::invalid_argument("AMR history flux query axis is invalid");
  for (std::size_t patch = 0; patch < s.patches.size(); ++patch)
    if (nd::face_box(s.patches[patch], query.axis).contains(query.face))
      return patch;
  throw std::out_of_range("AMR history flux query has no archived face support");
}

}  // namespace detail

// Validate once at publication/restore, before treating a shared_ptr<const Snapshot> as immutable.
// Projection values are conservative synthetic history transfers, not newly evaluated fluxes.
template <int Dim>
void validate_snapshot(const Snapshot<Dim>& snapshot, std::size_t max_levels) {
  std::set<const Snapshot<Dim>*> seen;
  std::set<std::string> identities;
  std::size_t bytes = 0;
  for (const auto* s = &snapshot; s; s = s->parent.get()) {
    if (seen.size() >= max_levels || !seen.insert(s).second ||
        !identities.insert(s->identity).second)
      throw std::invalid_argument("AMR history flux lineage exceeds its bound or repeats a source");
    detail::metadata(*s);
    bytes = detail::checked_add(bytes, detail::checked_mul(s->patches.size(), sizeof(Box<Dim>)));
    bytes = detail::checked_add(bytes, detail::checked_mul(s->owners.size(), sizeof(int)));
    const bool geometry_certified = s->certified_geometry_matches_();
    std::size_t local_count = 0;
    for (std::size_t i = 0; i < s->patches.size(); ++i) {
      const auto& box = s->patches[i];
      if (!s->domain.contains(box) || s->owners[i] < 0 || s->owners[i] >= s->rank_count)
        throw std::invalid_argument("AMR history flux patch or storage owner is invalid");
      if (!geometry_certified)
        for (std::size_t j = 0; j < i; ++j)
          if (!box.intersect(s->patches[j]).empty())
            throw std::invalid_argument("AMR history flux cell patches overlap or repeat");
      for (int axis = 0; axis < Dim; ++axis)
        (void)detail::samples(box, axis, s->components);
      local_count += s->owners[i] == s->local_rank;
    }
    if (s->parent)
      continue;
    if (s->owned.size() != local_count)
      throw std::invalid_argument("AMR history flux local patch coverage is incomplete");
    for (std::size_t i = 0; i < s->owned.size(); ++i) {
      const auto& payload = s->owned[i];
      if (payload.global_patch >= s->patches.size() ||
          s->owners[payload.global_patch] != s->local_rank ||
          (i && s->owned[i - 1].global_patch >= payload.global_patch))
        throw std::invalid_argument("AMR history flux payload ownership or ordering is invalid");
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& values = payload.density[axis];
        if (values.size() != detail::samples(s->patches[payload.global_patch], axis, s->components))
          throw std::invalid_argument("AMR history flux face payload is incomplete");
        bytes = detail::checked_add(bytes, detail::checked_mul(values.size(), sizeof(Real)));
        for (Real value : values)
          if (!std::isfinite(value))
            throw std::invalid_argument("AMR history flux face payload is nonfinite");
      }
    }
  }
}

// External data takes the exhaustive path unless it already carries an exactly matching private
// witness. Certification completes before immutable publication, never during a const getter.
template <int Dim>
void certify_snapshot(Snapshot<Dim>& snapshot, std::size_t max_levels) {
  validate_snapshot(snapshot, max_levels);
  if (!snapshot.certified_geometry_matches_())
    snapshot.geometry_certificate_ =
        std::make_shared<const typename Snapshot<Dim>::GeometryCertificate>(
            typename Snapshot<Dim>::GeometryCertificate{snapshot.domain, snapshot.patches});
}

// LevelLayout's constructor already proved domain containment and pairwise disjointness. Bind
// that proof to this exact ordered geometry in linear work. Payload and ownership validation is
// deliberately still required through validate_snapshot before publication.
template <int Dim>
void certify_prepared_layout(Snapshot<Dim>& snapshot,
                             const ::pops::amr::hierarchy::LevelLayout<Dim>& layout) {
  if (snapshot.domain != layout.domain() || snapshot.patches != layout.patches().boxes())
    throw std::invalid_argument("AMR history flux geometry differs from its prepared level layout");
  (void)detail::checked_mul(snapshot.patches.size(), sizeof(Box<Dim>));
  if (!snapshot.certified_geometry_matches_())
    snapshot.geometry_certificate_ =
        std::make_shared<const typename Snapshot<Dim>::GeometryCertificate>(
            typename Snapshot<Dim>::GeometryCertificate{snapshot.domain, snapshot.patches});
}

// Capacity-based accounting of this ancestry's snapshot containers and full sample payloads.
// String inline storage may be conservatively counted twice; allocator/control-block overhead
// is implementation-defined and excluded. A shared ancestor is counted once per call.
template <int Dim>
std::size_t resident_bytes(const Snapshot<Dim>& snapshot, std::size_t max_levels = 64) {
  std::set<const Snapshot<Dim>*> seen;
  std::set<const typename Snapshot<Dim>::GeometryCertificate*> certificates;
  std::size_t bytes = 0;
  for (const auto* s = &snapshot; s; s = s->parent.get()) {
    if (seen.size() >= max_levels || !seen.insert(s).second)
      throw std::invalid_argument("AMR history flux resident lineage exceeds its bound or cycles");
    bytes = detail::checked_add(bytes, sizeof(Snapshot<Dim>));
    bytes = detail::checked_add(bytes, detail::checked_add(s->identity.capacity(), 1));
    bytes = detail::checked_add(bytes, detail::checked_add(s->source_identity.capacity(), 1));
    bytes =
        detail::checked_add(bytes, detail::checked_mul(s->patches.capacity(), sizeof(Box<Dim>)));
    bytes = detail::checked_add(bytes, detail::checked_mul(s->owners.capacity(), sizeof(int)));
    bytes = detail::checked_add(bytes,
                                detail::checked_mul(s->owned.capacity(), sizeof(OwnedPatch<Dim>)));
    if (s->geometry_certificate_ && certificates.insert(s->geometry_certificate_.get()).second) {
      bytes = detail::checked_add(bytes, sizeof(typename Snapshot<Dim>::GeometryCertificate));
      bytes = detail::checked_add(
          bytes,
          detail::checked_mul(s->geometry_certificate_->patches.capacity(), sizeof(Box<Dim>)));
    }
    for (const auto& patch : s->owned)
      for (const auto& values : patch.density)
        bytes = detail::checked_add(bytes, detail::checked_mul(values.capacity(), sizeof(Real)));
  }
  return bytes;
}

// Resolve only requested faces. Normal linear interpolation and tangential constant injection
// commute with coarse-volume/interface flux integrals; they make no per-child divergence claim.
template <int Dim>
std::vector<SourceTerm<Dim>> source_queries(std::shared_ptr<const Snapshot<Dim>> snapshot,
                                            FaceQuery<Dim> query, std::size_t max_levels = 64) {
  if (!snapshot)
    throw std::invalid_argument("AMR history flux query requires a snapshot");
  std::map<FaceQuery<Dim>, ::pops::amr::Rational> queries{{query, {1, 1}}};
  std::size_t depth = 0;
  for (;;) {
    if (depth++ >= max_levels)
      throw std::invalid_argument("AMR history flux query lineage exceeds its bound");
    detail::metadata(*snapshot);
    for (const auto& [face, weight] : queries)
      (void)detail::supporting_patch(*snapshot, face);
    if (!snapshot->parent)
      break;
    std::map<FaceQuery<Dim>, ::pops::amr::Rational> next;
    for (const auto& [face, weight] : queries) {
      FaceQuery<Dim> lower{face.axis, {}};
      std::int64_t normal_remainder = 0;
      for (int d = 0; d < Dim; ++d) {
        const std::int64_t relative =
            static_cast<std::int64_t>(face.face[d]) - snapshot->domain.lo[d];
        const std::int64_t ratio = snapshot->ratio[d];
        const std::int64_t quotient = relative / ratio - (relative % ratio < 0 ? 1 : 0);
        const std::int64_t remainder = relative - quotient * ratio;
        lower.face[d] = pops::detail::checked_box_index(
            static_cast<std::int64_t>(snapshot->parent->domain.lo[d]) + quotient,
            "AMR history flux projected face exceeds integer coordinates");
        if (d == face.axis)
          normal_remainder = remainder;
      }
      const std::int64_t ratio = snapshot->ratio[face.axis];
      next[lower] = next[lower] + weight * ::pops::amr::Rational(ratio - normal_remainder, ratio);
      if (normal_remainder != 0) {
        auto upper = lower;
        upper.face[face.axis] = pops::detail::checked_box_index(
            static_cast<std::int64_t>(lower.face[face.axis]) + 1,
            "AMR history flux upper face exceeds integer coordinates");
        next[upper] = next[upper] + weight * ::pops::amr::Rational(normal_remainder, ratio);
      }
    }
    queries = std::move(next);
    snapshot = snapshot->parent;
  }
  std::vector<SourceTerm<Dim>> result;
  result.reserve(queries.size());
  for (const auto& [face, weight] : queries)
    result.push_back({snapshot, face, weight});
  return result;
}

template <int Dim>
std::optional<std::vector<Real>> local_payload(const Snapshot<Dim>& raw,
                                               const FaceQuery<Dim>& query) {
  detail::metadata(raw);
  if (raw.parent)
    throw std::invalid_argument("AMR history flux local payload requires a raw archive");
  const auto patch = detail::supporting_patch(raw, query);
  if (raw.owners[patch] < 0 || raw.owners[patch] >= raw.rank_count)
    throw std::invalid_argument("AMR history flux query owner is invalid");
  if (raw.owners[patch] != raw.local_rank)
    return std::nullopt;
  const OwnedPatch<Dim>* payload = nullptr;
  for (const auto& entry : raw.owned)
    if (entry.global_patch == patch) {
      if (payload)
        throw std::invalid_argument("AMR history flux duplicate local payload");
      payload = &entry;
    }
  if (!payload)
    throw std::invalid_argument("AMR history flux selected local patch is missing");
  const auto box = nd::face_box(raw.patches[patch], query.axis);
  const auto count = detail::points(box);
  const auto& values = payload->density[query.axis];
  if (values.size() != detail::samples(raw.patches[patch], query.axis, raw.components))
    throw std::invalid_argument("AMR history flux selected face payload is incomplete");
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int d = 0; d < Dim; ++d) {
    offset = detail::checked_add(
        offset, detail::checked_mul(
                    static_cast<std::size_t>(static_cast<std::int64_t>(query.face[d]) - box.lo[d]),
                    stride));
    stride = detail::checked_mul(stride, static_cast<std::size_t>(box.length(d)));
  }
  std::vector<Real> result(static_cast<std::size_t>(raw.components));
  for (int component = 0; component < raw.components; ++component) {
    const Real value = values[detail::checked_add(
        offset, detail::checked_mul(static_cast<std::size_t>(component), count))];
    if (!std::isfinite(value))
      throw std::invalid_argument("AMR history flux selected sample is nonfinite");
    result[static_cast<std::size_t>(component)] = value;
  }
  return result;
}

}  // namespace pops::runtime::program::history_flux
