/// @file
/// @brief Prepared ND clustering provider contract.

#pragma once

#include <pops/amr/tagging/tag_mask.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pops::amr::tagging {

struct ClusterWorkBudget {
  std::size_t shards = 0;
  std::size_t recursion_nodes = 0;
  std::size_t cell_visits = 0;
  std::size_t output_boxes = 0;
  std::size_t identity_bytes = 0;

  bool operator==(const ClusterWorkBudget&) const = default;
};

template <int Dim>
struct ClusterOptions {
  double min_efficiency = 0.0;
  std::array<int, Dim> min_box_size{};
  std::array<int, Dim> max_box_size{};
  ClusterWorkBudget budget{};

  bool operator==(const ClusterOptions&) const = default;
};

template <int Dim>
struct ClusterResultIdentity {
  std::string provider{};
  hierarchy::LevelLayoutIdentity<Dim> source_level{};
  ClusterOptions<Dim> options{};
  std::vector<TagShardIdentity<Dim>> canonical_shards{};
  std::vector<Box<Dim>> boxes{};

  bool operator==(const ClusterResultIdentity&) const = default;
};

template <int Dim>
struct ClusterResult {
  mesh::BoxArray<Dim> boxes{};
  ClusterResultIdentity<Dim> identity{};
};

template <int Dim>
class ClusterProvider {
 public:
  virtual ~ClusterProvider() = default;
  virtual std::string_view provider_identity() const noexcept = 0;
  virtual ClusterResult<Dim> cluster(std::span<const TagMask<Dim>> shards,
                                     const ClusterOptions<Dim>& options) const = 0;
};

}  // namespace pops::amr::tagging
