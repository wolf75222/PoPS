#pragma once

#include <cstddef>
#include <string>

namespace pops::runtime::field {

struct FieldTopologyReportRow {
  std::string patch_identity;
  std::string topology_digest;
  std::string provenance;
  std::size_t material_points = 0;
  std::size_t connected_components = 0;
  std::string source_layout_identity;
  std::string materialized_layout_identity;
};

}  // namespace pops::runtime::field
