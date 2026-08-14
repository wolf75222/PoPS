/// @file
/// @brief Public prepared native Tagger facade.

#pragma once

#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

/// Immutable RuntimeInstance authority for one native Tagger session.
struct PreparedTaggerComponentSpec {
  std::string component_id{};
  std::string manifest_identity{};
  std::string provider_identity{};
  std::string tagging_graph_identity{};
  std::string layout_identity{};
  std::string clock_identity{};
  std::uint32_t interface_version = 0;
  PopsTaggerExecutionModeV2 execution_mode = POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution{};
  std::string parameters_json{};
  std::string target_json{};
};

}  // namespace pops::runtime::amr

#include <pops/runtime/amr/detail/native_tagger_session.hpp>

#include <array>
#include <cstddef>

namespace pops::runtime::amr {

/// Prepared native Tagger adapter. It only produces owner-local candidate masks; the canonical
/// AMR engine retains equality policy, persistent hysteresis, clustering and regrid publication.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedTaggerComponent {
 public:
  using Program = PreparedTaggingProgram<Dim>;
  using Field = PreparedTaggingField<Dim, MemorySpace>;
  using Candidates = PreparedTaggerCandidates<Dim>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;

  PreparedTaggerComponent() = default;
  PreparedTaggerComponent(const PreparedTaggerComponent&) = delete;
  PreparedTaggerComponent& operator=(const PreparedTaggerComponent&) = delete;
  PreparedTaggerComponent(PreparedTaggerComponent&&) noexcept = default;
  PreparedTaggerComponent& operator=(PreparedTaggerComponent&&) noexcept = default;

  static PreparedTaggerComponent prepare(
      std::shared_ptr<component::LoadedComponent> component, PreparedTaggerComponentSpec spec,
      const Program& program, const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      std::uint32_t periodic_axes, const ExecutionLane& lane) {
    return PreparedTaggerComponent(Session::prepare(std::move(component), std::move(spec), program,
                                                    fields_by_level, layouts, budgets,
                                                    topology_generation, periodic_axes, lane));
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return session_ != nullptr && static_cast<bool>(*session_);
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return session_ == nullptr ? std::string_view{} : session_->collective_contract();
  }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return session_ == nullptr ? 0 : session_->topology_generation();
  }

  const Candidates& execute(std::size_t level_index,
                            const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
                            const std::array<Real, Dim>& spacing, std::uint64_t topology_generation,
                            std::int64_t tick, double physical_time) {
    return session_->execute(level_index, layout, spacing, topology_generation, tick,
                             physical_time);
  }

 private:
  using Session = detail::NativeTaggerSession<Dim, MemorySpace, PreparedTaggerComponentSpec>;

  explicit PreparedTaggerComponent(std::unique_ptr<Session> session) noexcept
      : session_(std::move(session)) {}

  std::unique_ptr<Session> session_{};
};

}  // namespace pops::runtime::amr
