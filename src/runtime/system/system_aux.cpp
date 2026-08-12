/// @file
/// @brief Exact owner-qualified auxiliary-provider registry and carrier publication for System.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace {

using runtime::system::AuxiliaryComponentKey;
using runtime::system::AuxiliaryEvaluationPoint;
using runtime::system::AuxiliaryProviderKind;
using runtime::system::AuxiliaryStorageShape;

template <int Dim>
std::size_t domain_cells(const Box<Dim>& domain) {
  const std::int64_t count = domain.numPts();
  if (count < 0 || static_cast<std::uint64_t>(count) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("System auxiliary carrier domain exceeds size_t");
  return static_cast<std::size_t>(count);
}

template <int Dim>
Extent<Dim> maximum_auxiliary_halo(const runtime::system::ExactAuxiliaryRegistry<Dim>& registry) {
  Extent<Dim> result{};
  for (std::size_t provider = 0; provider < registry.provider_count(); ++provider)
    for (const auto& output : registry.provider(provider).outputs())
      for (int axis = 0; axis < Dim; ++axis)
        result[axis] = std::max(result[axis], output.shape.halo[axis]);
  return result;
}

template <int Dim>
const typename runtime::system::ExactAuxiliaryRegistry<Dim>::provider_type* find_auxiliary_output(
    const runtime::system::ExactAuxiliaryRegistry<Dim>& registry,
    const AuxiliaryComponentKey& key) {
  const std::string exact_key = key.exact_key();
  const auto* result =
      static_cast<const typename runtime::system::ExactAuxiliaryRegistry<Dim>::provider_type*>(
          nullptr);
  for (std::size_t provider = 0; provider < registry.provider_count(); ++provider) {
    const auto& candidate = registry.provider(provider);
    for (const auto& output : candidate.outputs()) {
      if (output.key.exact_key() != exact_key)
        continue;
      if (result != nullptr)
        throw std::logic_error("sealed auxiliary registry has duplicate output ownership");
      result = &candidate;
    }
  }
  return result;
}

template <int Dim>
void write_auxiliary_component(MultiFab<Dim>& carrier, const Box<Dim>& domain,
                               std::size_t component, const std::vector<double>& values) {
  using namespace runtime::system::marshaling;
  if (component >= static_cast<std::size_t>(carrier.ncomp()) ||
      values.size() != domain_cells(domain))
    throw std::invalid_argument("System auxiliary input does not match its compact carrier slot");
  require_exact_domain_decomposition(carrier, domain);
  for (std::size_t local = 0; local < carrier.local_size(); ++local) {
    Fab<Dim>& fab = carrier.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
      host(storage_ordinal(fab, index, static_cast<int>(component))) =
          static_cast<Real>(values[domain_ordinal(domain, index)]);
    });
    fab.copy_from_host(host);
  }
}

template <int Dim>
std::vector<double> read_auxiliary_component(const MultiFab<Dim>& carrier, const Box<Dim>& domain,
                                             std::size_t component) {
  using namespace runtime::system::marshaling;
  if (component >= static_cast<std::size_t>(carrier.ncomp()))
    throw std::out_of_range("System auxiliary component is outside the compact carrier");
  require_exact_domain_decomposition(carrier, domain);
  std::vector<double> result(domain_cells(domain), 0.0);
  long local_failure = 0;
  try {
    for (std::size_t local = 0; local < carrier.local_size(); ++local) {
      const Fab<Dim>& fab = carrier.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
        result[domain_ordinal(domain, index)] =
            static_cast<double>(host(storage_ordinal(fab, index, static_cast<int>(component))));
      });
    }
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure) != 0)
    throw std::runtime_error("System auxiliary component gather failed collectively");
  all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

template <int Dim>
void require_collective_auxiliary_point(const AuxiliaryEvaluationPoint& point) {
  ExactContractBuilder exact;
  point.serialize_exact(exact);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-auxiliary-evaluation-point", std::move(exact).release()}}))
    throw std::invalid_argument("System auxiliary evaluation point differs across MPI ranks");
}

}  // namespace

template <int Dim>
void System<Dim>::install_prepared_auxiliary_provider(
    runtime::system::PreparedAuxiliaryProvider<Dim> provider) {
  require_assembling(p_->lifecycle_, "install_prepared_auxiliary_provider");
  for (const auto& output : provider.outputs())
    if (output.shape.value_components != 1)
      throw std::invalid_argument(
          "System compact provider carrier requires one ComponentKey per scalar value");
  p_->auxiliary_registry_.add(std::move(provider));
  p_->auxiliary_registry_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::install_auxiliary_consumer_plan(
    runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) {
  require_assembling(p_->lifecycle_, "install_auxiliary_consumer_plan");
  // A Program or bind-time analytic consumer can arrive after providers sealed. It contributes
  // only a resolved local gather plan, never a new global output or carrier component. Prepare the
  // complete value image on every rank and turn rank-local validation failures into one collective
  // rejection before any rank enters the exact-byte witness.
  auto candidate = p_->auxiliary_registry_;
  std::exception_ptr local_error;
  try {
    candidate.add_consumer_plan(std::move(plan));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System auxiliary consumer plan preparation failed collectively");
  }
  if (candidate.sealed()) {
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-auxiliary-consumer-plan", candidate.collective_contract()}}))
      throw std::runtime_error("System auxiliary consumer plan differs across MPI ranks");
    p_->auxiliary_registry_ = std::move(candidate);
    p_->auxiliary_registry_consensus_verified_ = true;
    return;
  }
  p_->auxiliary_registry_ = std::move(candidate);
  p_->auxiliary_registry_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::seal_auxiliary_providers() {
  if (p_->auxiliary_registry_.sealed()) {
    if (!p_->auxiliary_registry_consensus_verified_ &&
        !all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-auxiliary-registry", p_->auxiliary_registry_.collective_contract()}}))
      throw std::runtime_error("System auxiliary registry differs across MPI ranks");
    p_->auxiliary_registry_consensus_verified_ = true;
    return;
  }

  using registry_type = runtime::system::ExactAuxiliaryRegistry<Dim>;
  using carrier_type = runtime::system::AuxiliaryStorageGroups<Dim>;
  std::optional<registry_type> candidate_registry;
  std::optional<carrier_type> candidate_carrier;
  std::exception_ptr local_error;
  try {
    candidate_registry.emplace(p_->auxiliary_registry_);
    candidate_registry->seal();
    if (candidate_registry->slot_count() != 0) {
      candidate_carrier.emplace();
      for (const auto& group : candidate_registry->storage_groups()) {
        if (group.component_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::overflow_error("System auxiliary storage-group width exceeds int");
        Extent<Dim> ghosts{};
        for (int axis = 0; axis < Dim; ++axis)
          ghosts[axis] = group.shape.halo[axis];
        candidate_carrier->groups.emplace(
            group.identity, MultiFab<Dim>(p_->ba, p_->dm, p_->local_rank,
                                          static_cast<int>(group.component_count), ghosts));
      }
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System auxiliary registry preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-auxiliary-registry", candidate_registry->collective_contract()}}))
    throw std::runtime_error("System auxiliary registry differs across MPI ranks");

  p_->auxiliary_registry_ = std::move(*candidate_registry);
  p_->provider_carrier_ = std::move(candidate_carrier);
  p_->auxiliary_ghost_transport_.reset();
  p_->auxiliary_registry_consensus_verified_ = true;
}

template <int Dim>
void System<Dim>::stage_auxiliary_input(const AuxiliaryComponentKey& key,
                                        const std::vector<double>& values) {
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error("System auxiliary inputs require a sealed provider registry");
  const auto* provider = find_auxiliary_output<Dim>(p_->auxiliary_registry_, key);
  if (provider == nullptr)
    throw std::out_of_range("System auxiliary input key is not produced by this registry");
  if (provider->kind() != AuxiliaryProviderKind::input)
    throw std::invalid_argument(
        "System auxiliary input key belongs to a derived or field-output provider");
  if (values.size() != domain_cells(p_->dom))
    throw std::invalid_argument("System auxiliary input has the wrong exact-ranked global shape");
  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::length_error("System auxiliary input byte count exceeds size_t");
  const std::string identity = key.exact_key();
  const std::string_view payload(reinterpret_cast<const char*>(values.data()),
                                 values.size() * sizeof(double));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-auxiliary-input-key", identity}, {"system-auxiliary-input-values", payload}}))
    throw std::invalid_argument("System auxiliary input differs across MPI ranks");
  p_->staged_auxiliary_inputs_[identity] = values;
  if (std::find(p_->dirty_auxiliary_providers_.begin(), p_->dirty_auxiliary_providers_.end(),
                provider->identity()) == p_->dirty_auxiliary_providers_.end())
    p_->dirty_auxiliary_providers_.push_back(provider->identity());
}

template <int Dim>
void System<Dim>::refresh_auxiliary(const AuxiliaryEvaluationPoint& point) {
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error("System auxiliary refresh requires a sealed provider registry");
  require_collective_auxiliary_point<Dim>(point);
  if (p_->auxiliary_registry_.slot_count() == 0) {
    auto transaction =
        p_->auxiliary_registry_.begin_publication(point, p_->dirty_auxiliary_providers_);
    transaction.accept();
    p_->dirty_auxiliary_providers_.clear();
    return;
  }
  runtime::system::AuxiliaryStorageGroups<Dim> candidate;
  using transaction_type =
      typename runtime::system::ExactAuxiliaryRegistry<Dim>::PublicationTransaction;
  std::optional<transaction_type> prepared_transaction;
  std::exception_ptr candidate_error;
  try {
    if (!p_->provider_carrier_)
      throw std::logic_error("System sealed auxiliary registry has no compact carrier allocation");
    candidate = *p_->provider_carrier_;
    prepared_transaction.emplace(
        p_->auxiliary_registry_.begin_publication(point, p_->dirty_auxiliary_providers_));
  } catch (...) {
    candidate_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      candidate_error, nullptr,
      "System auxiliary candidate preparation failed collectively before lane duplication");
  auto& transaction = *prepared_transaction;
  try {
    if (!p_->auxiliary_ghost_lane_)
      p_->auxiliary_ghost_lane_.emplace(
          ExecutionLane::duplicate_world_collectively("pops.system.auxiliary-ghosts"));
    std::exception_ptr staging_error;
    try {
      for (const std::size_t index : p_->auxiliary_registry_.topological_order()) {
        const auto& provider = p_->auxiliary_registry_.provider(index);
        if (provider.kind() != AuxiliaryProviderKind::input ||
            !transaction.requires_staging(provider.identity()))
          continue;
        for (const auto& output : provider.outputs()) {
          const auto staged = p_->staged_auxiliary_inputs_.find(output.key.exact_key());
          if (staged == p_->staged_auxiliary_inputs_.end())
            throw std::runtime_error(
                "System auxiliary input provider is due but one output was never staged");
          const auto address = p_->auxiliary_registry_.address_of(output.key);
          auto* group = candidate.find(address.group);
          if (group == nullptr)
            throw std::logic_error("System auxiliary candidate lacks the resolved storage group");
          write_auxiliary_component(*group, p_->dom, address.component, staged->second);
        }
        transaction.stage_external(provider.identity());
      }
    } catch (...) {
      staging_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        staging_error, &*p_->auxiliary_ghost_lane_,
        "System auxiliary input staging failed collectively before ghost preparation");
    if (!p_->auxiliary_ghost_transport_)
      p_->auxiliary_ghost_transport_.emplace(runtime::system::prepare_auxiliary_ghost_transport(
          *p_->provider_carrier_, p_->auxiliary_registry_, p_->dom, p_->geom,
          BoundaryTopology<Dim>::axis_periodic(p_->periodicity), &*p_->auxiliary_ghost_lane_));
    const auto refresh_candidate_ghosts = [&] {
      p_->auxiliary_ghost_transport_->execute(candidate);
    };
    // Inputs become dependency-visible only after their exact same-level/remote and physical
    // ghosts exist.  Repeat after each derived launch so a stencil-valued derived provider never
    // observes a predecessor's valid-only candidate image.
    refresh_candidate_ghosts();
    transaction.launch_ready_native(
        {&*p_->provider_carrier_, &candidate}, [&](const auto&, std::exception_ptr local_error) {
          runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
              local_error, &*p_->auxiliary_ghost_lane_,
              "System auxiliary native provider launch failed collectively before ghost fill");
          refresh_candidate_ghosts();
        });
    std::exception_ptr fence_error;
    try {
      Kokkos::fence();
    } catch (...) {
      fence_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        fence_error, &*p_->auxiliary_ghost_lane_,
        "System auxiliary device fence failed collectively");
    runtime::system::require_finite_auxiliary_groups(candidate, &*p_->auxiliary_ghost_lane_,
                                                     "System auxiliary publication");
    transaction.accept();
    p_->provider_carrier_ = std::move(candidate);
    p_->dirty_auxiliary_providers_.clear();
  } catch (...) {
    transaction.reject();
    throw;
  }
}

template <int Dim>
runtime::system::AuxiliaryStorageAddress<Dim> System<Dim>::auxiliary_address(
    const AuxiliaryComponentKey& key) const {
  return p_->auxiliary_registry_.address_of(key);
}

template <int Dim>
std::vector<double> System<Dim>::auxiliary_component(const AuxiliaryComponentKey& key) const {
  const auto address = auxiliary_address(key);
  if (!p_->provider_carrier_)
    throw std::out_of_range("System auxiliary component belongs to an empty carrier");
  const auto* group = p_->provider_carrier_->find(address.group);
  if (group == nullptr)
    throw std::logic_error("System auxiliary carrier lacks the resolved storage group");
  return read_auxiliary_component(*group, p_->dom, address.component);
}

template <int Dim>
std::string System<Dim>::auxiliary_registry_contract() const {
  return std::string(p_->auxiliary_registry_.collective_contract());
}

template <int Dim>
const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
System<Dim>::prepared_auxiliary_consumer_plan(const std::string& consumer_qid) const {
  return p_->auxiliary_registry_.consumer_plan(consumer_qid);
}

template <int Dim>
runtime::system::AuxiliaryCheckpointAcceptedState<Dim>
System<Dim>::capture_auxiliary_checkpoint_accepted_state() const {
  runtime::system::AuxiliaryCheckpointAcceptedState<Dim> result;
  long local_failure = 0;
  std::string collective_contract;
  try {
    if (!p_->dirty_auxiliary_providers_.empty())
      throw std::logic_error(
          "System auxiliary checkpoint refuses dirty provider state before accepted publication");
    result = runtime::system::capture_auxiliary_checkpoint_state(p_->auxiliary_registry_);
    if (result.groups.empty()) {
      if (p_->provider_carrier_)
        throw std::logic_error("System auxiliary checkpoint has storage without registry groups");
    } else {
      if (!p_->provider_carrier_)
        throw std::logic_error("System auxiliary checkpoint has no storage groups");
      runtime::system::require_auxiliary_checkpoint_storage(result, *p_->provider_carrier_);
      const std::size_t cells = domain_cells(p_->dom);
      for (auto& descriptor : result.groups) {
        const MultiFab<Dim>* const group = p_->provider_carrier_->find(descriptor.identity);
        if (group == nullptr)
          throw std::logic_error("System auxiliary checkpoint lost a sealed storage group");
        if (descriptor.component_count > std::numeric_limits<std::size_t>::max() / cells)
          throw std::length_error("System auxiliary checkpoint payload exceeds size_t");
        descriptor.payload.reserve(descriptor.component_count * cells);
        for (std::size_t component = 0; component < descriptor.component_count; ++component) {
          auto values = read_auxiliary_component(*group, p_->dom, component);
          descriptor.payload.insert(descriptor.payload.end(), values.begin(), values.end());
        }
      }
    }
    const auto bytes = runtime::system::serialize_auxiliary_checkpoint_state(result);
    ExactContractBuilder exact;
    exact.text("pops.uniform-exact-auxiliary-checkpoint")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    collective_contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure) != 0)
    throw std::runtime_error("System auxiliary checkpoint capture failed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.uniform-auxiliary-checkpoint"), collective_contract}}))
    throw std::runtime_error("System auxiliary checkpoint differs between communicator ranks");
  return result;
}

template <int Dim>
void System<Dim>::restore_auxiliary_checkpoint_accepted_state(
    const runtime::system::AuxiliaryCheckpointAcceptedState<Dim>& state) {
  long local_failure = 0;
  std::string collective_contract;
  try {
    if (!p_->dirty_auxiliary_providers_.empty())
      throw std::logic_error("System auxiliary checkpoint restore refuses dirty live providers");
    if (state.groups.empty()) {
      if (p_->provider_carrier_)
        throw std::invalid_argument(
            "System auxiliary checkpoint storage groups differ from runtime");
    } else {
      if (!p_->provider_carrier_)
        throw std::invalid_argument("System auxiliary checkpoint requires live storage groups");
      runtime::system::require_auxiliary_checkpoint_storage(state, *p_->provider_carrier_);
      const std::size_t cells = domain_cells(p_->dom);
      for (const auto& descriptor : state.groups)
        if (descriptor.component_count > std::numeric_limits<std::size_t>::max() / cells ||
            descriptor.payload.size() != descriptor.component_count * cells)
          throw std::invalid_argument(
              "System auxiliary checkpoint payload differs from its exact domain shape");
    }
    const auto bytes = runtime::system::serialize_auxiliary_checkpoint_state(state);
    ExactContractBuilder exact;
    exact.text("pops.uniform-exact-auxiliary-checkpoint")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    collective_contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure) != 0)
    throw std::invalid_argument(
        "System auxiliary checkpoint preflight failed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.uniform-auxiliary-checkpoint"), collective_contract}}))
    throw std::runtime_error("System auxiliary checkpoint differs between communicator ranks");

  using registry_type = runtime::system::ExactAuxiliaryRegistry<Dim>;
  using carrier_type = runtime::system::AuxiliaryStorageGroups<Dim>;
  std::optional<registry_type> registry_snapshot;
  std::optional<carrier_type> storage_snapshot;
  decltype(p_->staged_auxiliary_inputs_) staged_snapshot;
  decltype(p_->dirty_auxiliary_providers_) dirty_snapshot;
  const bool had_storage = p_->provider_carrier_.has_value();
  std::exception_ptr snapshot_error;
  try {
    registry_snapshot.emplace(p_->auxiliary_registry_);
    if (had_storage)
      storage_snapshot.emplace(*p_->provider_carrier_);
    staged_snapshot = p_->staged_auxiliary_inputs_;
    dirty_snapshot = p_->dirty_auxiliary_providers_;
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      snapshot_error, nullptr,
      "System auxiliary checkpoint snapshot failed collectively before registry mutation");

  std::exception_ptr restore_error;
  try {
    registry_type candidate_registry = *registry_snapshot;
    std::optional<carrier_type> candidate_carrier;
    if (had_storage) {
      candidate_carrier.emplace(*storage_snapshot);
      const std::size_t cells = domain_cells(p_->dom);
      for (const auto& descriptor : state.groups) {
        MultiFab<Dim>* const group = candidate_carrier->find(descriptor.identity);
        if (group == nullptr)
          throw std::logic_error("System auxiliary checkpoint candidate lost a storage group");
        for (std::size_t component = 0; component < descriptor.component_count; ++component) {
          const auto begin =
              descriptor.payload.begin() + static_cast<std::ptrdiff_t>(component * cells);
          write_auxiliary_component(
              *group, p_->dom, component,
              std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(cells)));
        }
      }
      runtime::system::require_finite_auxiliary_groups(*candidate_carrier, nullptr,
                                                       "System auxiliary checkpoint candidate");
    }
    runtime::system::restore_auxiliary_checkpoint_state(state, candidate_registry);
    // Publish values before accepted generation/provenance so no observer can pair restored
    // metadata with the previous numerical image.
    p_->provider_carrier_ = std::move(candidate_carrier);
    p_->auxiliary_registry_ = std::move(candidate_registry);
    p_->dirty_auxiliary_providers_.clear();
  } catch (...) {
    restore_error = std::current_exception();
  }
  const long restore_failed = all_reduce_max(restore_error ? 1L : 0L);
  if (restore_failed != 0) {
    std::exception_ptr rollback_error;
    p_->auxiliary_ghost_transport_.reset();
    try {
      p_->auxiliary_registry_ = std::move(*registry_snapshot);
    } catch (...) {
      rollback_error = std::current_exception();
    }
    try {
      if (had_storage)
        p_->provider_carrier_ = std::move(*storage_snapshot);
      else
        p_->provider_carrier_.reset();
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    try {
      p_->staged_auxiliary_inputs_.swap(staged_snapshot);
      p_->dirty_auxiliary_providers_.swap(dirty_snapshot);
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        rollback_error, nullptr, "System auxiliary checkpoint rollback failed collectively");
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        restore_error, nullptr, "System auxiliary checkpoint restore failed collectively");
  }
}

template <int Dim>
const MultiFab<Dim>* System<Dim>::prepared_block_auxiliary_storage() const {
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error("System auxiliary storage is unavailable before registry seal");
  if (!p_->provider_carrier_)
    return nullptr;
  if (p_->provider_carrier_->groups.size() != 1)
    throw std::logic_error("mixed provider storage groups require plan-qualified access");
  return &p_->provider_carrier_->groups.begin()->second;
}

template <int Dim>
const runtime::system::AuxiliaryStorageGroups<Dim>*
System<Dim>::prepared_block_provider_storage_groups() const {
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error("System provider storage groups are unavailable before registry seal");
  return p_->provider_carrier_ ? &*p_->provider_carrier_ : nullptr;
}

template <int Dim>
runtime::system::AuxiliaryStorageGroups<Dim>* System<Dim>::prepared_amr_provider_storage_groups() {
  require_assembling(p_->lifecycle_, "prepared_amr_provider_storage_groups");
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error("System AMR provider storage groups require a sealed registry");
  return p_->provider_carrier_ ? &*p_->provider_carrier_ : nullptr;
}

template void System<kNativeDimension>::install_prepared_auxiliary_provider(
    runtime::system::PreparedAuxiliaryProvider<kNativeDimension>);
template void System<kNativeDimension>::install_auxiliary_consumer_plan(
    runtime::system::AuxiliaryConsumerProviderPlan<kNativeDimension>);
template void System<kNativeDimension>::seal_auxiliary_providers();
template void System<kNativeDimension>::stage_auxiliary_input(const AuxiliaryComponentKey&,
                                                              const std::vector<double>&);
template void System<kNativeDimension>::refresh_auxiliary(const AuxiliaryEvaluationPoint&);
template runtime::system::AuxiliaryStorageAddress<kNativeDimension>
System<kNativeDimension>::auxiliary_address(const AuxiliaryComponentKey&) const;
template std::vector<double> System<kNativeDimension>::auxiliary_component(
    const AuxiliaryComponentKey&) const;
template std::string System<kNativeDimension>::auxiliary_registry_contract() const;
template const runtime::system::ResolvedAuxiliaryConsumerPlan<kNativeDimension>&
System<kNativeDimension>::prepared_auxiliary_consumer_plan(const std::string&) const;
template runtime::system::AuxiliaryCheckpointAcceptedState<kNativeDimension>
System<kNativeDimension>::capture_auxiliary_checkpoint_accepted_state() const;
template void System<kNativeDimension>::restore_auxiliary_checkpoint_accepted_state(
    const runtime::system::AuxiliaryCheckpointAcceptedState<kNativeDimension>&);
template const MultiFab<kNativeDimension>*
System<kNativeDimension>::prepared_block_auxiliary_storage() const;
template const runtime::system::AuxiliaryStorageGroups<kNativeDimension>*
System<kNativeDimension>::prepared_block_provider_storage_groups() const;
template runtime::system::AuxiliaryStorageGroups<kNativeDimension>*
System<kNativeDimension>::prepared_amr_provider_storage_groups();

}  // namespace pops
