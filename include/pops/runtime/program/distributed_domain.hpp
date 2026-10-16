#pragma once
// Distributed domain authority and collective phase boundary, independent of value codecs.
#include <pops/parallel/execution_lane.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program::distributed {
using Bytes = std::vector<std::uint8_t>;
enum class Code : std::int64_t {
  allocation = 1,
  capacity = 2,
  domain = 3,
  index = 4,
  csr = 5,
  arithmetic = 6,
  wire = 7,
  user = 8,
  internal = 9
};
struct Failure : std::exception {
  Code code;
  std::int64_t rank, gid;
  Failure(Code c, std::int64_t r = -1, std::int64_t g = -1) noexcept : code(c), rank(r), gid(g) {
    // Exception construction must remain allocation-free even for capacity failures.
    // Three full-width signed integers and all literal labels fit in this buffer.
    char* cursor = message_.data();
    const auto text = [&](const char* value) {
      while (*value)
        *cursor++ = *value++;
    };
    const auto number = [&](std::int64_t value) {
      cursor = std::to_chars(cursor, message_.data() + message_.size() - 1, value).ptr;
    };
    text("distributed finite phase rejected [code=");
    number(std::int64_t(code));
    text(" rank=");
    number(rank);
    text(" gid=");
    number(gid);
    text("]");
    *cursor = '\0';
  }
  const char* what() const noexcept override { return message_.data(); }

 private:
  std::array<char, 160> message_{};
};
inline void local_require(bool ok, Code code, std::int64_t gid = -1) {
  if (!ok)
    throw Failure(code, -1, gid);
}
// Pure placement arithmetic: the caller supplies the allocation budget and a
// real lane's rank count. No communicator, domain or version authority is minted.
inline std::vector<std::int64_t> partition_balanced(std::int64_t size, int ranks,
                                                    std::uint64_t byte_capacity) {
  local_require(size >= 0 && ranks > 0, Code::domain);
  const auto count = std::uint64_t(ranks) + 1;
  local_require(count <= byte_capacity / sizeof(std::int64_t) &&
                    count <= std::vector<std::int64_t>().max_size(),
                Code::capacity);
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(count));
  const auto quotient = size / ranks, remainder = size % ranks;
  for (std::int64_t rank = 0; rank <= ranks; ++rank)
    offsets[static_cast<std::size_t>(rank)] = rank * quotient + std::min(rank, remainder);
  return offsets;
}

inline void mpi_status(int status) {
#ifdef POPS_HAS_MPI
  if (status != MPI_SUCCESS)
    throw std::runtime_error("distributed MPI transport failure");
#else
  (void)status;
#endif
}
// Fixed-size wire words are explicitly little endian; no long/double conversion of Int64.
inline void write_word(Bytes& out, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(std::uint8_t(value >> (8 * i)));
}
inline std::uint64_t read_word(std::span<const std::uint8_t> in, std::size_t& at) {
  local_require(at <= in.size() && in.size() - at >= 8, Code::wire);
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= std::uint64_t(in[at++]) << (8 * i);
  return value;
}
inline std::int64_t common_failure_rank(bool failed, const ExecutionLane& lane) {
  const long rank = failed ? lane.rank() : lane.size();
  return all_reduce_min(rank, lane);
}
// Exactly one rendezvous follows this rank-local phase, including ranks with zero work.
// The lowest failing rank wins; local loop order defines its first failing global index.
template <class F>
void phase(const ExecutionLane& lane, F&& body) {
  std::optional<Failure> failure;
  try {
    body();
  } catch (const Failure& e) {
    failure = e;
  } catch (const std::bad_alloc&) {
    failure.emplace(Code::allocation);
  } catch (const std::length_error&) {
    failure.emplace(Code::capacity);
  } catch (...) {
    failure.emplace(Code::internal);
  }
  auto owner = common_failure_rank(bool(failure), lane);
  if (owner == lane.size())
    return;
  std::array<std::int64_t, 2> fields{};
  if (lane.rank() == owner)
    fields = {std::int64_t(failure->code), failure->gid};
#ifdef POPS_HAS_MPI
  mpi_status(MPI_Bcast(fields.data(), 2, MPI_INT64_T, int(owner), lane.native_handle()));
#endif
  throw Failure(Code(fields[0]), owner, fields[1]);
}
inline void agree_i64(std::int64_t value, const ExecutionLane& lane) {
  std::int64_t lower = value, upper = value;
#ifdef POPS_HAS_MPI
  mpi_status(MPI_Allreduce(&value, &lower, 1, MPI_INT64_T, MPI_MIN, lane.native_handle()));
  mpi_status(MPI_Allreduce(&value, &upper, 1, MPI_INT64_T, MPI_MAX, lane.native_handle()));
#endif
  if (lower != upper)
    throw Failure(Code::domain);
}
// A declared byte/version budget is independent of MPI's one-call Int32 count.
inline constexpr std::uint64_t maximum_budget =
    std::uint64_t(std::numeric_limits<std::int64_t>::max());
inline bool valid_budget(std::uint64_t value) noexcept {
  return value > 0 && value <= maximum_budget;
}
inline std::size_t byte_size(std::uint64_t value) {
  local_require(value <= maximum_budget && value <= std::numeric_limits<std::size_t>::max(),
                Code::capacity);
  return std::size_t(value);
}
// Arithmetic-only plan: no allocation proportional to total. Oversized round
// queries return an empty tail without multiplying beyond the final valid chunk.
struct ByteChunks {
  std::uint64_t total, limit;
  explicit ByteChunks(std::uint64_t bytes, std::uint64_t chunk = INT_MAX)
      : total(bytes), limit(chunk) {
    local_require(bytes <= maximum_budget && chunk > 0 && chunk <= INT_MAX, Code::capacity);
  }
  std::uint64_t rounds() const noexcept { return total == 0 ? 0 : 1 + (total - 1) / limit; }
  std::uint64_t offset(std::uint64_t round) const noexcept {
    return round >= rounds() ? total : round * limit;
  }
  int count(std::uint64_t round) const noexcept {
    return int(std::min(limit, total - offset(round)));
  }
};
inline void broadcast(std::span<std::uint8_t> bytes, int root, const ExecutionLane& lane,
                      std::uint64_t chunk_limit = INT_MAX) {
  std::optional<ByteChunks> plan;
  phase(lane, [&] {
    local_require(root >= 0 && root < lane.size(), Code::domain);
    plan.emplace(bytes.size(), chunk_limit);
  });
  agree_i64(root, lane);
  agree_i64(std::int64_t(bytes.size()), lane);
  agree_i64(std::int64_t(chunk_limit), lane);
#ifdef POPS_HAS_MPI
  for (std::uint64_t round = 0; round < plan->rounds(); ++round)
    mpi_status(MPI_Bcast(bytes.data() + std::size_t(plan->offset(round)), plan->count(round),
                         MPI_BYTE, root, lane.native_handle()));
#else
  (void)root;
#endif
}
inline std::int64_t integer(unsigned op, std::int64_t a, std::int64_t b) {
  std::int64_t out = 0;
  if (op == 0)
    local_require(!__builtin_add_overflow(a, b, &out), Code::arithmetic);
  else if (op == 1)
    local_require(!__builtin_sub_overflow(a, b, &out), Code::arithmetic);
  else if (op == 2)
    local_require(!__builtin_mul_overflow(a, b, &out), Code::arithmetic);
  else {
    local_require(b != 0 && !(a == std::numeric_limits<std::int64_t>::min() && b == -1),
                  Code::arithmetic);
    const auto rem = a % b;
    const bool adjust = rem != 0 && ((rem < 0) != (b < 0));
    out = op == 3 ? a / b - (adjust ? 1 : 0) : rem + (adjust ? b : 0);
  }
  return out;
}
struct Identity {
  std::array<std::uint64_t, 2> issuer;
  std::uint64_t ordinal;
};
class IssuerAuthority {
 public:
  explicit IssuerAuthority(std::array<std::uint64_t, 2> identity) : identity_(identity) {}
  bool visible() const noexcept { return live_.load(std::memory_order_acquire); }

 private:
  friend class Registry;
  void revoke() const noexcept { live_.store(false, std::memory_order_release); }
  std::array<std::uint64_t, 2> identity_;
  mutable std::atomic<bool> live_{true};
};
class Registry;
template <int Dim>
void collect_garbage(Registry&);
class Domain {
 public:
  bool visible() const noexcept { return authority_ && authority_->visible(); }
  void require_live() const { local_require(visible(), Code::domain); }
  std::int64_t size() const {
    require_live();
    return offsets_.back();
  }
  std::int64_t capacity() const {
    require_live();
    return capacity_;
  }
  std::int64_t family() const {
    require_live();
    return family_;
  }
  Identity identity() const {
    require_live();
    return identity_;
  }
  std::int64_t begin() const {
    require_live();
    return offsets_[rank_];
  }
  std::int64_t end() const {
    require_live();
    return offsets_[rank_ + 1];
  }
  std::int64_t local_size() const {
    require_live();
    return end() - begin();
  }
  bool owns(std::int64_t gid) const {
    require_live();
    return begin() <= gid && gid < end();
  }
  int owner(std::int64_t gid) const {
    local_require(0 <= gid && gid < size(), Code::index, gid);
    return int(std::upper_bound(offsets_.begin(), offsets_.end(), gid) - offsets_.begin()) - 1;
  }
  const ExecutionLane& lane() const noexcept { return *lane_; }
  const std::vector<std::int64_t>& partition_offsets() const {
    require_live();
    return offsets_;
  }

 private:
  friend class Registry;
  Domain(std::shared_ptr<const IssuerAuthority> authority, const ExecutionLane& lane, Identity id,
         std::int64_t family, std::int64_t capacity, std::vector<std::int64_t> offsets)
      : authority_(std::move(authority)),
        lane_(&lane),
        pin_(lane.borrow_immutably()),
        identity_(id),
        family_(family),
        capacity_(capacity),
        offsets_(std::move(offsets)),
        rank_(lane.rank()) {}
  std::shared_ptr<const IssuerAuthority> authority_;
  const ExecutionLane* lane_;
  ExecutionLane::ImmutableBorrow pin_;
  Identity identity_;
  std::int64_t family_, capacity_;
  std::vector<std::int64_t> offsets_;
  int rank_;
};
using DomainRef = std::shared_ptr<const Domain>;
template <int Dim>
std::string array_type_tag() {
  return "pops.distributed.Array@1/" + std::to_string(Dim);
}
// Embed in the persistent ProgramContext outside AcceptedSnapshot. Never copy or rollback this issuer.
// Restart creates a fresh issuer; transport resolves existing capabilities without minting authority.
class Registry {
 public:
  explicit Registry(const ExecutionLane& lane) : lane_(&lane), pin_(lane.borrow_immutably()) {
    phase(lane, [&] {
      if (lane.rank() == 0) {
        std::random_device r;
        for (auto& x : issuer_)
          x = (std::uint64_t(r()) << 32) ^ std::uint64_t(r());
      }
    });
#ifdef POPS_HAS_MPI
    mpi_status(MPI_Bcast(issuer_.data(), 2, MPI_UINT64_T, 0, lane.native_handle()));
#endif
    phase(lane, [&] { authority_ = std::make_shared<const IssuerAuthority>(issuer_); });
  }
  ~Registry() noexcept { revoke(); }
  void revoke() const noexcept {
    if (authority_)
      authority_->revoke();
  }
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
  DomainRef issue(std::int64_t family, std::int64_t capacity,
                  std::span<const std::int64_t> offsets) {
    const auto& lane = *lane_;
    configure(storage_bytes_, storage_versions_);
    const auto charge = domain_charge(offsets.size());
    phase(lane, [&] {
      local_require(domains_.size() < storage_versions_ && used_bytes_ <= storage_bytes_ &&
                        charge <= storage_bytes_ - used_bytes_,
                    Code::capacity);
    });
    phase(lane, [&] {
      local_require(authority_ && authority_->visible() && family >= 0 && capacity >= 0 &&
                        offsets.size() == std::size_t(lane.size() + 1) && offsets.front() == 0 &&
                        std::is_sorted(offsets.begin(), offsets.end()) &&
                        offsets.back() <= capacity,
                    Code::domain);
    });
    agree_i64(family, lane);
    agree_i64(capacity, lane);
    for (auto x : offsets)
      agree_i64(x, lane);  // common rank-count loop, never a local cardinality loop
    phase(lane, [&] {
      local_require(next_ < std::uint64_t(std::numeric_limits<std::int64_t>::max()),
                    Code::capacity);
    });
    agree_i64(std::int64_t(next_), lane);
    ++next_;  // burned even if allocation or the enclosing trial later fails
    DomainRef result;
    phase(lane, [&] {
      result = DomainRef(new Domain(authority_, lane, Identity{issuer_, next_}, family, capacity,
                                    std::vector<std::int64_t>(offsets.begin(), offsets.end())));
    });
    try {
      phase(lane, [&] { domains_.emplace(next_, result); });
    } catch (...) {
      domains_.erase(next_);
      throw;
    }
    used_bytes_ += charge;
    return result;
  }
  DomainRef issue_balanced(std::int64_t family, std::int64_t capacity, std::int64_t size) {
    const auto& lane = *lane_;
    configure(storage_bytes_, storage_versions_);
    const auto charge = domain_charge(std::size_t(lane.size()) + 1);
    phase(lane, [&] {
      local_require(size >= 0 && size <= capacity, Code::domain);
      local_require(domains_.size() < storage_versions_ && used_bytes_ <= storage_bytes_ &&
                        charge <= storage_bytes_ - used_bytes_,
                    Code::capacity);
    });
    agree_i64(size, lane);
    std::vector<std::int64_t> offsets;
    phase(lane,
          [&] { offsets = partition_balanced(size, lane.size(), storage_bytes_ - used_bytes_); });
    return issue(family, capacity, offsets);
  }
  // Transport may resolve only an already issued live instance. Numeric identity alone
  // never creates authority; descriptor equality prevents same-size/domain substitution.
  DomainRef resolve(Identity id, std::int64_t family, std::int64_t capacity,
                    std::span<const std::int64_t> offsets) const {
    local_require(authority_ && authority_->visible() && id.issuer == issuer_, Code::domain);
    auto found = domains_.find(id.ordinal);
    local_require(found != domains_.end(), Code::domain);
    auto d = found->second;
    authenticate(d);
    local_require(
        d->family_ == family && d->capacity_ == capacity &&
            std::equal(offsets.begin(), offsets.end(), d->offsets_.begin(), d->offsets_.end()),
        Code::domain);
    return d;
  }
  DomainRef resolve_existing(Identity id) const {
    local_require(authority_ && authority_->visible() && id.issuer == issuer_, Code::domain);
    auto found = domains_.find(id.ordinal);
    local_require(found != domains_.end(), Code::domain);
    auto result = found->second;
    authenticate(result);
    return result;
  }
  DomainRef resolve_domain(Identity id) const { return resolve_existing(id); }
  void authenticate(const DomainRef& d) const {
    local_require(authority_ && authority_->visible() && d && d->visible() &&
                      d->authority_ == authority_ && d->lane_ == lane_ &&
                      d->identity_.issuer == issuer_ && d->identity_.ordinal > 0 &&
                      d->identity_.ordinal <= next_,
                  Code::domain);
  }
  void common(const DomainRef& d) const {
    phase(*lane_, [&] { authenticate(d); });
    agree_i64(std::int64_t(d->identity_.ordinal), *lane_);
    agree_i64(d->family_, *lane_);
    agree_i64(d->capacity_, *lane_);
  }
  // Resource policy is mathematical-program configuration, never a model switch.
  // Issued versions are retained until a collective liveness frontier or restart.
  void configure(std::uint64_t bytes, std::uint64_t versions) {
    phase(*lane_, [&] {
      local_require(
          valid_budget(bytes) && valid_budget(versions) &&
              (!configured_ || (bytes == storage_bytes_ && versions == storage_versions_)),
          Code::capacity);
    });
    agree_i64(std::int64_t(bytes), *lane_);
    agree_i64(std::int64_t(versions), *lane_);
    storage_bytes_ = bytes;
    storage_versions_ = versions;
    configured_ = true;
  }
  Identity publish_value(std::shared_ptr<const void> value, std::string type,
                         std::uint64_t charge) {
    configure(storage_bytes_, storage_versions_);
    phase(*lane_, [&] {
      local_require(configured_ && value && !type.empty() && values_.size() < storage_versions_ &&
                        used_bytes_ <= storage_bytes_ && charge <= storage_bytes_ - used_bytes_ &&
                        next_value_ < std::uint64_t(std::numeric_limits<std::int64_t>::max()),
                    Code::capacity);
    });
    agree_i64(std::int64_t(next_value_), *lane_);
    ++next_value_;
    try {
      phase(*lane_, [&] {
        values_.emplace(next_value_, StoredValue{std::move(value), std::move(type), charge});
      });
    } catch (...) {
      values_.erase(next_value_);
      throw;
    }
    used_bytes_ += charge;
    return Identity{issuer_, next_value_};
  }
  // Prepare a final immutable object with its exact issued identity before
  // publishing any registry entry. The local factory must not enter collectives.
  template <class Factory>
  Identity publish_value_with_identity(Factory&& factory, std::string type, std::uint64_t charge) {
    configure(storage_bytes_, storage_versions_);
    phase(*lane_, [&] {
      local_require(configured_ && !type.empty() && values_.size() < storage_versions_ &&
                        used_bytes_ <= storage_bytes_ && charge <= storage_bytes_ - used_bytes_ &&
                        next_value_ < std::uint64_t(std::numeric_limits<std::int64_t>::max()),
                    Code::capacity);
    });
    agree_i64(std::int64_t(next_value_), *lane_);
    ++next_value_;
    const Identity identity{issuer_, next_value_};
    try {
      phase(*lane_, [&] {
        std::shared_ptr<const void> value = factory(identity);
        local_require(bool(value), Code::domain);
        values_.emplace(next_value_, StoredValue{std::move(value), std::move(type), charge});
      });
    } catch (...) {
      values_.erase(next_value_);
      throw;
    }
    used_bytes_ += charge;
    return identity;
  }
  std::shared_ptr<const void> resolve_value(Identity id, const std::string& type) const {
    local_require(authority_ && authority_->visible() && id.issuer == issuer_, Code::domain);
    auto found = values_.find(id.ordinal);
    local_require(found != values_.end() && found->second.type == type, Code::domain);
    return found->second.value;
  }
  std::uint64_t value_charge(Identity id, const std::string& type) const {
    (void)resolve_value(id, type);
    return values_.at(id.ordinal).charge;
  }
  static std::uint64_t domain_charge(std::size_t offsets) {
    local_require(offsets <= (std::numeric_limits<std::uint64_t>::max() - 128) / 8, Code::capacity);
    return 128 + 8 * std::uint64_t(offsets);
  }
  std::uint64_t retained_domains() const noexcept { return domains_.size(); }
  std::uint64_t retained_values() const noexcept { return values_.size(); }
  std::uint64_t retained_bytes() const noexcept { return used_bytes_; }
  std::uint64_t storage_bytes() const noexcept { return storage_bytes_; }
  std::uint64_t storage_versions() const noexcept { return storage_versions_; }
  std::array<std::uint64_t, 2> issuer() const noexcept { return issuer_; }
  // Restart prepares an independent authority and tables without communication.
  // Only the surrounding authenticated checkpoint transaction may adopt it.
  static std::shared_ptr<Registry> restore_local(const ExecutionLane& lane,
                                                 std::array<std::uint64_t, 2> issuer,
                                                 std::uint64_t bytes, std::uint64_t versions) {
    local_require(valid_budget(bytes) && valid_budget(versions), Code::capacity);
    return std::shared_ptr<Registry>(new Registry(lane, issuer, bytes, versions));
  }
  DomainRef restore_domain_local(Identity id, std::int64_t family, std::int64_t capacity,
                                 std::span<const std::int64_t> offsets) {
    local_require(authority_ && authority_->visible() && id.issuer == issuer_ && id.ordinal > 0 &&
                      id.ordinal < std::uint64_t(std::numeric_limits<std::int64_t>::max()) &&
                      family >= 0 && capacity >= 0 &&
                      offsets.size() == std::size_t(lane_->size() + 1) && offsets.front() == 0 &&
                      std::is_sorted(offsets.begin(), offsets.end()) &&
                      offsets.back() <= capacity && !domains_.contains(id.ordinal),
                  Code::domain);
    const auto charge = domain_charge(offsets.size());
    local_require(domains_.size() < storage_versions_ && used_bytes_ <= storage_bytes_ &&
                      charge <= storage_bytes_ - used_bytes_,
                  Code::capacity);
    auto result = DomainRef(new Domain(authority_, *lane_, id, family, capacity,
                                       std::vector<std::int64_t>(offsets.begin(), offsets.end())));
    domains_.emplace(id.ordinal, result);
    used_bytes_ += charge;
    next_ = std::max(next_, id.ordinal);
    return result;
  }
  void restore_value_local(Identity id, std::shared_ptr<const void> value, std::string type,
                           std::uint64_t charge) {
    local_require(authority_ && authority_->visible() && id.issuer == issuer_ && id.ordinal > 0 &&
                      id.ordinal < std::uint64_t(std::numeric_limits<std::int64_t>::max()) &&
                      !values_.contains(id.ordinal) && value && !type.empty() &&
                      values_.size() < storage_versions_ && used_bytes_ <= storage_bytes_ &&
                      charge <= storage_bytes_ - used_bytes_,
                  Code::capacity);
    values_.emplace(id.ordinal, StoredValue{std::move(value), std::move(type), charge});
    used_bytes_ += charge;
    next_value_ = std::max(next_value_, id.ordinal);
  }

 private:
  template <int Dim>
  friend void collect_garbage(Registry&);
  // Only the collective collector may prepare/apply a sweep. Iteration borrows
  // handles: taking shared_ptr copies here would fabricate external roots.
  template <class F>
  void inspect_domains(F&& f) const {
    for (const auto& [ordinal, domain] : domains_)
      f(ordinal, domain);
  }
  template <class F>
  void inspect_values(F&& f) const {
    for (const auto& [ordinal, value] : values_)
      f(ordinal, value.value, value.type, value.charge);
  }
  struct SweepPlan {
    std::vector<std::uint64_t> values, domains;
    std::uint64_t reclaimed_bytes = 0;
  };
  template <class Values, class Domains>
  SweepPlan prepare_sweep(const Values& live_values, const Domains& live_domains) const {
    local_require(authority_ && authority_->visible(), Code::domain);
    for (auto id : live_values)
      local_require(values_.contains(id), Code::domain);
    for (auto id : live_domains)
      local_require(domains_.contains(id), Code::domain);
    SweepPlan plan;
    auto charge = [&](std::uint64_t amount) {
      local_require(
          plan.reclaimed_bytes <= used_bytes_ && amount <= used_bytes_ - plan.reclaimed_bytes,
          Code::internal);
      plan.reclaimed_bytes += amount;
    };
    for (const auto& [id, value] : values_)
      if (!live_values.contains(id)) {
        plan.values.push_back(id);
        charge(value.charge);
      }
    for (const auto& [id, domain] : domains_)
      if (!live_domains.contains(id)) {
        plan.domains.push_back(id);
        charge(domain_charge(domain->partition_offsets().size()));
      }
    return plan;
  }
  void apply_sweep(SweepPlan&& plan) noexcept {
    for (auto id : plan.values)
      values_.erase(id);
    for (auto id : plan.domains)
      domains_.erase(id);
    used_bytes_ -= plan.reclaimed_bytes;
  }

 public:
  const ExecutionLane& lane() const noexcept { return *lane_; }
  std::uint64_t issued() const noexcept { return next_; }

 private:
  Registry(const ExecutionLane& lane, std::array<std::uint64_t, 2> issuer, std::uint64_t bytes,
           std::uint64_t versions)
      : lane_(&lane),
        pin_(lane.borrow_immutably()),
        authority_(std::make_shared<const IssuerAuthority>(issuer)),
        issuer_(issuer),
        storage_bytes_(bytes),
        storage_versions_(versions),
        configured_(true) {}
  const ExecutionLane* lane_;
  ExecutionLane::ImmutableBorrow pin_;
  std::shared_ptr<const IssuerAuthority> authority_;
  std::array<std::uint64_t, 2> issuer_{};
  std::uint64_t next_ = 0;
  std::map<std::uint64_t, DomainRef> domains_;
  struct StoredValue {
    std::shared_ptr<const void> value;
    std::string type;
    std::uint64_t charge;
  };
  std::map<std::uint64_t, StoredValue> values_;
  std::uint64_t next_value_ = 0, storage_bytes_ = 64ULL * 1024 * 1024, storage_versions_ = 65536,
                used_bytes_ = 0;
  bool configured_ = false;
};
struct Index {
  DomainRef domain;
  std::int64_t gid;
};
inline Index index(DomainRef d, std::int64_t gid) {
  local_require(d && gid >= 0 && gid < d->size(), Code::index, gid);
  return {std::move(d), gid};
}
}  // namespace pops::runtime::program::distributed
