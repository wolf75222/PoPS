#pragma once

#include <pops/runtime/program/accepted_exchange.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// Algebra of already evaluated face records. Immutable expressions may be owned by native
/// History/Continuation values; this index itself owns no scientific field or acceptance.
class FieldExchangeExpressions {
 public:
  using Records = std::vector<ExchangeRecord>;
  using Cell = std::vector<std::int64_t>;
  struct Limits {
    std::size_t max_bases = 0;
    std::size_t max_terms = 0;
    std::size_t max_records_per_basis = 0;
    std::size_t max_bytes = 0;
    bool operator==(const Limits&) const = default;
  };
  struct Basis {
    std::string identity;
    int program_block = 0;
    Records records;
    std::vector<Cell> cells;
  };
  struct Term {
    std::shared_ptr<const Basis> basis;
    double coefficient = 0;
  };
  struct Expression {
    bool exact = true;
    std::vector<Term> terms;
  };
  using Ref = std::shared_ptr<const Expression>;
  struct WeightedField {
    const void* field;
    double coefficient;
  };
  struct Publication {
    Records records;
    std::vector<std::pair<int, Ref>> published;
  };

  void configure(Limits limits) {
    if (!limits.max_bases || !limits.max_terms || !limits.max_records_per_basis ||
        !limits.max_bytes)
      throw std::invalid_argument("field exchange limits require explicit positive capacities");
    if (limits_.max_bases && limits_ != limits)
      throw std::invalid_argument("field exchange resource contract changed after installation");
    limits_ = limits;
  }
  const Limits& limits() const noexcept { return limits_; }
  void reset(const void* field) noexcept {
    bindings_.erase(field);
    before_write_.erase(field);
  }
  Ref expression(const void* field) const { return lookup_(field); }
  void restore(const void* field, Ref expression) {
    require_expression(expression, limits_);
    if (expression)
      for (const auto& term : expression->terms) {
        for (auto it = bases_.begin(); it != bases_.end();) {
          if (it->second.expired())
            it = bases_.erase(it);
          else
            ++it;
        }
        bases_.insert_or_assign(term.basis->identity, term.basis);
      }
    install_(field, std::move(expression));
  }
  void restore_retained(const void* field, Ref expression,
                        const std::shared_ptr<const void>& lifetime) {
    if (!lifetime)
      throw std::invalid_argument("retained exchange field has no storage lifetime");
    restore(field, std::move(expression));
    if (auto found = bindings_.find(field); found != bindings_.end()) {
      found->second.owner = lifetime;
      found->second.retained = true;
    }
  }

  void clear() noexcept {
    bindings_.clear();
    before_write_.clear();
    published_.clear();
    capture_sequence_ = 0;
  }

  /// A write starts with an unproved expression whenever any input carries faces. An affine
  /// producer replaces this marker with its proof; an arbitrary transform cannot erase it.
  void prepare_write(const void* output, std::span<const void* const> inputs) {
    if (output == nullptr)
      throw std::invalid_argument("exchange expression output is null");
    bool tainted = false;
    for (const auto input : inputs) {
      if (input == nullptr)
        throw std::invalid_argument("exchange expression input is null");
      const auto value = lookup_(input);
      tainted = tainted || (value && (!value->exact || !value->terms.empty()));
    }
    const auto previous = lookup_(output);
    Ref next;
    if (tainted)
      next = std::make_shared<const Expression>(Expression{false, {}});
    before_write_.insert_or_assign(output, previous);
    if (next)
      bindings_.insert_or_assign(output, Binding{std::move(next), {}, false});
    else
      bindings_.erase(output);
  }

  void finish_write(const void* output) noexcept { before_write_.erase(output); }

  void copy(const void* output, const void* source) { install_(output, source_(source, output)); }

  void copy_retained(const void* output, const void* source,
                     const std::shared_ptr<const void>& lifetime) {
    if (!lifetime)
      throw std::invalid_argument("retained exchange field has no storage lifetime");
    copy(output, source);
    if (auto found = bindings_.find(output); found != bindings_.end()) {
      found->second.owner = lifetime;
      found->second.retained = true;
    }
  }

  void combine(const void* output, std::span<const WeightedField> fields) {
    Expression next;
    for (const auto& field : fields) {
      if (field.field == nullptr || !std::isfinite(field.coefficient))
        throw std::invalid_argument("exchange expression has an invalid affine coefficient");
      if (field.coefficient == 0)
        continue;
      const auto source = source_(field.field, output);
      if (!source)
        continue;
      next.exact = next.exact && source->exact;
      for (const auto& term : source->terms) {
        const auto factor = field.coefficient * term.coefficient;
        if (!std::isfinite(factor))
          throw std::overflow_error("exchange expression affine product is nonfinite");
        auto found = std::find_if(next.terms.begin(), next.terms.end(), [&](const Term& other) {
          return other.basis->identity == term.basis->identity;
        });
        if (found == next.terms.end()) {
          next.terms.push_back({term.basis, factor});
        } else {
          if (!same_basis(*found->basis, *term.basis))
            throw std::invalid_argument(
                "same field exchange source identity has different records");
          found->coefficient += factor;
          if (!std::isfinite(found->coefficient))
            throw std::overflow_error("exchange expression affine sum is nonfinite");
        }
      }
    }
    std::erase_if(next.terms, [](const Term& term) { return term.coefficient == 0; });
    require_expression(std::make_shared<const Expression>(next), limits_);
    install_(output, next.exact && next.terms.empty()
                         ? Ref{}
                         : std::make_shared<const Expression>(std::move(next)));
  }

  /// The qualifier runs while the evaluation's actual clock/interval is still active. The
  /// per-attempt ordinal separates repeated evaluations of one SSA without changing the
  /// source evaluation identity used by Python-authored external-trace selectors.
  std::uint64_t next_capture_sequence() const {
    if (capture_sequence_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("exchange evaluation ordinal exhausted uint64_t");
    return capture_sequence_ + 1;
  }

  static void qualify_capture(ExchangeRecord& record,
                              const multiblock::BoundaryEvaluationPoint& point,
                              std::uint64_t sequence) {
    if (sequence == 0)
      throw std::invalid_argument("exchange evaluation ordinal is zero");
    record.qualify_runtime_point(point);
    const auto suffix = "/" + std::to_string(record.source_evaluation_identity.size()) + ":" +
                        record.source_evaluation_identity;
    record.evaluation_context.insert(record.evaluation_context.size() - suffix.size(),
                                     "/capture:" + std::to_string(sequence));
    record.validate();
  }

  void capture(const void* output, Records records, std::uint64_t sequence, std::string identity,
               int program_block, std::vector<Cell> cells) {
    if (sequence != next_capture_sequence())
      throw std::invalid_argument("exchange evaluation ordinal is not the next issued capture");
    Basis captured{std::move(identity), program_block, std::move(records), std::move(cells)};
    canonicalize_basis(captured);
    auto basis = std::make_shared<const Basis>(std::move(captured));
    auto expression = std::make_shared<const Expression>(Expression{true, {{std::move(basis), 1}}});
    require_expression(expression, limits_);
    // The pool sees retained roots without keeping their data alive. A failed capture never
    // installs its expression or consumes the per-attempt evaluation ordinal.
    for (auto it = bases_.begin(); it != bases_.end();) {
      if (it->second.expired())
        it = bases_.erase(it);
      else
        ++it;
    }
    bases_.insert_or_assign(expression->terms.front().basis->identity,
                            expression->terms.front().basis);
    install_(output, std::move(expression));
    capture_sequence_ = sequence;
  }

  Publication prepare_publication(
      int block, const void* accepted,
      const multiblock::BoundaryEvaluationPoint* delivery = nullptr) const {
    Publication result;
    result.published = published_;
    const auto expression = lookup_(accepted);
    if (!expression)
      return result;
    require_expression(expression, limits_);
    if (!expression->exact)
      throw std::invalid_argument(
          "accepted transform hides stencil faces outside a proved affine expression");
    if (std::find(result.published.begin(), result.published.end(),
                  std::pair<int, Ref>{block, expression}) != result.published.end())
      return result;
    std::size_t count = 0;
    for (const auto& term : expression->terms) {
      if (!term.basis || term.basis->records.size() > result.records.max_size() - count)
        throw std::length_error("exchange expression exceeds the native record allocation size");
      count += term.basis->records.size();
    }
    result.records.reserve(count);
    for (const auto& term : expression->terms)
      for (auto record : term.basis->records) {
        record.temporal_weight *= term.coefficient;
        if (delivery) {
          const auto source_context = record.evaluation_context;
          const auto source_evaluation = record.source_evaluation_identity;
          record.evaluation_context = source_context;
          record.qualify_runtime_point(*delivery);
          record.evaluation_context = "pops.exchange.delivery.v1/" + record.evaluation_context;
          record.source_evaluation_identity = source_evaluation;
        }
        record.validate();
        result.records.push_back(std::move(record));
      }
    result.published.emplace_back(block, expression);
    return result;
  }

  void commit_publication(Publication&& publication) noexcept {
    published_.swap(publication.published);
  }

  static bool same_basis(const Basis& left, const Basis& right) noexcept;
  static void require_expression(const Ref& expression, const Limits& limits);
  static void canonicalize_basis(Basis& basis);
  static std::vector<std::uint8_t> encode_roots(std::span<const Ref> roots, const Limits& limits);
  static std::vector<Ref> decode_roots(std::span<const std::uint8_t> bytes, const Limits& limits);
  static std::string collective_identity(const Ref& expression, const Limits& limits);
  static std::size_t decoded_storage_bound(std::span<const std::uint8_t> bytes,
                                           const Limits& limits);
  static std::size_t rematerialized_storage_bound(std::span<const Ref> sources,
                                                  const Limits& limits);
  using RematerializationCache = std::map<std::string, std::shared_ptr<const Basis>>;
  template <class OwnsCell>
  static Ref rematerialize(std::span<const Ref> sources, const Limits& limits, OwnsCell owns_cell,
                           RematerializationCache* cache = nullptr);

 private:
  struct Binding {
    Ref expression;
    std::weak_ptr<const void> owner;
    bool retained = false;
  };
  Ref lookup_(const void* field) const {
    const auto found = bindings_.find(field);
    return found == bindings_.end() || (found->second.retained && found->second.owner.expired())
               ? Ref{}
               : found->second.expression;
  }
  Ref source_(const void* field, const void* output) const {
    if (field == output) {
      const auto found = before_write_.find(output);
      if (found != before_write_.end())
        return found->second;
    }
    return lookup_(field);
  }
  void install_(const void* field, Ref expression) {
    if (field == nullptr)
      throw std::invalid_argument("exchange expression field is null");
    if (expression)
      bindings_.insert_or_assign(field, Binding{std::move(expression), {}, false});
    else
      bindings_.erase(field);
    before_write_.erase(field);
  }

  std::map<const void*, Binding> bindings_;
  std::map<const void*, Ref> before_write_;
  std::vector<std::pair<int, Ref>> published_;
  std::uint64_t capture_sequence_ = 0;
  Limits limits_;
  std::map<std::string, std::weak_ptr<const Basis>> bases_;
};

#include <pops/runtime/program/field_exchange_expression_codec.inc>

}  // namespace pops::runtime::program
