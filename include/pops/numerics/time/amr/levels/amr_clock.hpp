#pragma once

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::amr {

/// Exact rational used by the AMR scheduler.  Scheduling decisions must not be
/// reconstructed from rounded physical times.
struct Rational {
  std::int64_t numerator = 0;
  std::int64_t denominator = 1;

  Rational() = default;
  Rational(std::int64_t n, std::int64_t d) : numerator(n), denominator(d) {
    if (denominator == 0)
      throw std::invalid_argument("AMR clock rational denominator must be non-zero");
    if (denominator < 0) {
      numerator = -numerator;
      denominator = -denominator;
    }
    const std::int64_t g = std::gcd(numerator < 0 ? -numerator : numerator, denominator);
    numerator /= g;
    denominator /= g;
  }

  double value() const {
    return static_cast<double>(numerator) / static_cast<double>(denominator);
  }
  bool integral() const { return denominator == 1; }

  friend bool operator==(const Rational&, const Rational&) = default;
  friend bool operator<(Rational a, Rational b) {
    return a.numerator * b.denominator < b.numerator * a.denominator;
  }
  friend Rational operator+(Rational a, Rational b) {
    return {a.numerator * b.denominator + b.numerator * a.denominator,
            a.denominator * b.denominator};
  }
  friend Rational operator-(Rational a, Rational b) {
    return {a.numerator * b.denominator - b.numerator * a.denominator,
            a.denominator * b.denominator};
  }
  friend Rational operator*(Rational a, Rational b) {
    return {a.numerator * b.numerator, a.denominator * b.denominator};
  }
  friend Rational operator/(Rational a, Rational b) {
    if (b.numerator == 0)
      throw std::invalid_argument("AMR clock rational division by zero");
    return {a.numerator * b.denominator, a.denominator * b.numerator};
  }
};

/// A clock stamp is qualified by level and accepted macro step.  phase is the
/// exact position in that macro step, not a floating-point reconstruction.
struct ClockStamp {
  int level = 0;
  std::int64_t macro_step = 0;
  Rational phase{};
  double physical_time = 0.0;

  friend bool operator==(const ClockStamp&, const ClockStamp&) = default;
};

struct ClockWindow {
  ClockStamp begin;
  ClockStamp end;

  Rational alpha(const ClockStamp& target) const {
    if (begin.level != end.level || target.level != begin.level ||
        begin.macro_step != end.macro_step || target.macro_step != begin.macro_step ||
        !(begin.phase < end.phase) || target.phase < begin.phase || end.phase < target.phase)
      throw std::runtime_error("target is outside its qualified AMR clock window");
    return (target.phase - begin.phase) / (end.phase - begin.phase);
  }
};

enum class RemainderPolicy {
  /// A parent interval must contain an integral number of child intervals.
  IntegralOnly,
  /// A shorter, explicitly identified final child interval closes the parent window.
  ExplicitFinalSubstep,
};

struct ChildSubstep {
  ClockWindow window;
  bool is_declared_remainder = false;
};

/// Explicit parent/child temporal relation.  temporal_ratio is
/// parent_dt/nominal_child_dt and is independent from spatial refinement.
class ParentChildClockRelation {
 public:
  ParentChildClockRelation(int parent_level, int child_level, Rational temporal_ratio,
                           RemainderPolicy remainder_policy)
      : parent_level_(parent_level),
        child_level_(child_level),
        ratio_(temporal_ratio),
        remainder_policy_(remainder_policy) {
    if (parent_level < 0 || child_level != parent_level + 1 ||
        temporal_ratio < Rational(1, 1))
      throw std::invalid_argument("invalid AMR parent/child clock relation");
  }

  int parent_level() const { return parent_level_; }
  int child_level() const { return child_level_; }
  Rational temporal_ratio() const { return ratio_; }
  RemainderPolicy remainder_policy() const { return remainder_policy_; }

  std::vector<ChildSubstep> partition(const ClockWindow& parent) const {
    if (parent.begin.level != parent_level_ || parent.end.level != parent_level_ ||
        !(parent.begin.phase < parent.end.phase) ||
        !(parent.begin.physical_time < parent.end.physical_time))
      throw std::runtime_error("AMR parent clock window does not match its relation");
    if (!ratio_.integral() && remainder_policy_ == RemainderPolicy::IntegralOnly)
      throw std::runtime_error(
          "non-integral AMR temporal relation requires an explicit remainder policy");

    const Rational span = parent.end.phase - parent.begin.phase;
    const Rational nominal = span / ratio_;
    const std::int64_t full = ratio_.numerator / ratio_.denominator;
    std::vector<ChildSubstep> result;
    result.reserve(static_cast<std::size_t>(full + (ratio_.integral() ? 0 : 1)));
    Rational cursor = parent.begin.phase;
    for (std::int64_t s = 0; s < full; ++s) {
      const Rational next = cursor + nominal;
      result.push_back({make_child_window_(parent, cursor, next), false});
      cursor = next;
    }
    if (cursor < parent.end.phase)
      result.push_back({make_child_window_(parent, cursor, parent.end.phase), true});
    return result;
  }

 private:
  ClockWindow make_child_window_(const ClockWindow& parent, Rational begin, Rational end) const {
    const double span_time = parent.end.physical_time - parent.begin.physical_time;
    const Rational parent_span = parent.end.phase - parent.begin.phase;
    const auto physical = [&](Rational phase) {
      return parent.begin.physical_time +
             span_time * ((phase - parent.begin.phase) / parent_span).value();
    };
    return {{child_level_, parent.begin.macro_step, begin, physical(begin)},
            {child_level_, parent.begin.macro_step, end, physical(end)}};
  }

  int parent_level_;
  int child_level_;
  Rational ratio_;
  RemainderPolicy remainder_policy_;
};

/// Fully qualified history identity.  A ring cannot alias another owner,
/// state, representation space, or level clock merely because labels match.
struct HistoryIdentity {
  std::string owner;
  std::string state;
  std::string space;
  int level = 0;
  ClockStamp clock;

  friend bool operator<(const HistoryIdentity& a, const HistoryIdentity& b) {
    return std::tie(a.owner, a.state, a.space, a.level, a.clock.macro_step,
                    a.clock.phase.numerator, a.clock.phase.denominator) <
           std::tie(b.owner, b.state, b.space, b.level, b.clock.macro_step,
                    b.clock.phase.numerator, b.clock.phase.denominator);
  }
};

}  // namespace pops::amr
