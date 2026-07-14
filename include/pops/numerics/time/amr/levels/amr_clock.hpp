#pragma once

#include <cstdint>
#include <limits>
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
  Rational(std::int64_t n, std::int64_t d) {
    if (d == 0)
      throw std::invalid_argument("AMR clock rational denominator must be non-zero");
    const bool negative = (n < 0) != (d < 0);
    std::uint64_t numerator_magnitude = magnitude_(n);
    std::uint64_t denominator_magnitude = magnitude_(d);
    const std::uint64_t g = std::gcd(numerator_magnitude, denominator_magnitude);
    numerator_magnitude /= g;
    denominator_magnitude /= g;
    if (denominator_magnitude >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::overflow_error("AMR clock rational exceeds the exact int64 range");
    numerator = signed_magnitude_(numerator_magnitude, negative && numerator_magnitude != 0);
    denominator = static_cast<std::int64_t>(denominator_magnitude);
  }

  double value() const {
    return static_cast<double>(numerator) / static_cast<double>(denominator);
  }
  bool integral() const { return denominator == 1; }

  friend bool operator==(const Rational&, const Rational&) = default;
  friend bool operator<(Rational a, Rational b) {
    if (a.numerator < 0 && b.numerator >= 0)
      return true;
    if (a.numerator >= 0 && b.numerator < 0)
      return false;
    if (a.numerator < 0)
      return positive_less_(magnitude_(b.numerator), b.denominator,
                            magnitude_(a.numerator), a.denominator);
    return positive_less_(magnitude_(a.numerator), a.denominator,
                          magnitude_(b.numerator), b.denominator);
  }
  friend Rational operator+(Rational a, Rational b) {
    return add_(a, b, false);
  }
  friend Rational operator-(Rational a, Rational b) {
    return add_(a, b, true);
  }
  friend Rational operator*(Rational a, Rational b) {
    // Cross-cancel before either product.  This both keeps the canonical result in int64 whenever
    // possible and makes every remaining multiplication explicitly checked.
    const std::uint64_t ga = std::gcd(magnitude_(a.numerator),
                                     static_cast<std::uint64_t>(b.denominator));
    const std::uint64_t gb = std::gcd(magnitude_(b.numerator),
                                     static_cast<std::uint64_t>(a.denominator));
    return {checked_mul_(divide_exact_(a.numerator, ga), divide_exact_(b.numerator, gb)),
            checked_mul_(a.denominator / gb, b.denominator / ga)};
  }
  friend Rational operator/(Rational a, Rational b) {
    if (b.numerator == 0)
      throw std::invalid_argument("AMR clock rational division by zero");
    // Divide directly rather than materializing b's reciprocal: 1/INT64_MIN cannot be represented
    // with a positive int64 denominator, while INT64_MIN/INT64_MIN is exactly representable.
    const std::uint64_t gn = std::gcd(magnitude_(a.numerator), magnitude_(b.numerator));
    const std::uint64_t gd = std::gcd(static_cast<std::uint64_t>(a.denominator),
                                     static_cast<std::uint64_t>(b.denominator));
    return {checked_mul_(divide_exact_(a.numerator, gn),
                         divide_exact_(b.denominator, gd)),
            checked_mul_(divide_exact_(a.denominator, gd),
                         divide_exact_(b.numerator, gn))};
  }

 private:
  static std::uint64_t magnitude_(std::int64_t value) {
    // -(INT64_MIN) is undefined, but -(value + 1) is representable for every negative value.
    return value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1u
                     : static_cast<std::uint64_t>(value);
  }

  static std::int64_t signed_magnitude_(std::uint64_t magnitude, bool negative) {
    const std::uint64_t negative_limit = std::uint64_t{1} << 63u;
    const std::uint64_t limit = negative
        ? negative_limit
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (magnitude > limit)
      throw std::overflow_error("AMR clock rational arithmetic overflow");
    if (negative && magnitude == negative_limit)
      return std::numeric_limits<std::int64_t>::min();
    const std::int64_t value = static_cast<std::int64_t>(magnitude);
    return negative ? -value : value;
  }

  static std::int64_t divide_exact_(std::int64_t value, std::uint64_t divisor) {
    return signed_magnitude_(magnitude_(value) / divisor, value < 0);
  }

  static std::int64_t checked_mul_(std::int64_t a, std::int64_t b) {
    if (a == 0 || b == 0) return 0;
    const bool negative = (a < 0) != (b < 0);
    const std::uint64_t left = magnitude_(a);
    const std::uint64_t right = magnitude_(b);
    const std::uint64_t limit = negative
        ? (std::uint64_t{1} << 63u)
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (left > limit / right)
      throw std::overflow_error("AMR clock rational arithmetic overflow");
    return signed_magnitude_(left * right, negative);
  }

  struct UInt128 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
  };

  struct Signed128 {
    UInt128 magnitude;
    bool negative = false;
  };

  static UInt128 multiply_wide_(std::uint64_t a, std::uint64_t b) {
    constexpr std::uint64_t mask = 0xffffffffu;
    const std::uint64_t a0 = a & mask;
    const std::uint64_t a1 = a >> 32u;
    const std::uint64_t b0 = b & mask;
    const std::uint64_t b1 = b >> 32u;
    const std::uint64_t p00 = a0 * b0;
    const std::uint64_t p01 = a0 * b1;
    const std::uint64_t p10 = a1 * b0;
    const std::uint64_t p11 = a1 * b1;
    const std::uint64_t middle =
        (p00 >> 32u) + (p01 & mask) + (p10 & mask);
    return {p11 + (p01 >> 32u) + (p10 >> 32u) + (middle >> 32u),
            (middle << 32u) | (p00 & mask)};
  }

  static bool less_(UInt128 a, UInt128 b) {
    return a.high != b.high ? a.high < b.high : a.low < b.low;
  }

  static UInt128 add_(UInt128 a, UInt128 b) {
    const std::uint64_t low = a.low + b.low;
    return {a.high + b.high + (low < a.low ? 1u : 0u), low};
  }

  static UInt128 subtract_(UInt128 a, UInt128 b) {
    const std::uint64_t low = a.low - b.low;
    return {a.high - b.high - (a.low < b.low ? 1u : 0u), low};
  }

  static Signed128 add_products_(std::int64_t a, std::uint64_t a_factor,
                                 std::int64_t b, std::uint64_t b_factor,
                                 bool subtract_b) {
    const UInt128 left = multiply_wide_(magnitude_(a), a_factor);
    const UInt128 right = multiply_wide_(magnitude_(b), b_factor);
    const bool left_negative = a < 0;
    const bool right_negative = (b < 0) != subtract_b;
    if (left_negative == right_negative)
      return {add_(left, right), left_negative};
    if (less_(left, right))
      return {subtract_(right, left), right_negative};
    const UInt128 difference = subtract_(left, right);
    return {difference, (difference.high != 0 || difference.low != 0) && left_negative};
  }

  static bool bit_(UInt128 value, int index) {
    return index < 64 ? ((value.low >> index) & 1u) != 0
                      : ((value.high >> (index - 64)) & 1u) != 0;
  }

  static void set_bit_(UInt128& value, int index) {
    if (index < 64)
      value.low |= std::uint64_t{1} << index;
    else
      value.high |= std::uint64_t{1} << (index - 64);
  }

  static std::uint64_t remainder_(UInt128 value, std::uint64_t divisor) {
    std::uint64_t remainder = 0;
    for (int index = 127; index >= 0; --index) {
      remainder = remainder * 2u + (bit_(value, index) ? 1u : 0u);
      if (remainder >= divisor) remainder -= divisor;
    }
    return remainder;
  }

  static UInt128 divide_(UInt128 value, std::uint64_t divisor) {
    UInt128 quotient;
    std::uint64_t remainder = 0;
    for (int index = 127; index >= 0; --index) {
      remainder = remainder * 2u + (bit_(value, index) ? 1u : 0u);
      if (remainder < divisor) continue;
      remainder -= divisor;
      set_bit_(quotient, index);
    }
    return quotient;
  }

  static Rational add_(Rational a, Rational b, bool subtract) {
    const std::uint64_t common = std::gcd(
        static_cast<std::uint64_t>(a.denominator),
        static_cast<std::uint64_t>(b.denominator));
    const Signed128 numerator = add_products_(
        a.numerator, static_cast<std::uint64_t>(b.denominator) / common,
        b.numerator, static_cast<std::uint64_t>(a.denominator) / common, subtract);
    // Any factor common to the numerator and the shared denominator can be removed before forming
    // the least-common denominator, avoiding a spurious denominator overflow.
    const std::uint64_t reduction = std::gcd(remainder_(numerator.magnitude, common), common);
    const UInt128 reduced = divide_(numerator.magnitude, reduction);
    if (reduced.high != 0)
      throw std::overflow_error("AMR clock rational arithmetic overflow");
    return {signed_magnitude_(reduced.low, numerator.negative),
            checked_mul_(a.denominator / reduction, b.denominator / common)};
  }

  // Exact overflow-free comparison of two non-negative rationals.  Continued-fraction quotients
  // replace the unsafe cross products; taking a reciprocal reverses the ordering at each iteration.
  static bool positive_less_(std::uint64_t an, std::uint64_t ad,
                             std::uint64_t bn, std::uint64_t bd) {
    bool reversed = false;
    for (;;) {
      const std::uint64_t aq = an / ad;
      const std::uint64_t bq = bn / bd;
      if (aq != bq)
        return reversed ? aq > bq : aq < bq;
      const std::uint64_t ar = an % ad;
      const std::uint64_t br = bn % bd;
      if (ar == 0 || br == 0) {
        if (ar == br)
          return false;
        return reversed ? br == 0 : ar == 0;
      }
      an = ad;
      ad = ar;
      bn = bd;
      bd = br;
      reversed = !reversed;
    }
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
