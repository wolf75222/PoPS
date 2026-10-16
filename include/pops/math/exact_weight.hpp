#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace exact_weight_detail {

// Unsigned base-2^32 integer. Every allocating operation checks its limb budget before
// allocation. Arithmetic is host metadata work, never a device kernel or physical operator.
class Natural {
 public:
  std::vector<std::uint32_t> words;
  Natural() = default;
  explicit Natural(std::uint64_t value) {
    if (value)
      words.push_back(static_cast<std::uint32_t>(value));
    if (value >> 32)
      words.push_back(static_cast<std::uint32_t>(value >> 32));
  }
  bool zero() const noexcept { return words.empty(); }
  void trim() {
    while (!words.empty() && words.back() == 0)
      words.pop_back();
  }
  std::size_t bits() const noexcept {
    return zero() ? 0 : 32 * (words.size() - 1) + 32 - std::countl_zero(words.back());
  }
  static void capacity(std::size_t count, std::size_t bytes) {
    if (count > bytes / sizeof(std::uint32_t))
      throw std::length_error("exact coefficient limb budget exhausted");
  }
  int compare(const Natural& other) const noexcept {
    if (words.size() != other.words.size())
      return words.size() < other.words.size() ? -1 : 1;
    for (std::size_t i = words.size(); i != 0; --i)
      if (words[i - 1] != other.words[i - 1])
        return words[i - 1] < other.words[i - 1] ? -1 : 1;
    return 0;
  }
  static Natural add(const Natural& a, const Natural& b, std::size_t bytes) {
    Natural out;
    const auto n = std::max(a.words.size(), b.words.size());
    capacity(n, bytes);
    out.words.resize(n);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t v = (i < a.words.size() ? std::uint64_t(a.words[i]) : 0) +
                              (i < b.words.size() ? std::uint64_t(b.words[i]) : 0) + carry;
      out.words[i] = static_cast<std::uint32_t>(v);
      carry = v >> 32;
    }
    if (carry) {
      capacity(n + 1, bytes);
      out.words.push_back(static_cast<std::uint32_t>(carry));
    }
    return out;
  }
  static Natural subtract(const Natural& a, const Natural& b) {
    if (a.compare(b) < 0)
      throw std::logic_error("negative exact unsigned subtraction");
    Natural out = a;
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < out.words.size(); ++i) {
      const std::uint64_t right = (i < b.words.size() ? std::uint64_t(b.words[i]) : 0) + borrow;
      const std::uint64_t left = out.words[i];
      out.words[i] = static_cast<std::uint32_t>(left - right);
      borrow = left < right;
    }
    out.trim();
    return out;
  }
  static Natural multiply(const Natural& a, const Natural& b, std::size_t bytes) {
    if (a.zero() || b.zero())
      return {};
    if (a.words.size() > bytes / 4 || b.words.size() > bytes / 4 - a.words.size())
      throw std::length_error("exact coefficient multiplication budget exhausted");
    Natural out;
    out.words.assign(a.words.size() + b.words.size(), 0);
    for (std::size_t i = 0; i < a.words.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < b.words.size(); ++j) {
        const std::uint64_t v = std::uint64_t(a.words[i]) * b.words[j] + out.words[i + j] + carry;
        out.words[i + j] = static_cast<std::uint32_t>(v);
        carry = v >> 32;
      }
      out.words[i + b.words.size()] = static_cast<std::uint32_t>(carry);
    }
    out.trim();
    return out;
  }
  static Natural shift(const Natural& a, std::size_t count, std::size_t bytes) {
    if (a.zero())
      return {};
    const auto whole = count / 32;
    const auto part = static_cast<unsigned>(count % 32);
    if (whole > bytes / 4 || a.words.size() > bytes / 4 - whole)
      throw std::length_error("exact coefficient shift budget exhausted");
    Natural out;
    out.words.assign(whole + a.words.size(), 0);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < a.words.size(); ++i) {
      const std::uint64_t value = (std::uint64_t(a.words[i]) << part) | carry;
      out.words[whole + i] = static_cast<std::uint32_t>(value);
      carry = value >> 32;
    }
    if (carry) {
      capacity(out.words.size() + 1, bytes);
      out.words.push_back(static_cast<std::uint32_t>(carry));
    }
    return out;
  }
  void half() {
    std::uint32_t carry = 0;
    for (std::size_t i = words.size(); i != 0; --i) {
      const auto value = words[i - 1];
      words[i - 1] = (value >> 1) | carry;
      carry = value << 31;
    }
    trim();
  }
  static std::pair<Natural, Natural> divide(const Natural& a, const Natural& b, std::size_t bytes) {
    if (b.zero())
      throw std::domain_error("zero exact coefficient denominator");
    if (a.compare(b) < 0)
      return {Natural{}, a};
    const auto steps = a.bits() - b.bits();
    Natural divisor = shift(b, steps, bytes), remainder = a, quotient;
    capacity(steps / 32 + 1, bytes);
    quotient.words.assign(steps / 32 + 1, 0);
    for (std::size_t i = steps + 1; i != 0; --i) {
      const auto bit = i - 1;
      if (remainder.compare(divisor) >= 0) {
        remainder = subtract(remainder, divisor);
        quotient.words[bit / 32] |= std::uint32_t{1} << (bit % 32);
      }
      divisor.half();
    }
    quotient.trim();
    return {std::move(quotient), std::move(remainder)};
  }
  static Natural gcd(Natural a, Natural b, std::size_t bytes) {
    while (!b.zero()) {
      auto r = divide(a, b, bytes).second;
      a = std::move(b);
      b = std::move(r);
    }
    return a;
  }
  std::uint64_t small() const {
    if (words.size() > 2)
      throw std::overflow_error("exact integer does not fit uint64");
    return (words.empty() ? 0 : std::uint64_t(words[0])) |
           (words.size() < 2 ? 0 : std::uint64_t(words[1]) << 32);
  }
  std::string decimal() const {
    if (zero())
      return "0";
    Natural n = *this;
    std::string out;
    while (!n.zero()) {
      std::uint64_t remainder = 0;
      for (std::size_t i = n.words.size(); i != 0; --i) {
        const std::uint64_t v = (remainder << 32) | n.words[i - 1];
        n.words[i - 1] = static_cast<std::uint32_t>(v / 10);
        remainder = v % 10;
      }
      out.push_back(static_cast<char>('0' + remainder));
      n.trim();
    }
    std::reverse(out.begin(), out.end());
    return out;
  }
};
}  // namespace exact_weight_detail

// Canonical arbitrary rational weights. The byte budget bounds stored numerator + denominator
// limbs and each arithmetic temporary. Operators keep the stricter explicit operand budget.
// Integer literals are budget-neutral; an operation on literals alone uses the documented
// default resource budget. There is no fixed denominator/exponent limit. All operations build
// a new value before publication, including resource failures. At most a fixed number of limb
// vectors is live per arithmetic operation; no recursive arithmetic or process-wide budget.
class ExactWeight {
 public:
  static constexpr std::size_t default_max_bytes = 64 * 1024;
  ExactWeight() = default;
  ExactWeight(std::int64_t numerator, std::int64_t denominator = 1, std::size_t max_bytes = 0)
      : negative_(numerator < 0), denominator_(), max_bytes_(max_bytes) {
    if (max_bytes)
      validate_budget_(max_bytes);
    numerator_ = Natural(magnitude_(numerator));
    denominator_ = Natural(magnitude_(denominator));
    if (denominator < 0)
      negative_ = !negative_;
    normalize_();
  }
  template <class Rational>
    requires requires(const Rational& value) {
      value.numerator;
      value.denominator;
    }
  ExactWeight(const Rational& value) : ExactWeight(value.numerator, value.denominator) {}
  bool zero() const noexcept { return numerator_.zero(); }
  bool negative() const noexcept { return negative_; }
  std::size_t storage_bytes() const noexcept {
    return 4 * (numerator_.words.size() + denominator_.words.size());
  }
  std::size_t budget_bytes() const noexcept { return limit_(); }
  ExactWeight with_budget(std::size_t bytes) const {
    validate_budget_(bytes);
    if (storage_bytes() > bytes)
      throw std::length_error("exact coefficient storage budget exhausted");
    ExactWeight out = *this;
    out.max_bytes_ = bytes;
    out.require_storage_();
    return out;
  }
  static ExactWeight from_double(double value, std::size_t bytes = default_max_bytes) {
    validate_budget_(bytes);
    if (!std::isfinite(value))
      throw std::invalid_argument("exact coefficient is not finite");
    ExactWeight out;
    out.max_bytes_ = bytes;
    if (value == 0.0)
      return out;
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    out.negative_ = (bits >> 63) != 0;
    const auto exponent = static_cast<int>((bits >> 52) & 2047);
    const std::uint64_t mantissa =
        (bits & ((UINT64_C(1) << 52) - 1)) | (exponent ? UINT64_C(1) << 52 : 0);
    out.numerator_ = Natural(mantissa);
    const int power = exponent ? exponent - 1023 - 52 : -1074;
    if (power >= 0)
      out.numerator_ = Natural::shift(out.numerator_, static_cast<std::size_t>(power), bytes);
    else
      out.denominator_ = Natural::shift(Natural(1), static_cast<std::size_t>(-power), bytes);
    out.normalize_();
    return out;
  }
  static ExactWeight from_decimal(std::string_view numerator, std::string_view denominator,
                                  std::size_t bytes = default_max_bytes) {
    validate_budget_(bytes);
    const auto parse = [bytes](std::string_view text) {
      if (text.empty() || (text.size() > 1 && text.front() == '0'))
        throw std::invalid_argument("noncanonical exact integer text");
      Natural out;
      for (const char digit : text) {
        if (digit < '0' || digit > '9')
          throw std::invalid_argument("invalid exact integer digit");
        out = Natural::add(Natural::multiply(out, Natural(10), bytes), Natural(digit - '0'), bytes);
      }
      return out;
    };
    ExactWeight out;
    out.max_bytes_ = bytes;
    out.negative_ = !numerator.empty() && numerator.front() == '-';
    const auto magnitude = out.negative_ ? numerator.substr(1) : numerator;
    out.numerator_ = parse(magnitude);
    out.denominator_ = parse(denominator);
    out.normalize_();
    if (out.numerator_text() != numerator || out.denominator_text() != denominator)
      throw std::invalid_argument("noncanonical exact ratio text");
    return out;
  }
  static ExactWeight from_hex(std::string_view numerator, std::string_view denominator,
                              std::size_t bytes = default_max_bytes) {
    validate_budget_(bytes);
    const auto parse = [bytes](std::string_view text) {
      if (!text.starts_with("0x"))
        throw std::invalid_argument("exact hexadecimal integer lacks prefix");
      text.remove_prefix(2);
      if (text.empty() || (text.size() > 1 && text.front() == '0'))
        throw std::invalid_argument("noncanonical exact hexadecimal integer");
      Natural out;
      for (const char digit : text) {
        const int value = digit >= '0' && digit <= '9'   ? digit - '0'
                          : digit >= 'a' && digit <= 'f' ? digit - 'a' + 10
                                                         : -1;
        if (value < 0)
          throw std::invalid_argument("invalid exact hexadecimal digit");
        out = Natural::add(Natural::shift(out, 4, bytes), Natural(value), bytes);
      }
      return out;
    };
    ExactWeight out;
    out.max_bytes_ = bytes;
    out.negative_ = !numerator.empty() && numerator.front() == '-';
    out.numerator_ = parse(out.negative_ ? numerator.substr(1) : numerator);
    out.denominator_ = parse(denominator);
    const auto before = out.encode();
    out.normalize_();
    if (before != out.encode())
      throw std::invalid_argument("noncanonical exact hexadecimal ratio");
    return out;
  }
  double value() const {
    if (zero())
      return 0.0;
    // Exact nearest-even IEEE conversion, including subnormals and ratios whose operands
    // individually exceed double. The constant extra scratch is solely the 53-bit rounding.
    const auto scratch =
        limit_() > std::numeric_limits<std::size_t>::max() - 16 ? limit_() : limit_() + 16;
    const auto nb = numerator_.bits(), db = denominator_.bits();
    if (nb > db && nb - db > 1024)
      return negative_ ? -INFINITY : INFINITY;
    if (db > nb && db - nb > 1075)
      return negative_ ? -0.0 : 0.0;
    int exponent = nb >= db ? static_cast<int>(nb - db) : -static_cast<int>(db - nb);
    if (exponent >= 0) {
      if (numerator_.compare(Natural::shift(denominator_, exponent, scratch)) < 0)
        --exponent;
    } else if (Natural::shift(numerator_, -exponent, scratch).compare(denominator_) < 0)
      --exponent;
    if (exponent > 1023)
      return negative_ ? -INFINITY : INFINITY;
    const int shift = exponent < -1022 ? 1074 : 52 - exponent;
    const auto a = shift >= 0 ? Natural::shift(numerator_, shift, scratch) : numerator_;
    const auto b = shift < 0 ? Natural::shift(denominator_, -shift, scratch) : denominator_;
    auto [quotient, remainder] = Natural::divide(a, b, scratch);
    auto significand = quotient.small();
    const int half = remainder.compare(Natural::subtract(b, remainder));
    if (half > 0 || (half == 0 && (significand & 1)))
      ++significand;
    const double result = std::ldexp(static_cast<double>(significand), -shift);
    return negative_ ? -result : result;
  }
  std::string numerator_text() const { return (negative_ ? "-" : "") + numerator_.decimal(); }
  std::string denominator_text() const { return denominator_.decimal(); }
  std::string text() const { return numerator_text() + "/" + denominator_text(); }
  bool fits_int64() const noexcept {
    return numerator_.words.size() <= 2 && denominator_.words.size() <= 2 &&
           numerator_.small() <= (negative_ ? UINT64_C(1) << 63 : std::uint64_t(INT64_MAX)) &&
           denominator_.small() <= std::uint64_t(INT64_MAX);
  }
  std::int64_t numerator_int64() const {
    if (!fits_int64())
      throw std::overflow_error("exact weight does not fit int64");
    const auto n = numerator_.small();
    return negative_ ? (n == (UINT64_C(1) << 63) ? INT64_MIN : -static_cast<std::int64_t>(n))
                     : static_cast<std::int64_t>(n);
  }
  std::int64_t denominator_int64() const {
    if (!fits_int64())
      throw std::overflow_error("exact weight does not fit int64");
    return static_cast<std::int64_t>(denominator_.small());
  }
  // EW1, sign, two uint64 little-endian limb counts, then little-endian uint32 limbs.
  // No native padding/endianness, no budget in mathematical identity, no alternate encodings.
  std::string encode() const {
    std::string out("EW1", 3);
    out.push_back(negative_ ? 1 : 0);
    append_(out, numerator_.words.size(), 8);
    append_(out, denominator_.words.size(), 8);
    for (auto word : numerator_.words)
      append_(out, word, 4);
    for (auto word : denominator_.words)
      append_(out, word, 4);
    return out;
  }
  static ExactWeight decode(std::string_view bytes, std::size_t max_bytes = default_max_bytes) {
    validate_budget_(max_bytes);
    if (bytes.size() < 20 || bytes.substr(0, 3) != "EW1" || (bytes[3] != 0 && bytes[3] != 1))
      throw std::invalid_argument("invalid exact weight wire header");
    const auto n = read_(bytes, 4, 8), d = read_(bytes, 12, 8);
    if (!d || n > max_bytes / 4 || d > max_bytes / 4 - n)
      throw std::length_error("exact coefficient wire budget exhausted");
    if (n + d != (bytes.size() - 20) / 4 || (bytes.size() - 20) % 4)
      throw std::invalid_argument("invalid exact weight wire length");
    ExactWeight out;
    out.max_bytes_ = max_bytes;
    out.negative_ = bytes[3] != 0;
    out.numerator_.words.resize(static_cast<std::size_t>(n));
    out.denominator_.words.resize(static_cast<std::size_t>(d));
    std::size_t offset = 20;
    for (auto& word : out.numerator_.words) {
      word = static_cast<std::uint32_t>(read_(bytes, offset, 4));
      offset += 4;
    }
    for (auto& word : out.denominator_.words) {
      word = static_cast<std::uint32_t>(read_(bytes, offset, 4));
      offset += 4;
    }
    if ((!out.zero() && !out.numerator_.words.back()) || !out.denominator_.words.back())
      throw std::invalid_argument("noncanonical exact weight limbs");
    const auto before = out.encode();
    out.normalize_();
    if (before != out.encode())
      throw std::invalid_argument("noncanonical exact weight ratio");
    return out;
  }
  friend bool operator==(const ExactWeight& a, const ExactWeight& b) noexcept {
    return a.negative_ == b.negative_ && a.numerator_.compare(b.numerator_) == 0 &&
           a.denominator_.compare(b.denominator_) == 0;
  }
  friend ExactWeight operator-(const ExactWeight& a) {
    auto out = a;
    if (!out.zero())
      out.negative_ = !out.negative_;
    return out;
  }
  friend ExactWeight operator+(const ExactWeight& a, const ExactWeight& b) {
    const auto bytes = joint_budget_(a, b);
    const auto gcd = Natural::gcd(a.denominator_, b.denominator_, bytes);
    const auto ad = Natural::divide(a.denominator_, gcd, bytes).first;
    const auto bd = Natural::divide(b.denominator_, gcd, bytes).first;
    const auto left = Natural::multiply(a.numerator_, bd, bytes),
               right = Natural::multiply(b.numerator_, ad, bytes);
    ExactWeight out;
    out.max_bytes_ = bytes;
    out.denominator_ = Natural::multiply(ad, b.denominator_, bytes);
    if (a.negative_ == b.negative_) {
      out.numerator_ = Natural::add(left, right, bytes);
      out.negative_ = a.negative_;
    } else if (left.compare(right) >= 0) {
      out.numerator_ = Natural::subtract(left, right);
      out.negative_ = a.negative_;
    } else {
      out.numerator_ = Natural::subtract(right, left);
      out.negative_ = b.negative_;
    }
    out.normalize_();
    return out;
  }
  friend ExactWeight operator-(const ExactWeight& a, const ExactWeight& b) { return a + (-b); }
  friend ExactWeight operator*(const ExactWeight& a, const ExactWeight& b) {
    const auto bytes = joint_budget_(a, b);
    const auto g1 = Natural::gcd(a.numerator_, b.denominator_, bytes),
               g2 = Natural::gcd(b.numerator_, a.denominator_, bytes);
    ExactWeight out;
    out.max_bytes_ = bytes;
    out.negative_ = a.negative_ != b.negative_;
    out.numerator_ = Natural::multiply(Natural::divide(a.numerator_, g1, bytes).first,
                                       Natural::divide(b.numerator_, g2, bytes).first, bytes);
    out.denominator_ = Natural::multiply(Natural::divide(a.denominator_, g2, bytes).first,
                                         Natural::divide(b.denominator_, g1, bytes).first, bytes);
    out.normalize_();
    return out;
  }
  friend ExactWeight operator/(const ExactWeight& a, const ExactWeight& b) {
    if (b.zero())
      throw std::domain_error("zero exact coefficient divisor");
    (void)joint_budget_(a, b);
    ExactWeight inverse = b;
    std::swap(inverse.numerator_, inverse.denominator_);
    return a * inverse;
  }

 private:
  using Natural = exact_weight_detail::Natural;
  bool negative_ = false;
  Natural numerator_;
  Natural denominator_{1};
  std::size_t max_bytes_ = 0;
  static void validate_budget_(std::size_t bytes) {
    if (bytes < 8 || bytes > std::numeric_limits<std::size_t>::max() - 20)
      throw std::invalid_argument("exact coefficient byte budget is outside the native range");
  }
  std::size_t limit_() const noexcept { return max_bytes_ ? max_bytes_ : default_max_bytes; }
  static std::size_t joint_budget_(const ExactWeight& a, const ExactWeight& b) {
    const auto bytes = !a.max_bytes_   ? b.limit_()
                       : !b.max_bytes_ ? a.limit_()
                                       : std::min(a.max_bytes_, b.max_bytes_);
    if (a.storage_bytes() > bytes || b.storage_bytes() > bytes)
      throw std::length_error("exact coefficient operands exceed their joint byte budget");
    return bytes;
  }
  static std::uint64_t magnitude_(std::int64_t v) noexcept {
    return v < 0 ? std::uint64_t(-(v + 1)) + 1 : std::uint64_t(v);
  }
  void require_storage_() const {
    if (storage_bytes() > limit_())
      throw std::length_error("exact coefficient storage budget exhausted");
  }
  void normalize_() {
    if (denominator_.zero())
      throw std::domain_error("zero exact coefficient denominator");
    if (zero()) {
      negative_ = false;
      denominator_ = Natural(1);
    } else {
      const auto divisor = Natural::gcd(numerator_, denominator_, limit_());
      numerator_ = Natural::divide(numerator_, divisor, limit_()).first;
      denominator_ = Natural::divide(denominator_, divisor, limit_()).first;
    }
    require_storage_();
  }
  static void append_(std::string& out, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i)
      out.push_back(static_cast<char>((value >> (8 * i)) & 255));
  }
  static std::uint64_t read_(std::string_view in, std::size_t offset, unsigned count) {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < count; ++i)
      out |= std::uint64_t(static_cast<unsigned char>(in[offset + i])) << (8 * i);
    return out;
  }
};
}  // namespace pops
