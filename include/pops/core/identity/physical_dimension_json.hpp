#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::identity {
namespace physical_dimension_json_detail {

// The exact json.dumps(..., sort_keys=True, separators=(",", ":"),
// ensure_ascii=True) image of PhysicalDimension.to_data(), not a general JSON
// reader. Integer arithmetic below is decimal and has no machine-integer bound.
class Reader {
 public:
  explicit Reader(std::string_view text) : text_(text) {}

  void require() {
    token("{\"kind\":\"physical_dimension\",\"powers\":[");
    std::u32string previous;
    bool first = true;
    if (!take(']')) {
      do {
        token("[");
        auto name = string();
        if (name.empty() || (!first && !(previous < name))) fail();
        first = false;
        previous = std::move(name);
        token(",");
        take('-');
        const auto numerator = positive_integer();
        token(",");
        const auto denominator = positive_integer();
        if (!coprime(numerator, denominator)) fail();
        token("]");
      } while (take(','));
      token("]");
    }
    token("}");
    if (pos_ != text_.size()) fail();
  }

 private:
  [[noreturn]] static void fail() {
    throw std::invalid_argument("physical dimension requires canonical typed JSON units");
  }
  bool take(char ch) {
    if (pos_ < text_.size() && text_[pos_] == ch) { ++pos_; return true; }
    return false;
  }
  void token(std::string_view value) {
    if (text_.substr(pos_, value.size()) != value) fail();
    pos_ += value.size();
  }
  unsigned hex4() {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      if (pos_ == text_.size()) fail();
      const char ch = text_[pos_++];
      if (!(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f')) fail();
      value = value * 16 + (ch <= '9' ? ch - '0' : ch - 'a' + 10);
    }
    return value;
  }
  std::u32string string() {
    token("\"");
    std::u32string result;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      unsigned ch = static_cast<unsigned char>(text_[pos_++]);
      if (ch == '\\') {
        if (pos_ == text_.size()) fail();
        switch (text_[pos_++]) {
          case '"': ch = '"'; break;
          case '\\': ch = '\\'; break;
          case 'b': ch = 8; break;
          case 'f': ch = 12; break;
          case 'n': ch = 10; break;
          case 'r': ch = 13; break;
          case 't': ch = 9; break;
          case 'u': {
            ch = hex4();
            // Python uses short escapes for these controls, and leaves printable
            // ASCII literal. DEL and non-ASCII are escaped with lowercase hex.
            if ((ch >= 32 && ch < 127) || ch == 8 || ch == 9 || ch == 10 ||
                ch == 12 || ch == 13) fail();
            if (ch >= 0xd800 && ch <= 0xdbff &&
                text_.substr(pos_, 2) == "\\u") {
              const auto saved = pos_;
              pos_ += 2;
              const unsigned low = hex4();
              if (low >= 0xdc00 && low <= 0xdfff)
                ch = 0x10000 + ((ch - 0xd800) << 10) + low - 0xdc00;
              else
                pos_ = saved;
            }
            break;
          }
          default: fail();
        }
      } else if (ch < 32 || ch >= 127) {
        fail();
      }
      result.push_back(static_cast<char32_t>(ch));
    }
    token("\"");
    return result;
  }
  std::string positive_integer() {
    const auto begin = pos_;
    if (pos_ == text_.size() || text_[pos_] < '1' || text_[pos_] > '9') fail();
    do { ++pos_; } while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9');
    return std::string(text_.substr(begin, pos_ - begin));
  }
  static bool less(const std::string& a, const std::string& b) {
    return a.size() < b.size() || (a.size() == b.size() && a < b);
  }
  static void subtract(std::string& a, const std::string& b) {
    int borrow = 0;
    for (std::size_t offset = 0; offset < a.size(); ++offset) {
      const auto i = a.size() - 1 - offset;
      int digit = a[i] - '0' - borrow;
      if (offset < b.size()) digit -= b[b.size() - 1 - offset] - '0';
      borrow = digit < 0;
      a[i] = static_cast<char>('0' + digit + (borrow ? 10 : 0));
    }
    const auto first = a.find_first_not_of('0');
    a = first == std::string::npos ? "0" : a.substr(first);
  }
  static std::string remainder(const std::string& a, const std::string& b) {
    std::string value = "0";
    for (const char digit : a) {
      if (value == "0") value.clear();
      value.push_back(digit);
      // Decimal long division: at most nine subtractions for each input digit.
      while (!less(value, b)) subtract(value, b);
    }
    return value;
  }
  static bool coprime(std::string a, std::string b) {
    while (b != "0") {
      auto next = remainder(a, b);
      a = std::move(b);
      b = std::move(next);
    }
    return a == "1";
  }
  std::string_view text_;
  std::size_t pos_ = 0;
};
}  // namespace physical_dimension_json_detail

inline void require_canonical_physical_dimension_json(std::string_view text) {
  physical_dimension_json_detail::Reader(text).require();
}
}  // namespace pops::identity
